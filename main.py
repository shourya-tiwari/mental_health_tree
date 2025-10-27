import os
import json
from datetime import datetime
from google import genai
from google.genai import types
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load Gemini API key from environment variable for security
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=API_KEY)
DATA_FILE = "mental_health_ai_data.json"

# =============================================================================
# DATA MODELS
# =============================================================================

class CheckinRequest(BaseModel):
    """Structure for incoming check-in data from frontend"""
    mood: int        # Mood rating (1-5)
    anxiety: int     # Anxiety level (1-5, inverted)
    motivation: int  # Motivation level (1-5)
    connection: int  # Social connection (1-5)
    free_text: str   # User's written thoughts

class CheckinResponse(BaseModel):
    """Structure for response data sent to frontend"""
    mood_rating: str
    feedback: str
    new_health_score: int
    is_emergency: bool = False

# =============================================================================
# FASTAPI SETUP
# =============================================================================

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the 'images' folder to be accessible at '/images'
app.mount("/images", StaticFiles(directory="images"), name="images")

# Create a route for the signup page (new default)
@app.get("/", response_class=FileResponse)
async def read_signup():
    return FileResponse('signup.html')

# Create a route for the main app page
@app.get("/app", response_class=FileResponse)
async def read_app():
    return FileResponse('index.html')

# Create a route for the results page
@app.get("/results", response_class=FileResponse)
async def read_results():
    return FileResponse('results.html')

# Create a route for the login page
@app.get("/login", response_class=FileResponse)
async def read_login():
    return FileResponse('login.html')

# =============================================================================
# DATA MANAGEMENT FUNCTIONS
# =============================================================================

def load_data():
    """
    Loads user data from JSON file or creates default structure for new users.
    Returns: dict with 'entries' list and 'current_health' starting at 50
    """
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Handle corrupted file by creating new one
            pass
    
    # Default structure for new users
    return {"entries": [], "current_health": 50}

def save_data(data):
    """Saves user data to JSON file with proper formatting"""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# =============================================================================
# AI ANALYSIS FUNCTIONS
# =============================================================================

def analyze_text(text):
    """
    Analyzes sentiment of user's free text using Google Gemini API.
    Returns: "positive", "negative", or "neutral"
    """
    if not text.strip():
        return "neutral"

    system_instruction = "Analyze the user's text sentiment and respond with one word ONLY: positive, neutral, or negative."

    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0
        )
    )
    sentiment = response.text.strip().lower()

    if "positive" in sentiment:
        return "positive"
    elif "negative" in sentiment:
        return "negative"
    else:
        return "neutral"

def get_mood_rating(mcq_score, sentiment):
    """
    Calculates overall mood rating based on MCQ score and sentiment.
    
    Scoring Logic:
    - Base score = MCQ score (sum of 4 sliders)
    - Sentiment bonus: +2 for positive, -2 for negative, 0 for neutral
    - Mood rating: Good (≥21), Moderate (13-20), Low (<13)
    """
    base_score = mcq_score
    if sentiment == "positive":
        base_score += 2
    elif sentiment == "negative":
        base_score -= 2

    if base_score >= 21:
        return "Good"
    elif base_score >= 13:
        return "Moderate"
    else:
        return "Low"

def generate_feedback(mcq_score, sentiment, free_text):
    """
    Generates personalized AI feedback based on user's check-in data.
    Returns compassionate, supportive message tailored to their mood.
    """
    mood_rating = get_mood_rating(mcq_score, sentiment)

    prompt = f"""
    Act as a compassionate and supportive mental health companion.
    A user has just completed their daily check-in. Here is their data:
    - MCQ Score: {mcq_score}/20
    - Free Text Sentiment: {sentiment}
    - Their private thoughts: "{free_text}"
    - Their calculated overall mood: "{mood_rating}"

    Please write a thoughtful, personalized feedback message of about 150-200 words.
    Speak directly to the user in a warm and understanding tone.

    - If the mood is "Low", be extra supportive. Validate their feelings, show empathy for what they wrote in their thoughts, and perhaps offer one gentle, simple suggestion (like getting some fresh air, listening to a favorite song, or just being kind to themselves).
    - If the mood is "Moderate", acknowledge their day. Reflect on their free text and encourage them to keep going.
    - If the mood is "Good", celebrate this with them! Reinforce their positive state and encourage them to savor it.
    
    Do not sound like a robot. Be human and supportive.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt,
        )
        
        feedback = response.text.strip()
        
        if not feedback:
            return "Thank you for checking in. Your entry has been logged."
            
        return feedback

    except Exception as e:
        return f"Sorry, there was an error generating feedback. Your check-in is still saved. (Error: {e})"

# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.post("/checkin", response_model=CheckinResponse)
def create_checkin(request: CheckinRequest):
    """
    Processes daily check-in data and returns AI feedback.
    
    Scoring Process:
    1. Calculate MCQ score (sum of 4 sliders)
    2. Analyze free text sentiment
    3. Determine mood rating (Good/Moderate/Low)
    4. Generate AI feedback
    5. Update tree health score
    6. Save entry to JSON file
    """
    # Calculate MCQ score from slider values
    mcq_score = request.mood + request.anxiety + request.motivation + request.connection

    # Analyze user's free text and determine mood
    sentiment = analyze_text(request.free_text)
    mood_rating = get_mood_rating(mcq_score, sentiment)
    feedback = generate_feedback(mcq_score, sentiment, request.free_text)

    # Load current data and update health score
    data = load_data()
    current_health = data.get("current_health", 50)

    # Update health based on mood rating
    if mood_rating == "Good":
        current_health += 5
    elif mood_rating == "Moderate":
        current_health += 1
    elif mood_rating == "Low":
        current_health -= 5

    # Cap health score between 0 and 100
    current_health = max(0, min(100, current_health))

    # Emergency detection logic
    is_emergency = False
    if "die" in request.free_text.lower():
        is_emergency = True
        current_health = 7

    # Create new entry and save data
    today_str = datetime.now().strftime("%Y-%m-%d")
    new_entry = {
        "date": today_str,
        "mcq_score": mcq_score,
        "sentiment": sentiment,
        "mood": mood_rating,
        "free_text": request.free_text,
        "feedback": feedback
    }

    data["entries"].append(new_entry)
    data["current_health"] = current_health
    save_data(data)

    return CheckinResponse(
        mood_rating=mood_rating,
        feedback=feedback,
        new_health_score=current_health,
        is_emergency=is_emergency
    )

@app.get("/tree-health")
def get_tree_health():
    """
    Returns current tree health score and corresponding image file.
    Used by frontend to display tree visualization.
    """
    data = load_data()
    health = data.get("current_health", 50)

    # Determine tree image based on health score
    if health > 80:
        image_file = "tree-flourishing.png"
    elif health > 60:
        image_file = "tree-healthy.png"
    elif health > 40:
        image_file = "tree-neutral.png"
    elif health > 20:
        image_file = "tree-wilting.png"
    else:
        image_file = "tree-rotting.png"

    return {"health_score": health, "image_file": image_file}