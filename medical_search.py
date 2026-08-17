import os
import time
import requests
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List

# Initialize Gemini Client with automatic HTTP retries for 503/429 status codes
gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=3,
            initial_delay=1.0,
            max_delay=5.0,
            http_status_codes=[408, 429, 500, 502, 503, 504]
        )
    )
)


class MedicalInsightSchema(BaseModel):
    meaning: str
    causes: List[str]
    consequences_and_effects: List[str]
    solutions_and_remedies: List[str]
    recommended_action: str
    doctor_consultation_advice: str


def fetch_wikipedia_summary(condition_name: str) -> str:
    """Fetches text summary from Wikipedia REST API for any skin condition."""
    try:
        # Strip trailing slashes, parentheses, and extra spaces for clean Wiki queries
        clean_name = condition_name.split("/")[0].split("(")[0].strip()
        formatted_query = clean_name.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_query}"

        headers = {"User-Agent": "PocketDocAI/1.0 (contact@pocketdoc.ai)"}
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()
            return data.get("extract", "")
        return ""
    except Exception as e:
        print(f"Wikipedia fetch error for {condition_name}: {e}")
        return ""


def fetch_dynamic_disease_info(condition_name: str) -> dict:
    """
    Dynamically fetches clinical context for ANY predicted disease 
    and uses Gemini to parse it into structured JSON with built-in model fallbacks.
    """
    wiki_summary = fetch_wikipedia_summary(condition_name)

    system_prompt = f"""
    You are PocketDoc AI, a medical information parser.
    Your job is to structure detailed medical guidance for the condition: '{condition_name}'.
    
    Use the provided Wikipedia context if available:
    Context: "{wiki_summary}"

    Requirements:
    - Fill out all fields accurately based on established medical knowledge.
    - 'meaning': Clear 1-3 sentence explanation of what the condition is.
    - 'causes': 2-5 primary causes or triggers.
    - 'consequences_and_effects': 2-4 symptoms or potential complications.
    - 'solutions_and_remedies': 2-3 safe home care remedies or standard treatments.
    - 'recommended_action': Immediate self-care step.
    - 'doctor_consultation_advice': Clear advice on when to schedule an in-app doctor consultation.
    """

    candidate_models = [
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
    ]

    for model_name in candidate_models:
        try:
            print(f"Attempting medical info generation using model: {model_name}...")
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=f"Provide structured insights for {condition_name}.",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=MedicalInsightSchema,
                    temperature=0.2,
                ),
            )

            # Parse and return as dictionary
            structured_data = MedicalInsightSchema.model_validate_json(response.text)
            return structured_data.model_dump()

        except Exception as e:
            print(f"Model {model_name} failed for {condition_name}: {e}. Retrying with next model...")
            time.sleep(0.5)

    # Safe Fallback if all API attempts fail
    return {
        "meaning": f"{condition_name} is a dermatological condition identified by the AI visual scan.",
        "causes": ["Clinical evaluation required for precise cause determination."],
        "consequences_and_effects": ["Consult a certified doctor or dermatologist for a proper diagnosis."],
        "solutions_and_remedies": ["Again consult a healthcare professional for appropriate treatment."],
        "recommended_action": "Seek immediate medical attention.",
        "doctor_consultation_advice": "Consult a certified doctor or dermatologist via video call for diagnosis."
    }