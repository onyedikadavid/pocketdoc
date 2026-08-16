import os
import requests
from openai import OpenAI
from pydantic import BaseModel
from typing import List

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        formatted_query = condition_name.strip().replace(" ", "_")
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
    and uses an LLM to parse it into structured JSON.
    """
    try:
        # 1. Fetch live summary from Wikipedia
        wiki_summary = fetch_wikipedia_summary(condition_name)

        # 2. Instruct LLM to analyze and structure the medical context
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

        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Provide structured insights for {condition_name}."}
            ],
            response_format=MedicalInsightSchema,
        )

        return completion.choices[0].message.parsed.model_dump()

    except Exception as e:
        print(f"Dynamic medical parsing failed for {condition_name}: {e}")
        # Fallback response
        return {
            "meaning": f"{condition_name} is a dermatological condition identified by the AI visual scan.",
            "causes": ["Clinical evaluation required for precise cause determination."],
            "consequences_and_effects": ["consult a certified doctor or dermatologist for a proper diagnosis."],
            "solutions_and_remedies": [" Again consult a healthcare professional for appropriate treatment."],
            "recommended_action": "Seek immediate medical attention.",
            "doctor_consultation_advice": "Consult a certified doctor or dermatologist via video call for diagnosis."
        }