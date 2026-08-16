import os
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from google import genai
from google.genai import types

from medical_search import fetch_dynamic_disease_info

router = APIRouter()

# Initialize Gemini Client (reads GEMINI_API_KEY from environment)
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = ""


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    scan_context: Optional[dict] = None


# Define Tool Declarations using google-genai Types
lookup_medical_tool = types.FunctionDeclaration(
    name="lookup_medical_knowledge",
    description="Fetch clinical information for skin diseases, general conditions (e.g., Malaria, Typhoid, Gastroenteritis), or symptoms.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "condition_name": types.Schema(
                type=types.Type.STRING,
                description="The medical term or disease name to research."
            )
        },
        required=["condition_name"]
    )
)

intake_summary_tool = types.FunctionDeclaration(
    name="generate_doctor_intake_summary",
    description="Generate a structured intake note containing chief complaint, duration, severity, and suspected condition to transfer to a consulting physician.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "chief_complaint": types.Schema(type=types.Type.STRING, description="Primary issue reported by patient (e.g. High fever, chills, rash)."),
            "symptom_duration": types.Schema(type=types.Type.STRING, description="How long the patient has felt sick."),
            "severity_scale": types.Schema(
                type=types.Type.STRING, 
                enum=["LOW", "MODERATE", "HIGH", "EMERGENCY"],
                description="Patient severity level"
            ),
            "suspected_category": types.Schema(type=types.Type.STRING, description="General medicine, Dermatology, Pediatrics, etc."),
            "summary_notes": types.Schema(type=types.Type.STRING, description="Brief clinical summary for the doctor review.")
        },
        required=["chief_complaint", "symptom_duration", "severity_scale", "suspected_category", "summary_notes"]
    )
)

AGENT_TOOLS = [types.Tool(function_declarations=[lookup_medical_tool, intake_summary_tool])]


def execute_tool(name: str, arguments: dict) -> dict:
    try:
        if name == "lookup_medical_knowledge":
            condition = arguments.get("condition_name", "")
            return fetch_dynamic_disease_info(condition)

        elif name == "generate_doctor_intake_summary":
            return {
                "status": "SUMMARY_GENERATED",
                "summary": arguments,
                "action": "Ready to transfer to a doctor or present 'Connect to Doctor' booking option."
            }

        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": "Tool execution failed", "details": str(e)}


AGENT_SYSTEM_PROMPT = """
You are PocketDoc AI, an empathetic, clinical pre-consultation assistant for the PocketDoc Telehealth Platform.

Platform Overview:
PocketDoc provides full healthcare services:
1. Skin AI Diagnostic Scanning (Optional - used if patient has a skin lesion/rash).
2. General Pre-Consultation Intake (For any health condition: Malaria, Typhoid, Flu, Stomach pain, headaches, etc.).
3. Virtual Doctor Consultations.
4. E-Pharmacy Medication Orders & Home Delivery.

Your Goal:
1. Gather the patient's symptoms (Chief complaint, duration, severity, fever, pain level 1-10, medication history).
2. If the user attached a Skin Scan Result, incorporate that context into your conversation.
3. If the user has non-skin concerns (e.g., fever, malaria symptoms, gastro issue), triage them effectively.
4. When you have collected enough symptom detail, call `generate_doctor_intake_summary` so the patient can seamlessly book and transfer their records to a certified physician.

Guardrails:
- You offer preliminary clinical guidance, NOT a final diagnosis.
- For emergency signs (e.g., severe chest pain, difficulty breathing, unresponsiveness), advise immediate emergency care.
- Keep output empathetic, concise, and structured.

Current Patient Context:
{scan_info}
"""


@router.post("/stream")
async def stream_agentic_chat(payload: ChatRequest):
    try:
        ctx = payload.scan_context or {}
        if ctx and "top_prediction" in ctx:
            top_pred = ctx.get("top_prediction", {})
            scan_info = (
                f"- Skin Scan Prediction: {top_pred.get('class_name', 'Unknown')}\n"
                f"- Model Confidence: {float(top_pred.get('confidence', 0)) * 100:.1f}%\n"
                f"- Triage Level: {ctx.get('triage_level', 'UNKNOWN')}\n"
            )
        else:
            scan_info = "No skin image scan attached. Patient is using general symptom consultation."

        system_instruction = AGENT_SYSTEM_PROMPT.format(scan_info=scan_info)

        # Convert input message history into Gemini Content format
        contents = []
        for msg in payload.messages:
            role = "user" if msg.role in ["user", "system"] else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.content or "")]
                )
            )

        captured_intake_summary = None
        MAX_TOOL_ITERATIONS = 3
        iterations = 0

        # Handle tool calls in non-streaming loop first
        while iterations < MAX_TOOL_ITERATIONS:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=AGENT_TOOLS,
                temperature=0.3
            )

            response = gemini_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=contents,
                config=config
            )

            # Check if Gemini triggered tool calls
            function_calls = response.function_calls
            if not function_calls:
                # No more tools requested, ready to stream final response
                break

            # Process function calls
            for call in function_calls:
                fn_name = call.name
                fn_args = call.args or {}
                tool_result = execute_tool(fn_name, fn_args)

                if fn_name == "generate_doctor_intake_summary":
                    captured_intake_summary = fn_args

                # Append model choice and function response to history
                contents.append(response.candidates[0].content)
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=fn_name,
                                response={"result": tool_result}
                            )
                        ]
                    )
                )

            iterations += 1

        async def generate_stream():
            # If an intake summary was created, send it to the frontend first
            if captured_intake_summary is not None:
                yield f"data: {json.dumps({'type': 'intake_summary', 'summary': captured_intake_summary})}\n\n"

            # Stream final assistant text response back to the client
            stream_config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4
            )

            response_stream = gemini_client.models.generate_content_stream(
                model="gemini-1.5-flash",
                contents=contents,
                config=stream_config
            )

            for chunk in response_stream:
                if chunk.text:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.text})}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent intake error: {str(e)}")