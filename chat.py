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

# Initialize Gemini Client with auto HTTP retries to eliminate transient 503 errors
gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=3,
            initial_delay=0.5,
            max_delay=3.0,
            http_status_codes=[408, 429, 500, 502, 503, 504]
        )
    )
)


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
            "chief_complaint": types.Schema(type=types.Type.STRING, description="Primary issue reported by patient."),
            "symptom_duration": types.Schema(type=types.Type.STRING, description="How long the patient has felt sick."),
            "severity_scale": types.Schema(
                type=types.Type.STRING, 
                enum=["LOW", "MODERATE", "HIGH", "EMERGENCY"],
                description="Patient severity level"
            ),
            "suspected_category": types.Schema(type=types.Type.STRING, description="General medicine, Dermatology, Pediatrics, etc."),
            "summary_notes": types.Schema(type=types.Type.STRING, description="Brief clinical summary for doctor review.")
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
1. Skin AI Diagnostic Scanning.
2. General Pre-Consultation Intake (For any health condition: Malaria, Typhoid, Flu, Stomach pain, etc.).
3. Virtual Doctor Consultations.
4. E-Pharmacy Medication Orders.

Your Goal:
1. Gather the patient's symptoms (Chief complaint, duration, severity, fever, pain level 1-10, medication history).
2. If the user attached a Skin Scan Result, incorporate that context into your conversation.
3. If the user has non-skin concerns, triage them effectively.
4. When you have collected enough symptom detail, call `generate_doctor_intake_summary` so the patient can seamlessly book and transfer records to a physician.

Guardrails:
- Offer preliminary clinical guidance, NOT a final diagnosis.
- For emergency signs, advise immediate emergency care.
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

        # Determine if the message requires agentic tool evaluation
        latest_message = payload.messages[-1].content if payload.messages else ""
        requires_tools = len(latest_message.split()) > 4  # Longer symptom reports warrant tool evaluation

        if requires_tools:
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=AGENT_TOOLS,
                    temperature=0.3
                )

                response = gemini_client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=contents,
                    config=config
                )

                if response.function_calls:
                    for call in response.function_calls:
                        fn_name = call.name
                        fn_args = call.args or {}
                        tool_result = execute_tool(fn_name, fn_args)

                        if fn_name == "generate_doctor_intake_summary":
                            captured_intake_summary = fn_args

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
            except Exception as tool_err:
                print(f"Tool check bypassed due to transient error: {tool_err}")

        async def generate_stream():
            if captured_intake_summary is not None:
                yield f"data: {json.dumps({'type': 'intake_summary', 'summary': captured_intake_summary})}\n\n"

            stream_config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3
            )

            # High-speed streaming response
            response_stream = gemini_client.models.generate_content_stream(
                model="gemini-3.5-flash-lite",
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