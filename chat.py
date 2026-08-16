# chat.py
import os
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI

from medical_search import fetch_dynamic_disease_info

router = APIRouter()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = ""


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    scan_context: Optional[dict] = None


# Define Tools for General Intake, Medical Knowledge Lookup, and Doctor Handoff
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_medical_knowledge",
            "description": "Fetch clinical information for skin diseases, general conditions (e.g., Malaria, Typhoid, Gastroenteritis), or symptoms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition_name": {
                        "type": "string",
                        "description": "The medical term or disease name to research."
                    }
                },
                "required": ["condition_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_doctor_intake_summary",
            "description": "Generate a structured intake note containing chief complaint, duration, severity, and suspected condition to transfer to a consulting physician.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chief_complaint": {"type": "string", "description": "Primary issue reported by patient (e.g. High fever, chills, rash)."},
                    "symptom_duration": {"type": "string", "description": "How long the patient has felt sick."},
                    "severity_scale": {"type": "string", "enum": ["LOW", "MODERATE", "HIGH", "EMERGENCY"]},
                    "suspected_category": {"type": "string", "description": "General medicine, Dermatology, Pediatrics, etc."},
                    "summary_notes": {"type": "string", "description": "Brief clinical summary for the doctor review."}
                },
                "required": ["chief_complaint", "symptom_duration", "severity_scale", "suspected_category", "summary_notes"]
            }
        }
    }
]


def execute_tool(name: str, arguments: dict) -> str:
    try:
        if name == "lookup_medical_knowledge":
            condition = arguments.get("condition_name", "")
            return json.dumps(fetch_dynamic_disease_info(condition))

        elif name == "generate_doctor_intake_summary":
            # Formats a summary payload ready for doctor handoff or database persistence
            return json.dumps({
                "status": "SUMMARY_GENERATED",
                "summary": arguments,
                "action": "Ready to transfer to a doctor or present 'Connect to Doctor' booking option."
            })

        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return json.dumps({"error": "Tool execution failed", "details": str(e)})


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

        formatted_messages = [{"role": "system", "content": system_instruction}]
        for msg in payload.messages:
            formatted_messages.append({"role": msg.role, "content": msg.content or ""})

        MAX_TOOL_ITERATIONS = 3
        iterations = 0

        # Captures the structured intake payload if the agent decides to
        # generate one during this turn, so it can be pushed to the frontend
        # as a distinct SSE event (separate from plain text token frames).
        captured_intake_summary = None

        while iterations < MAX_TOOL_ITERATIONS:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=formatted_messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if not tool_calls:
                break

            formatted_messages.append(response_message)

            for tool_call in tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments or "{}")
                tool_result = execute_tool(fn_name, fn_args)

                if fn_name == "generate_doctor_intake_summary":
                    captured_intake_summary = fn_args

                formatted_messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": tool_result
                })

            iterations += 1

        async def generate_stream():
            # Send the structured intake summary FIRST (if one was generated
            # this turn) so the frontend can render a "ready to transfer"
            # card before the assistant's closing message even finishes
            # streaming.
            if captured_intake_summary is not None:
                yield f"data: {json.dumps({'type': 'intake_summary', 'summary': captured_intake_summary})}\n\n"

            stream_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=formatted_messages,
                stream=True,
                temperature=0.4
            )

            for chunk in stream_response:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent intake error: {str(e)}")
