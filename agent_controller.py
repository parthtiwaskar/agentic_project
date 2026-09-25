"""
agent_controller.py
Groq-based agentic orchestrator for the hotel booking assistant.
Uses prompt-based tool calling: Groq outputs JSON tool calls → backend executes → loop.
"""

import os
import json
import re
from typing import Any
from groq import Groq
from dotenv import load_dotenv

from hotel_api import search_hotels
from agent_tools import (
    HotelRegistry,
    tool_search_hotels,
    tool_check_availability,
    tool_get_hotel_details,
    tool_calculate_booking_cost,
    tool_check_cancellation_policy,
    tool_get_hotel_location,
    tool_create_booking,
    tool_get_booking_status,
    GROQ_TOOLS,
)
from prompt_templates import build_agent_system_prompt, build_evaluation_prompt

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

# ── Agent Status Constants ───────────────────────────────────────────────

STATUS_PLANNING = "PLANNING"
STATUS_SEARCHING = "SEARCHING"
STATUS_FILTERING = "FILTERING"
STATUS_VERIFYING = "VERIFYING"
STATUS_COMPARING = "COMPARING"
STATUS_AWAITING = "AWAITING_CONFIRMATION"
STATUS_BOOKING = "BOOKING"
STATUS_VERIFYING_BOOKING = "VERIFYING_BOOKING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

TOOL_TO_STATUS = {
    "searchHotels": STATUS_SEARCHING,
    "checkAvailability": STATUS_VERIFYING,
    "getHotelDetails": STATUS_COMPARING,
    "calculateBookingCost": STATUS_COMPARING,
    "checkCancellationPolicy": STATUS_COMPARING,
    "getHotelLocation": STATUS_COMPARING,
    "createBooking": STATUS_BOOKING,
    "getBookingStatus": STATUS_VERIFYING_BOOKING,
}

TOOL_TO_ACTIVITY = {
    "searchHotels": "Searching hotels",
    "checkAvailability": "Checking availability",
    "getHotelDetails": "Getting hotel details",
    "calculateBookingCost": "Calculating booking cost",
    "checkCancellationPolicy": "Checking cancellation policy",
    "getHotelLocation": "Checking hotel location",
    "createBooking": "Creating reservation",
    "getBookingStatus": "Verifying booking",
}


# ── Session State ────────────────────────────────────────────────────────

class AgentSession:
    """Maintains state across a conversation session."""

    def __init__(self):
        self.registry = HotelRegistry()
        self.messages: list[dict] = []
        self.state: dict[str, Any] = {
            "destination": "",
            "checkIn": "",
            "checkOut": "",
            "guests": 0,
            "budget": 0,
            "preferences": {},
            "searchResults": [],
            "selectedHotel": None,
            "selectedRoom": None,
            "bookingStatus": "idle",
            "bookingId": None,
        }
        self.activities: list[dict] = []
        self.requirements: dict = {}
        self.hotel_evaluations: list[dict] = []

        # Initialize with system prompt
        self.messages.append({
            "role": "system",
            "content": build_agent_system_prompt(),
        })

    def add_activity(self, status: str, message: str, done: bool = False) -> None:
        self.activities.append({
            "status": status,
            "message": message,
            "done": done,
        })

    def update_state(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k in self.state and v:
                self.state[k] = v

    def clear_activities(self) -> None:
        self.activities.clear()


# ── Tool Executor ────────────────────────────────────────────────────────

def _execute_tool(session: AgentSession, tool_name: str, arguments: dict) -> dict:
    """Execute a tool by name with the given arguments."""
    reg = session.registry

    if tool_name == "searchHotels":
        result = tool_search_hotels(reg, search_hotels, arguments)
        if result.get("success") and result.get("hotels"):
            session.update_state(
                destination=arguments.get("destination", ""),
                checkIn=arguments.get("checkIn", ""),
                checkOut=arguments.get("checkOut", ""),
                guests=arguments.get("guests", 0),
                budget=arguments.get("maxPrice", 0),
                searchResults=result["hotels"],
            )
            # Extract requirements for display
            session.requirements = {
                "destination": arguments.get("destination", ""),
                "checkIn": arguments.get("checkIn", ""),
                "checkOut": arguments.get("checkOut", ""),
                "guests": arguments.get("guests", 0),
                "maxPrice": arguments.get("maxPrice", 0),
                "minPrice": arguments.get("minPrice", 0),
                "stars": arguments.get("stars"),
            }
            # Merge any soft preferences from prior state
            if session.state.get("preferences"):
                session.requirements.update(session.state["preferences"])
        return result

    elif tool_name == "checkAvailability":
        return tool_check_availability(
            reg,
            hotel_id=arguments.get("hotelId", ""),
            check_in=arguments.get("checkIn", ""),
            check_out=arguments.get("checkOut", ""),
            guests=int(arguments.get("guests", 2)),
        )

    elif tool_name == "getHotelDetails":
        return tool_get_hotel_details(reg, hotel_id=arguments.get("hotelId", ""))

    elif tool_name == "calculateBookingCost":
        return tool_calculate_booking_cost(
            reg,
            hotel_id=arguments.get("hotelId", ""),
            room_type=arguments.get("roomType", ""),
            check_in=arguments.get("checkIn", ""),
            check_out=arguments.get("checkOut", ""),
        )

    elif tool_name == "checkCancellationPolicy":
        return tool_check_cancellation_policy(
            reg,
            hotel_id=arguments.get("hotelId", ""),
            room_type=arguments.get("roomType", ""),
        )

    elif tool_name == "getHotelLocation":
        return tool_get_hotel_location(reg, hotel_id=arguments.get("hotelId", ""))

    elif tool_name == "createBooking":
        result = tool_create_booking(reg, arguments)
        if result.get("success") and result.get("booked"):
            booking = result["booking"]
            session.update_state(
                bookingStatus="confirmed",
                bookingId=booking["booking_id"],
                selectedHotel=booking["hotel_id"],
                selectedRoom=booking["room_type"],
            )
        elif result.get("success") and not result.get("booked"):
            session.update_state(bookingStatus="failed")
        return result

    elif tool_name == "getBookingStatus":
        return tool_get_booking_status(reg, booking_id=arguments.get("bookingId", ""))

    else:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}


# ── Tool Call Parser ─────────────────────────────────────────────────────

def _parse_tool_call(text: str) -> tuple[str | None, dict | None, str]:
    """
    Parse a tool_call block from the model's response.
    Returns (tool_name, arguments, remaining_text) or (None, None, full_text).
    """
    # Pattern: ```tool_call\n{...}\n```
    pattern = r'```tool_call\s*\n?\s*(\{.*?\})\s*\n?\s*```'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        try:
            call = json.loads(match.group(1))
            tool_name = call.get("tool")
            arguments = call.get("arguments", {})
            remaining = text[:match.start()].strip() + text[match.end():].strip()
            return tool_name, arguments, remaining
        except json.JSONDecodeError:
            pass

    # Also try inline JSON (some models skip the backticks)
    pattern2 = r'\{"tool"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}'
    match2 = re.search(pattern2, text, re.DOTALL)
    if match2:
        try:
            tool_name = match2.group(1)
            arguments = json.loads(match2.group(2))
            remaining = text[:match2.start()].strip() + text[match2.end():].strip()
            return tool_name, arguments, remaining
        except json.JSONDecodeError:
            pass

    return None, None, text


# ── Evaluation Runner ────────────────────────────────────────────────────

def _fallback_evaluations(hotels: list[dict], reqs: dict) -> list[dict]:
    """Deterministic fallback evaluations if LLM evaluation is unavailable."""
    evals = []
    max_p = int(reqs.get("maxPrice", reqs.get("max_price", 500)) or 500)
    for h in hotels:
        price = h.get("price", 0)
        reasons = []
        tradeoffs = []
        if price <= max_p:
            reasons.append(f"Within budget at ${price}/night (target: ${max_p})")
        amenities = h.get("amenities", [])
        if "breakfast" in amenities:
            reasons.append("Complimentary breakfast included")
        if "pool" in amenities or "spa" in amenities:
            reasons.append("Premium wellness facilities (pool / spa)")
        if h.get("rating") and h.get("rating") != "N/A":
            reasons.append(f"High guest score of {h.get('rating')}/10")
        if h.get("area"):
            reasons.append(f"Central location in {h.get('area')}")

        if price > max_p * 0.85:
            tradeoffs.append(f"Premium end of requested budget (${price}/night)")
        if h.get("cancellation", {}).get("type") != "free":
            tradeoffs.append("Cancellation fees may apply (partial/non-refundable)")
        if not tradeoffs:
            tradeoffs.append("High demand property for specified dates")

        fit = "excellent" if price <= max_p and len(reasons) >= 3 else "good"
        evals.append({
            "hotel_id": h.get("hotel_id"),
            "match_reasons": reasons[:3] if reasons else ["Matches specified search criteria"],
            "tradeoffs": tradeoffs[:2],
            "overall_fit": fit,
        })
    return evals


def _evaluate_hotels(session: AgentSession) -> list[dict]:
    """Run hotel evaluation against user requirements via Groq."""
    hotels = session.state.get("searchResults", [])
    if not hotels or not session.requirements:
        return []

    try:
        prompt = build_evaluation_prompt(hotels, session.requirements)
        resp = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1500,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json\n").strip()
        evaluations = json.loads(raw)
        if isinstance(evaluations, list) and len(evaluations) > 0:
            session.hotel_evaluations = evaluations
            return evaluations
    except Exception as exc:
        print(f"[agent] evaluation note: {exc}")

    # Fallback to deterministic evaluations
    evals = _fallback_evaluations(hotels, session.requirements)
    session.hotel_evaluations = evals
    return evals


# ── Main Agent Loop ──────────────────────────────────────────────────────

MAX_TOOL_ITERATIONS = 8


def run_agent(session: AgentSession, user_message: str) -> dict[str, Any]:
    """
    Main agent entry point.
    Takes a user message, runs the Groq tool-calling loop, returns:
    {
        "response": str,           # Agent's text response to user
        "activities": list[dict],  # Activity log
        "requirements": dict,      # Extracted requirements
        "searchResults": list,     # Hotel search results (if any)
        "evaluations": list,       # Hotel evaluations (if any)
        "bookingResult": dict|None,# Booking result (if confirmed)
        "state": dict,             # Current session state
    }
    """
    session.clear_activities()
    session.add_activity(STATUS_PLANNING, "Understanding request & context", done=True)

    # Add user message to conversation
    session.messages.append({"role": "user", "content": user_message})

    final_response = ""
    booking_result = None
    search_happened = False

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            resp = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=session.messages,
                tools=GROQ_TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=2500,
            )
            msg = resp.choices[0].message
            assistant_text = (msg.content or "").strip()
            tool_calls = msg.tool_calls or []
        except Exception as exc:
            print(f"[agent] Groq call error: {exc}")
            session.add_activity(STATUS_FAILED, f"LLM note: {str(exc)[:80]}", done=True)
            final_response = "I encountered an issue contacting the AI assistant. Please try again or rephrase your request."
            break

        # Check for native tool calls first
        if tool_calls:
            # Append assistant message with tool calls as a clean dict
            session.messages.append({
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                t_name = tc.function.name
                try:
                    t_args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else (tc.function.arguments or {})
                except Exception:
                    t_args = {}

                status = TOOL_TO_STATUS.get(t_name, STATUS_PLANNING)
                activity_msg = TOOL_TO_ACTIVITY.get(t_name, f"Running {t_name}")
                if t_name == "searchHotels":
                    activity_msg = f"Searching hotels in {t_args.get('destination', 'destination')}"
                elif t_name == "checkAvailability":
                    activity_msg = f"Checking availability for hotel {t_args.get('hotelId', '')}"
                elif t_name == "createBooking":
                    activity_msg = "Finalizing hotel reservation"

                session.add_activity(status, activity_msg, done=False)
                tool_result = _execute_tool(session, t_name, t_args)

                if session.activities:
                    session.activities[-1]["done"] = True

                if t_name == "searchHotels" and tool_result.get("success"):
                    search_happened = True
                    count = tool_result.get("count", 0)
                    dest = tool_result.get("destination", "")
                    session.add_activity(STATUS_FILTERING, f"Found {count} hotels matching criteria in {dest}", done=True)

                if t_name == "createBooking":
                    if tool_result.get("booked"):
                        booking_result = tool_result.get("booking")
                        session.add_activity(STATUS_COMPLETED, f"Booking confirmed: {booking_result.get('booking_id')}", done=True)
                    else:
                        session.add_activity(STATUS_FAILED, tool_result.get("message", "Booking failed"), done=True)

                # Append tool response
                session.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result),
                })

            # Continue loop so model generates a response with the tool results
            continue

        # Fallback: check if model produced a prompt-based tool call in text
        t_name, t_args, remaining_text = _parse_tool_call(assistant_text)
        if t_name and t_args:
            status = TOOL_TO_STATUS.get(t_name, STATUS_PLANNING)
            activity_msg = TOOL_TO_ACTIVITY.get(t_name, f"Running {t_name}")
            session.add_activity(status, activity_msg, done=False)

            tool_result = _execute_tool(session, t_name, t_args)
            if session.activities:
                session.activities[-1]["done"] = True

            if t_name == "searchHotels" and tool_result.get("success"):
                search_happened = True
                count = tool_result.get("count", 0)
                dest = tool_result.get("destination", "")
                session.add_activity(STATUS_FILTERING, f"Found {count} hotels in {dest}", done=True)

            if t_name == "createBooking":
                if tool_result.get("booked"):
                    booking_result = tool_result.get("booking")
                    session.add_activity(STATUS_COMPLETED, "Booking confirmed", done=True)
                else:
                    session.add_activity(STATUS_FAILED, tool_result.get("message", "Booking failed"), done=True)

            session.messages.append({"role": "assistant", "content": assistant_text})
            session.messages.append({
                "role": "user",
                "content": f"[Tool Result for {t_name}]:\n{json.dumps(tool_result, indent=2)}",
            })
            if remaining_text:
                final_response = remaining_text
            continue

        # Normal text response (no tool calls)
        final_response = assistant_text
        session.messages.append({"role": "assistant", "content": assistant_text})
        break

    # Run evaluations if we got search results
    evaluations = []
    if search_happened and session.state.get("searchResults"):
        session.add_activity(STATUS_COMPARING, "Evaluating hotels against your constraints & trade-offs", done=False)
        evaluations = _evaluate_hotels(session)
        if session.activities:
            session.activities[-1]["done"] = True
        session.add_activity(STATUS_COMPLETED, "Recommendations ready on Jira board", done=True)

    return {
        "response": final_response,
        "activities": list(session.activities),
        "requirements": dict(session.requirements),
        "searchResults": list(session.state.get("searchResults", [])),
        "evaluations": evaluations,
        "bookingResult": booking_result,
        "state": dict(session.state),
    }

