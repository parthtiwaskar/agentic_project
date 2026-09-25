"""
prompt_templates.py
Prompt builders for the Groq LLM.
Extended with agentic system prompts and evaluation prompts.
"""

import json
from agent_tools import TOOL_DEFINITIONS


def build_search_prompt(
    destination: str,
    check_in:    str,
    check_out:   str,
    min_price:   int,
    max_price:   int,
    adults:      int,
    stars:       int | None,
) -> str:
    """
    Returns a system + user message pair that instructs the Groq model to
    output a strict JSON object we can pass directly to search_hotels().
    """
    star_clause = f" Only {stars}-star hotels." if stars else ""
    return (
        "You are a hotel‑booking assistant. Given the user's search criteria, "
        "output ONLY a valid JSON object (no markdown, no extra text) with these keys:\n"
        "  destination (string), check_in (YYYY-MM-DD), check_out (YYYY-MM-DD),\n"
        "  min_price (integer USD), max_price (integer USD), adults (integer),\n"
        "  stars (integer or null).\n\n"
        f"User criteria: destination={destination}, check_in={check_in}, "
        f"check_out={check_out}, min_price={min_price}, max_price={max_price}, "
        f"adults={adults}, stars={'null' if stars is None else stars}.{star_clause}"
    )


def build_summary_prompt(hotels: list[dict]) -> str:
    """
    Ask the LLM to produce a natural-language recommendation
    from the raw list of hotel dicts.
    """
    lines = "\n".join(
        f"- {h['name']} | ⭐ {h['stars']} | ${h['price']}/night | "
        f"Rating: {h['rating']} | {h['address']}"
        for h in hotels
    )
    return (
        "You are a friendly hotel‑booking assistant. "
        "Below is a list of available hotels. "
        "Write a concise, helpful recommendation (3–5 sentences) highlighting "
        "the best value option and the most luxurious option. "
        "Keep the tone warm and professional.\n\n"
        f"Hotels:\n{lines}"
    )


# ── Agentic System Prompt ────────────────────────────────────────────────

def build_agent_system_prompt() -> str:
    """
    System prompt for the Groq-based agentic hotel booking assistant.
    Supports native tool calling and function definitions.
    """
    return """You are an intelligent, proactive AI Hotel Booking Agent. You help users search, compare, evaluate, and book hotels through natural conversation.

## Core Rules & Behavior

1. PROACTIVE SEARCH:
   - When the user provides a destination and dates (or destination alone with implied dates/guests), call `searchHotels` immediately!
   - If the user provides a budget like "$200" or "under 200", map that directly to `maxPrice`.
   - If `maxPrice` is not provided, use a reasonable default of 400. If `guests` is not provided, default to 2.
   - Only ask for missing info if the destination is completely missing.

2. PRESENT RESULTS WITH REASONING:
   - After receiving results from `searchHotels`, present them clearly.
   - For each hotel, explicitly explain WHY it matches the user's criteria (location, amenities, budget) and any TRADE-OFFS (price vs location, cancellation rules).

3. HARD vs SOFT CONSTRAINTS:
   - HARD: Destination, check-in, check-out, guest count, price ceiling — must strictly satisfy.
   - SOFT: Star rating, amenities (breakfast, pool, wifi), area/landmark — prioritize and evaluate fit.

4. BOOKING WORKFLOW:
   - When a user selects a hotel to book (e.g., "Book the Grand Tokyo" or "I want hotel 1"):
     Step 1: Check availability using `checkAvailability`.
     Step 2: Calculate total cost using `calculateBookingCost`.
     Step 3: Present the booking summary (nights, room type, total cost with 12% tax, cancellation policy) and ask for confirmation.
     Step 4: Once confirmed, call `createBooking` to generate the booking confirmation.
     Step 5: Call `getBookingStatus` to verify and display the confirmed booking ID.

5. CONTEXT & FOLLOW-UPS:
   - Remember previous conversation state. When the user says "Show cheaper options", "Only 5-star", or "Closer to city centre", reuse the existing destination and dates, updating only the requested criteria.

6. ACCURACY:
   - Never invent hotel names, IDs, or rates. Only use real hotels and data returned by the tools.
   - Be helpful, concise, and professional."""



# ── Evaluation Prompt ────────────────────────────────────────────────────

def build_evaluation_prompt(
    hotels: list[dict],
    requirements: dict,
) -> str:
    """
    Ask the LLM to evaluate hotels against user requirements.
    Returns structured match reasons and trade-offs.
    """
    hotel_lines = json.dumps(hotels, indent=2)
    req_lines = json.dumps(requirements, indent=2)

    return f"""You are a hotel evaluation assistant. Evaluate each hotel against the user's requirements.

User Requirements:
{req_lines}

Hotels:
{hotel_lines}

For each hotel, output a JSON array with this structure (no markdown, no extra text):
[
  {{
    "hotel_id": "...",
    "match_reasons": ["Under budget at $X/night", "Has breakfast", "Near requested area"],
    "tradeoffs": ["$X more than cheapest option but better location", "No pool available"],
    "overall_fit": "excellent" | "good" | "fair" | "poor"
  }}
]

Rules:
- match_reasons: list what MATCHES the user's requirements
- tradeoffs: list any compromises the user should know about
- overall_fit: based on how many hard+soft constraints are met
- Be specific with numbers and names, not generic
- Output ONLY valid JSON"""
