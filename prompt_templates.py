"""
prompt_templates.py
Prompt builders for the Groq LLM.
"""


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
