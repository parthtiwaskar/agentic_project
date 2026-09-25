"""
app.py  –  Hotel Booking Assistant (Jira / Partian Enterprise Edition)
──────────────────────────────────────────────────────────────────────────
Gradio 6 + Groq AI (gpt-oss-120b) + RapidAPI Hotels with Jira UI System & Agent Tools
"""

import os
import json
from datetime import datetime
from typing import Any, Optional
import pandas as pd
import gradio as gr
from fastapi import FastAPI
from gradio.themes import Base
from dotenv import load_dotenv
from groq import Groq

from hotel_api import search_hotels
from prompt_templates import build_search_prompt, build_summary_prompt
from agent_tools import (
    HotelRegistry,
    tool_calculate_booking_cost,
    tool_check_cancellation_policy,
    tool_get_hotel_details,
    tool_create_booking,
    tool_get_booking_status,
)
from agent_controller import AgentSession, run_agent, _fallback_evaluations

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
CSS_PATH = os.path.join(os.path.dirname(__file__), "ui.css")
css = open(CSS_PATH).read() if os.path.exists(CSS_PATH) else ""


# ─── Jira / Partian Design System Theme ───────────────────────────────────
jira_theme = Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    # ── Dark Backgrounds ──────────────────────────────────────────────
    background_fill_primary="#0A0E1A",
    background_fill_secondary="#111827",
    # Blocks
    block_background_fill="#111827",
    block_border_width="1px",
    block_border_color="rgba(255,255,255,0.08)",
    block_shadow="0 2px 8px rgba(0,0,0,0.5)",
    block_radius="10px",
    block_padding="18px 20px",
    block_label_background_fill="transparent",
    block_label_text_color="#8B96A8",
    block_label_text_size="xs",
    block_label_text_weight="700",
    # Inputs
    input_background_fill="#1A2235",
    input_border_color="rgba(255,255,255,0.08)",
    input_border_width="1.5px",
    input_border_color_focus="#3B82F6",
    input_shadow_focus="0 0 0 3px rgba(37,99,235,0.3)",
    input_radius="8px",
    input_text_size="sm",
    # Primary Button
    button_primary_background_fill="#2563EB",
    button_primary_background_fill_hover="#1D4ED8",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="transparent",
    button_primary_shadow="0 3px 12px rgba(37,99,235,0.35)",
    button_large_radius="8px",
    button_large_padding="11px 22px",
    button_large_text_size="sm",
    button_large_text_weight="700",
    # Secondary Button
    button_secondary_background_fill="#1A2235",
    button_secondary_background_fill_hover="#243048",
    button_secondary_text_color="#E8EEF7",
    button_secondary_border_color="rgba(255,255,255,0.08)",
    # Text
    body_text_color="#E8EEF7",
    body_text_color_subdued="#8B96A8",
    body_text_size="sm",
    # Table
    table_even_background_fill="#131B2E",
    table_odd_background_fill="#111827",
    table_border_color="rgba(255,255,255,0.06)",
    # Panel / Section
    panel_background_fill="#111827",
    panel_border_color="rgba(255,255,255,0.08)",
    # Checkbox / Radio
    checkbox_background_color="#1A2235",
    checkbox_border_color="rgba(255,255,255,0.15)",
)


# ─── Render Helpers ─────────────────────────────────────────────────────────

def _render_agent_activities(activities: list[dict]) -> str:
    """Render the step-by-step activity panel with animated indicators."""
    if not activities:
        return ""

    steps_html = []
    for a in activities:
        done = a.get("done", False)
        msg = a.get("message", "")
        step_class = "agent-step-done" if done else "agent-step-active"
        icon = "✓" if done else "●"
        steps_html.append(f"""
        <div class="agent-activity-step {step_class}">
            <span class="agent-step-icon">{icon}</span>
            <span>{msg}</span>
        </div>
        """)

    return f"""
    <div class="agent-activity-panel">
        <div class="agent-activity-header">
            <span class="agent-activity-dot"></span>
            <span>Agent Execution & Verification Activity</span>
        </div>
        <div class="agent-activity-steps">
            {''.join(steps_html)}
        </div>
    </div>
    """


def _render_agent_requirements(req: dict) -> str:
    """Render the extracted travel constraints grid."""
    if not req or not any(req.values()):
        return ""

    items_html = []
    dest = req.get("destination")
    if dest:
        items_html.append(f"""
        <div class="agent-req-item">
            <span class="agent-req-label">DESTINATION</span>
            <span class="agent-req-value">{dest.title()}</span>
        </div>
        """)

    cin = req.get("checkIn") or req.get("check_in")
    cout = req.get("checkOut") or req.get("check_out")
    if cin and cout:
        items_html.append(f"""
        <div class="agent-req-item">
            <span class="agent-req-label">DATES</span>
            <span class="agent-req-value">{cin} → {cout}</span>
        </div>
        """)

    guests = req.get("guests") or req.get("adults")
    if guests:
        items_html.append(f"""
        <div class="agent-req-item">
            <span class="agent-req-label">GUESTS</span>
            <span class="agent-req-value">{guests} Adults</span>
        </div>
        """)

    max_p = req.get("maxPrice") or req.get("max_price")
    if max_p:
        items_html.append(f"""
        <div class="agent-req-item">
            <span class="agent-req-label">BUDGET CEILING</span>
            <span class="agent-req-value">${max_p}/night</span>
        </div>
        """)

    stars_val = req.get("stars")
    if stars_val is not None and str(stars_val) not in ("Any", "None", ""):
        s_int = int(stars_val)
        items_html.append(f"""
        <div class="agent-req-item">
            <span class="agent-req-label">STAR TIER</span>
            <span class="agent-req-value">{'★' * s_int} ({s_int}-Star)</span>
        </div>
        """)

    if not items_html:
        return ""

    return f"""
    <div class="agent-requirements-grid">
        <div class="agent-requirements-title">
            <span>📋</span>
            <span>Extracted Constraints & Preferences</span>
        </div>
        <div class="agent-req-items">
            {''.join(items_html)}
        </div>
    </div>
    """


def _render_conversation_history(history: list[dict]) -> str:
    """Render interactive conversation history for the agent chat area."""
    if not history:
        return """
        <div class="agent-conversation">
            <div class="agent-msg-bot">
                <strong>🤖 Partian Hotel Concierge Agent</strong><br>
                Welcome! Tell me where you'd like to travel, your dates, guest count, and any budget or amenities preferences. I will autonomously parse your request, search the live inventory, analyze trade-offs, and assist with your booking!
            </div>
        </div>
        """
    msgs = []
    for item in history:
        sender = item.get("sender", "bot")
        text = item.get("text", "")
        cls = "agent-msg-user" if sender == "user" else "agent-msg-bot"
        prefix = "<strong>👤 You</strong><br>" if sender == "user" else "<strong>🤖 AI Agent</strong><br>"
        formatted_text = text.replace("\n", "<br>")
        msgs.append(f"""
        <div class="{cls}">
            {prefix}{formatted_text}
        </div>
        """)
    return f"""<div class="agent-conversation">{''.join(msgs)}</div>"""


def _render_jira_kanban_cards(hotels: list[dict], destination: str, evaluations: list[dict] = None) -> str:
    """Generate Jira Kanban board HTML with Issue Tickets categorized by tier, enriched with match reasons & trade-offs."""
    if not hotels:
        return "<div style='padding:30px;text-align:center;color:#626F86;'>No issues found on the board.</div>"

    eval_map = {e.get("hotel_id"): e for e in (evaluations or []) if isinstance(e, dict)}

    amenity_icons = {
        "breakfast": "🥞 Breakfast",
        "wifi": "📶 Wifi",
        "pool": "🏊 Pool",
        "gym": "💪 Gym",
        "spa": "💆 Spa",
        "parking": "🅿️ Parking",
        "room_service": "🛎️ Room Service",
        "concierge": "👔 Concierge",
        "laundry": "🧺 Laundry",
        "minibar": "🍸 Bar",
    }

    luxury_cards = []
    premium_cards = []
    value_cards = []

    for idx, h in enumerate(hotels, start=101):
        issue_key = f"HTL-{idx}"
        stars_count = int(h.get("stars", 3))
        rating = h.get("rating", "N/A")
        price = h.get("price", "N/A")
        name = h.get("name", "Hotel")
        address = h.get("address", destination)
        url = h.get("url", "https://www.booking.com")
        area = h.get("area")
        dist = h.get("distance_km")

        # Status badge config
        if stars_count >= 5:
            badge_class = "jira-badge-purple"
            badge_text = "LUXURY"
            tier_class = "jira-card-luxury"
        elif stars_count == 4:
            badge_class = "jira-badge-blue"
            badge_text = "PREMIUM"
            tier_class = "jira-card-premium"
        else:
            badge_class = "jira-badge-green"
            badge_text = "VALUE"
            tier_class = "jira-card-value"

        # Amenities badges
        amenities = h.get("amenities", [])
        amenities_html = ""
        if amenities:
            chips = [f"<span class='jira-badge jira-badge-slate' style='font-size:0.68rem;padding:2px 6px;'>{amenity_icons.get(a, a.title())}</span>" for a in amenities[:4]]
            amenities_html = f"<div style='display:flex;gap:4px;flex-wrap:wrap;margin:6px 0;'>{''.join(chips)}</div>"

        # Location details
        location_sub = f" · {area}" if area else ""
        if dist is not None and dist > 0:
            location_sub += f" ({dist}km from centre)"

        # Match reasons & Trade-offs
        ev = eval_map.get(h.get("hotel_id"))
        reasons_html = ""
        tradeoffs_html = ""
        if ev:
            reasons = ev.get("match_reasons", [])
            tradeoffs = ev.get("tradeoffs", [])
            if reasons:
                items = "".join([f"<div class='hotel-match-item'>{r}</div>" for r in reasons[:3]])
                reasons_html = f"""
                <div class="hotel-match-reasons">
                    <div class="hotel-match-title">Why This Matches</div>
                    {items}
                </div>
                """
            if tradeoffs:
                tradeoff_txt = " · ".join(tradeoffs[:2])
                tradeoffs_html = f"""
                <div class="hotel-tradeoff">
                    <span>{tradeoff_txt}</span>
                </div>
                """

        card_html = f"""
        <div class="jira-issue-card {tier_class}">
            <div class="jira-issue-top">
                <span class="jira-issue-key">{issue_key}</span>
                <span class="jira-badge {badge_class}">{badge_text}</span>
            </div>
            <div class="jira-issue-title">{name}</div>
            <div class="jira-issue-meta">
                <span class="jira-issue-meta-item">⭐ {'★' * stars_count}</span>
                <span class="jira-issue-meta-item">📊 {rating} Rating</span>
                <span class="jira-issue-meta-item">📍 {address}{location_sub}</span>
            </div>
            {amenities_html}
            {reasons_html}
            {tradeoffs_html}
            <div class="jira-issue-footer" style="margin-top:10px;">
                <div>
                    <span class="jira-price-tag">${price}</span>
                    <span style="font-size:0.72rem;color:#6E7681;"> / night</span>
                </div>
                <a href="{url}" target="_blank" class="jira-book-btn">Book Ticket ➔</a>
            </div>
        </div>
        """

        if stars_count >= 5:
            luxury_cards.append(card_html)
        elif stars_count == 4:
            premium_cards.append(card_html)
        else:
            value_cards.append(card_html)

    board_html = f"""
    <div class="jira-kanban-board">
        <!-- Column 1: Luxury -->
        <div class="jira-column">
            <div class="jira-column-header">
                <span>🌟 5-Star Luxury</span>
                <span class="jira-column-count">{len(luxury_cards)}</span>
            </div>
            {''.join(luxury_cards) if luxury_cards else '<div class="jira-empty-state"><div class="jira-empty-state-icon">🌟</div><div class="jira-empty-state-title">No Luxury Items</div><div class="jira-empty-state-desc">No 5-star hotels matched your search criteria</div></div>'}
        </div>
        <!-- Column 2: Premium -->
        <div class="jira-column">
            <div class="jira-column-header">
                <span>💎 4-Star Premium</span>
                <span class="jira-column-count">{len(premium_cards)}</span>
            </div>
            {''.join(premium_cards) if premium_cards else '<div class="jira-empty-state"><div class="jira-empty-state-icon">💎</div><div class="jira-empty-state-title">No Premium Items</div><div class="jira-empty-state-desc">No 4-star hotels matched your search criteria</div></div>'}
        </div>
        <!-- Column 3: Value -->
        <div class="jira-column">
            <div class="jira-column-header">
                <span>🎯 3-Star Smart Value</span>
                <span class="jira-column-count">{len(value_cards)}</span>
            </div>
            {''.join(value_cards) if value_cards else '<div class="jira-empty-state"><div class="jira-empty-state-icon">🎯</div><div class="jira-empty-state-title">No Value Items</div><div class="jira-empty-state-desc">No 3-star hotels matched your search criteria</div></div>'}
        </div>
    </div>
    """
    return board_html


def _render_booking_summary(data: dict) -> str:
    """Render the Booking Summary panel with cost breakdown and policy."""
    if not data:
        return ""
    hname = data.get("hotel_name", "Hotel")
    rtype = data.get("room_type", "Standard Room")
    cin = data.get("check_in", "")
    cout = data.get("check_out", "")
    nights = data.get("nights", 1)
    guests = data.get("guests", 2)
    rate = data.get("price_per_night", 0)
    subtotal = data.get("subtotal", 0)
    tax = data.get("taxes", 0)
    total = data.get("total", 0)
    policy = data.get("cancellation_policy", "Standard cancellation applies.")

    return f"""
    <div class="booking-summary-panel">
        <div class="booking-summary-header">
            <span>📋</span>
            <span>Booking Summary Review · {hname}</span>
        </div>
        <div class="booking-summary-row">
            <span class="booking-summary-label">Selected Hotel</span>
            <span class="booking-summary-value">{hname}</span>
        </div>
        <div class="booking-summary-row">
            <span class="booking-summary-label">Room Category</span>
            <span class="booking-summary-value">{rtype}</span>
        </div>
        <div class="booking-summary-row">
            <span class="booking-summary-label">Dates of Stay</span>
            <span class="booking-summary-value">{cin} → {cout} ({nights} nights)</span>
        </div>
        <div class="booking-summary-row">
            <span class="booking-summary-label">Guest Count</span>
            <span class="booking-summary-value">{guests} Adults</span>
        </div>
        <div class="booking-summary-row">
            <span class="booking-summary-label">Nightly Rate</span>
            <span class="booking-summary-value">${rate} / night</span>
        </div>
        <div class="booking-summary-row">
            <span class="booking-summary-label">Lodging Subtotal</span>
            <span class="booking-summary-value">${subtotal}</span>
        </div>
        <div class="booking-summary-row">
            <span class="booking-summary-label">Taxes & Service Fees (12%)</span>
            <span class="booking-summary-value">${tax}</span>
        </div>
        <div class="booking-summary-total">
            <span class="booking-summary-total-label">Total Estimated Cost</span>
            <span class="booking-summary-total-value">${total}</span>
        </div>
        <div class="booking-summary-policy">
            <span>🛡️ Policy: {policy}</span>
        </div>
    </div>
    """


def _render_booking_confirmation(booking: dict) -> str:
    """Render the verified reservation confirmation display."""
    if not booking:
        return ""
    bid = booking.get("booking_id", "HTL-CONFIRMED")
    hname = booking.get("hotel_name", "Hotel")
    rtype = booking.get("room_type", "Standard Room")
    cin = booking.get("check_in", "")
    cout = booking.get("check_out", "")
    nights = booking.get("nights", 1)
    guests = booking.get("guests", 2)
    total = booking.get("total", 0)
    created = booking.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    policy_desc = booking.get("cancellation", {}).get("description", "Free cancellation policy applies.")

    return f"""
    <div class="booking-confirmed">
        <div class="booking-confirmed-icon">✅</div>
        <div class="booking-confirmed-title">Reservation Successfully Booked!</div>
        <div style="color:var(--text-secondary);font-size:0.9rem;">Your reservation has been confirmed and verified by the AI Agent.</div>
        <div class="booking-confirmed-id">{bid}</div>
        <div class="booking-confirmed-details">
            <div class="booking-summary-row">
                <span class="booking-summary-label">Hotel Property</span>
                <span class="booking-summary-value">{hname}</span>
            </div>
            <div class="booking-summary-row">
                <span class="booking-summary-label">Reserved Room</span>
                <span class="booking-summary-value">{rtype}</span>
            </div>
            <div class="booking-summary-row">
                <span class="booking-summary-label">Stay Period</span>
                <span class="booking-summary-value">{cin} → {cout} ({nights} nights)</span>
            </div>
            <div class="booking-summary-row">
                <span class="booking-summary-label">Guests</span>
                <span class="booking-summary-value">{guests} Adults</span>
            </div>
            <div class="booking-summary-row">
                <span class="booking-summary-label">Total Charged</span>
                <span class="booking-summary-value" style="color:#58A6FF;font-weight:700;">${total}</span>
            </div>
            <div class="booking-summary-row">
                <span class="booking-summary-label">Booking Timestamp</span>
                <span class="booking-summary-value">{created}</span>
            </div>
            <div class="booking-summary-row">
                <span class="booking-summary-label">Cancellation Terms</span>
                <span class="booking-summary-value" style="color:#34D399;">{policy_desc}</span>
            </div>
            <div class="booking-summary-row">
                <span class="booking-summary-label">Dispatch Status</span>
                <span class="booking-summary-value" style="color:#34D399;font-weight:700;">ISSUED · VERIFIED ON BOARD</span>
            </div>
        </div>
    </div>
    """


# ─── Core Logic ─────────────────────────────────────────────────────────────

def find_hotels(destination, check_in, check_out, min_price, max_price, adults, stars, session: AgentSession = None):
    """
    1. Groq LLM: user intent → structured params
    2. Hotel API: search results
    3. Groq LLM: smart recommendation
    4. Return (kanban_html, results_df, recommendation_text, status_label, hotel_choices)
    """
    dest_str = (destination or "").strip()
    if not dest_str:
        dest_str = "Paris"

    min_p = int(min_price) if min_price is not None else 80
    max_p = int(max_price) if max_price is not None else 400
    if min_p > max_p:
        min_p, max_p = max_p, min_p

    adults_cnt = int(adults) if adults is not None else 2
    star_val = int(stars) if stars and str(stars) not in ("Any", "None") else None
    check_in_val = check_in or "2025-12-01"
    check_out_val = check_out or "2025-12-07"

    # ── 1. Groq → structured params ──────────────────────────────────────
    prompt = build_search_prompt(
        dest_str, check_in_val, check_out_val, min_p, max_p, adults_cnt, star_val
    )
    try:
        resp = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json\n").strip()
        params = json.loads(raw)
    except Exception as exc:
        print(f"[app] Groq parse note: {exc}")
        params = {
            "destination": dest_str,
            "check_in": check_in_val,
            "check_out": check_out_val,
            "min_price": min_p,
            "max_price": max_p,
            "adults": adults_cnt,
            "stars": star_val,
        }

    # ── 2. Hotel API ──────────────────────────────────────────────────────
    hotels = search_hotels(
        destination=params.get("destination", dest_str),
        check_in=params.get("check_in", check_in_val),
        check_out=params.get("check_out", check_out_val),
        min_price=params.get("min_price", min_p),
        max_price=params.get("max_price", max_p),
        adults=params.get("adults", adults_cnt),
        stars=params.get("stars", star_val),
    )

    if session:
        session.registry.register_hotels(hotels)
        session.update_state(searchResults=hotels, destination=dest_str)

    if not hotels:
        empty_df = pd.DataFrame(columns=["Key", "Name", "Stars", "Price/night", "Rating", "Address"])
        return (
            "<div style='padding:20px;color:#626F86;'>No hotel tickets found for this query.</div>",
            empty_df,
            "⚠️ No matching hotels found. Try adjusting the price range or filters.",
            "Board updated · 0 issues",
            gr.Dropdown(choices=["No hotels loaded"], value="No hotels loaded"),
        )

    # ── 3. Groq → Smart Summary ───────────────────────────────────────────
    rec_text = ""
    try:
        rec_prompt = build_summary_prompt(hotels)
        rec_resp = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": rec_prompt}],
            temperature=0.6,
            max_tokens=300,
        )
        rec_text = (rec_resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[app] Groq summary note: {exc}")
        rec_text = "✅ Hotels located and dispatched to the Jira Board."

    # ── 4. Kanban & Dataframe outputs ─────────────────────────────────────
    reqs = {
        "destination": dest_str,
        "checkIn": check_in_val,
        "checkOut": check_out_val,
        "maxPrice": max_p,
        "minPrice": min_p,
        "guests": adults_cnt,
        "stars": star_val,
    }
    evaluations = _fallback_evaluations(hotels, reqs)
    kanban_html = _render_jira_kanban_cards(hotels, dest_str, evaluations)

    df_rows = []
    hotel_dropdown_choices = []
    for idx, h in enumerate(hotels, start=101):
        key = f"HTL-{idx}"
        df_rows.append({
            "Key": key,
            "Name": h.get("name", "N/A"),
            "Stars": "★" * int(h.get("stars", 0)) or "—",
            "Price/night": f"${h.get('price', '?')}",
            "Rating": f"{h.get('rating', '—')} / 10",
            "Address": h.get("address", "—"),
        })
        hotel_dropdown_choices.append(f"{key}: {h.get('name', 'Hotel')} (${h.get('price', 0)}/nt)")

    df = pd.DataFrame(df_rows)
    status = f"⚡ Board synced: {len(hotels)} issues found for {dest_str.title()} (Groq AI active)"

    dropdown_update = gr.Dropdown(
        choices=hotel_dropdown_choices,
        value=hotel_dropdown_choices[0] if hotel_dropdown_choices else "No hotels loaded",
    )

    return kanban_html, df, rec_text, status, dropdown_update


# ─── Agent Turn Handler ──────────────────────────────────────────────────────

def on_agent_submit(
    user_msg: str,
    session: AgentSession,
    history: list[dict],
    cur_dest: str,
    cur_cin: str,
    cur_cout: str,
    cur_minp: float,
    cur_maxp: float,
    cur_adults: int,
    cur_stars: str,
):
    """
    Execute an agentic turn from the chat input.
    Returns:
    - agent_input (cleared)
    - conversation_html
    - activities_html
    - requirements_html
    - kanban_html
    - dataframe
    - recommendation
    - status_msg
    - hotel_dropdown
    - destination
    - check_in
    - check_out
    - min_price
    - max_price
    - adults
    - stars
    - booking_display
    - session (state)
    - history (state)
    """
    clean_msg = (user_msg or "").strip()
    if not clean_msg:
        conv_html = _render_conversation_history(history)
        return (
            "", conv_html, "", "", gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), cur_dest, cur_cin, cur_cout,
            cur_minp, cur_maxp, cur_adults, cur_stars, gr.update(), session, history
        )

    if session is None:
        session = AgentSession()
    if history is None:
        history = []

    # Run agent loop
    result = run_agent(session, clean_msg)

    # Append to conversation history
    history.append({"sender": "user", "text": clean_msg})
    if result.get("response"):
        history.append({"sender": "bot", "text": result["response"]})

    conv_html = _render_conversation_history(history)
    act_html = _render_agent_activities(result.get("activities", []))
    req_html = _render_agent_requirements(result.get("requirements", {}))

    # Search results check
    hotels = result.get("searchResults", [])
    evaluations = result.get("evaluations", [])
    reqs = result.get("requirements", {})

    new_dest = cur_dest
    new_cin = cur_cin
    new_cout = cur_cout
    new_minp = cur_minp
    new_maxp = cur_maxp
    new_adults = cur_adults
    new_stars = cur_stars

    if reqs.get("destination"):
        new_dest = reqs["destination"].title()
    if reqs.get("checkIn"):
        new_cin = reqs["checkIn"]
    if reqs.get("checkOut"):
        new_cout = reqs["checkOut"]
    if reqs.get("maxPrice"):
        new_maxp = float(reqs["maxPrice"])
    if reqs.get("minPrice"):
        new_minp = float(reqs["minPrice"])
    if reqs.get("guests"):
        new_adults = int(reqs["guests"])
    if reqs.get("stars") is not None:
        new_stars = str(reqs["stars"])

    booking_res = result.get("bookingResult")
    booking_html = _render_booking_confirmation(booking_res) if booking_res else gr.update()

    if hotels:
        kanban_html = _render_jira_kanban_cards(hotels, new_dest, evaluations)
        df_rows = []
        hotel_dropdown_choices = []
        for idx, h in enumerate(hotels, start=101):
            key = f"HTL-{idx}"
            df_rows.append({
                "Key": key,
                "Name": h.get("name", "N/A"),
                "Stars": "★" * int(h.get("stars", 0)) or "—",
                "Price/night": f"${h.get('price', '?')}",
                "Rating": f"{h.get('rating', '—')} / 10",
                "Address": h.get("address", "—"),
            })
            hotel_dropdown_choices.append(f"{key}: {h.get('name', 'Hotel')} (${h.get('price', 0)}/nt)")

        df = pd.DataFrame(df_rows)
        status = f"⚡ Agent synced {len(hotels)} verified properties to Jira Board for {new_dest}"
        rec = result.get("response", "Hotels evaluated and dispatched to Kanban board.")
        dropdown_update = gr.Dropdown(
            choices=hotel_dropdown_choices,
            value=hotel_dropdown_choices[0] if hotel_dropdown_choices else "No hotels loaded",
        )
        return (
            "", conv_html, act_html, req_html, kanban_html, df, rec,
            status, dropdown_update, new_dest, new_cin, new_cout,
            new_minp, new_maxp, new_adults, new_stars, booking_html, session, history
        )
    else:
        status = f"⚡ Agent interaction active · Ready"
        return (
            "", conv_html, act_html, req_html, gr.update(), gr.update(),
            gr.update(), status, gr.update(), new_dest, new_cin, new_cout,
            new_minp, new_maxp, new_adults, new_stars, booking_html, session, history
        )


def on_reset_session():
    """Reset session state and conversation."""
    new_sess = AgentSession()
    empty_history = []
    conv_html = _render_conversation_history([])
    return "", conv_html, "", "", new_sess, empty_history


def on_review_booking(selected_hotel_str: Any, room_type: str, cin: str, cout: str, adults: int, session: AgentSession):
    """Calculate costs, check cancellation, and render the booking review modal."""
    if isinstance(selected_hotel_str, (list, tuple)):
        selected_hotel_str = str(selected_hotel_str[0]) if selected_hotel_str else ""
    else:
        selected_hotel_str = str(selected_hotel_str or "")

    if not selected_hotel_str or "No hotels" in selected_hotel_str or not session:
        return "<div style='padding:14px;color:#FBBF24;'>⚠️ Please search for hotels and select one from the board first.</div>"

    # Find hotel by name or ID in registry
    hotel_name_clean = selected_hotel_str.split(":", 1)[-1].split("($")[0].strip() if ":" in selected_hotel_str else selected_hotel_str
    target_hotel = None
    for hid in session.registry.list_hotel_ids():
        h = session.registry.get_hotel(hid)
        if h and (h.get("name") == hotel_name_clean or hotel_name_clean in h.get("name", "")):
            target_hotel = h
            break

    if not target_hotel:
        # Fallback to searchResults
        for h in session.state.get("searchResults", []):
            if h.get("name") == hotel_name_clean or hotel_name_clean in h.get("name", ""):
                target_hotel = h
                session.registry.register_hotels([h])
                break

    if not target_hotel:
        return f"<div style='padding:14px;color:#FBBF24;'>⚠️ Property '{hotel_name_clean}' not found in current session inventory.</div>"

    hotel_id = target_hotel["hotel_id"]
    cost_res = tool_calculate_booking_cost(
        session.registry,
        hotel_id=hotel_id,
        room_type=room_type,
        check_in=cin or "2025-12-01",
        check_out=cout or "2025-12-07",
    )
    if not cost_res.get("success"):
        # Attempt with first available room
        first_room = target_hotel.get("rooms", [{}])[0].get("type", "Standard Room")
        cost_res = tool_calculate_booking_cost(
            session.registry,
            hotel_id=hotel_id,
            room_type=first_room,
            check_in=cin or "2025-12-01",
            check_out=cout or "2025-12-07",
        )
        if not cost_res.get("success"):
            return f"<div style='padding:14px;color:#EF4444;'>❌ Error calculating costs: {cost_res.get('error')}</div>"

    policy_res = tool_check_cancellation_policy(session.registry, hotel_id=hotel_id, room_type=room_type)
    policy_desc = policy_res.get("cancellation", {}).get("description", "Free cancellation up to 2 days before check-in.")

    summary_data = {
        "hotel_name": target_hotel["name"],
        "room_type": cost_res.get("room_type", room_type),
        "check_in": cost_res.get("check_in", cin),
        "check_out": cost_res.get("check_out", cout),
        "nights": cost_res.get("nights", 1),
        "guests": int(adults or 2),
        "price_per_night": cost_res.get("price_per_night", target_hotel.get("price", 0)),
        "subtotal": cost_res.get("subtotal", 0),
        "taxes": cost_res.get("taxes", 0),
        "total": cost_res.get("total", 0),
        "cancellation_policy": policy_desc,
    }
    return _render_booking_summary(summary_data)


def on_confirm_booking(selected_hotel_str: Any, room_type: str, cin: str, cout: str, adults: int, session: AgentSession):
    """Execute reservation creation and render confirmation ticket."""
    if isinstance(selected_hotel_str, (list, tuple)):
        selected_hotel_str = str(selected_hotel_str[0]) if selected_hotel_str else ""
    else:
        selected_hotel_str = str(selected_hotel_str or "")

    if not selected_hotel_str or "No hotels" in selected_hotel_str or not session:
        return "<div style='padding:14px;color:#FBBF24;'>⚠️ Please search for hotels and select one from the board first.</div>"

    hotel_name_clean = selected_hotel_str.split(":", 1)[-1].split("($")[0].strip() if ":" in selected_hotel_str else selected_hotel_str
    target_hotel = None
    for hid in session.registry.list_hotel_ids():
        h = session.registry.get_hotel(hid)
        if h and (h.get("name") == hotel_name_clean or hotel_name_clean in h.get("name", "")):
            target_hotel = h
            break

    if not target_hotel:
        for h in session.state.get("searchResults", []):
            if h.get("name") == hotel_name_clean or hotel_name_clean in h.get("name", ""):
                target_hotel = h
                session.registry.register_hotels([h])
                break


    if not target_hotel:
        return f"<div style='padding:14px;color:#FBBF24;'>⚠️ Property '{hotel_name_clean}' not found in current session inventory.</div>"

    hotel_id = target_hotel["hotel_id"]
    book_res = tool_create_booking(session.registry, {
        "hotel_id": hotel_id,
        "room_type": room_type,
        "check_in": cin or "2025-12-01",
        "check_out": cout or "2025-12-07",
        "guests": int(adults or 2),
    })

    if not book_res.get("success"):
        # Fallback to first room type if chosen room isn't matched
        first_room = target_hotel.get("rooms", [{}])[0].get("type", "Standard Room")
        book_res = tool_create_booking(session.registry, {
            "hotel_id": hotel_id,
            "room_type": first_room,
            "check_in": cin or "2025-12-01",
            "check_out": cout or "2025-12-07",
            "guests": int(adults or 2),
        })

    if book_res.get("success") and book_res.get("booked"):
        return _render_booking_confirmation(book_res["booking"])
    else:
        return f"<div style='padding:14px;color:#EF4444;'>❌ Booking could not be completed: {book_res.get('message', 'Unavailable')}</div>"


# ─── UI Layout (Jira Enterprise Style) ──────────────────────────────────────

with gr.Blocks(title="Partian Hotels · Jira Work Management") as demo:

    # Global Session & History States
    session_state = gr.State(lambda: AgentSession())
    history_state = gr.State(lambda: [])

    # ── Top Jira Navigation Bar ───────────────────────────────────────────
    gr.HTML("""
    <div class="jira-top-navbar">
        <div class="jira-nav-left">
            <div class="jira-logo">
                <div class="jira-logo-icon">🔷</div>
                <span>Partian Jira</span>
            </div>
            <div class="jira-breadcrumbs">
                <span>Projects</span>
                <span style="color:rgba(255,255,255,0.2);">/</span>
                <span>Travel Operations</span>
                <span style="color:rgba(255,255,255,0.2);">/</span>
                <strong>Hotel Booking Hub</strong>
            </div>
        </div>
        <div class="jira-nav-right">
            <span class="jira-badge jira-badge-green" style="animation:breathe 2s ease-in-out infinite;">⚡ GROQ LLM: ONLINE</span>
            <div class="jira-user-pill">
                <div class="jira-avatar">PT</div>
                <span>Concierge Lead</span>
            </div>
        </div>
    </div>
    """)

    # ── AI Agent Assistant Workspace (Top Feature) ─────────────────────────
    with gr.Group():
        gr.HTML("""
        <div class="agent-chat-header" style="margin-bottom:8px;">
            <div class="agent-chat-icon">🤖</div>
            <div>
                <div class="agent-chat-title">AI Autonomous Booking Concierge</div>
                <div class="agent-chat-subtitle">Natural Language Intent · Multi-Tool Execution · Trade-Off Analysis · Instant Reservation</div>
            </div>
        </div>
        """)

        # Conversation history container
        conversation_view = gr.HTML(value=_render_conversation_history([]))

        # Agent chat input
        with gr.Row():
            agent_input = gr.Textbox(
                label="",
                placeholder="Type your request: e.g. 'I need a 4-star hotel in Mumbai from 2025-11-10 to 2025-11-15 for 2 people under $200 with pool and breakfast' or 'Book HTL-101' …",
                show_label=False,
                scale=5,
                lines=1,
            )
            agent_submit_btn = gr.Button("⚡ Ask AI Agent", variant="primary", scale=1)
            agent_reset_btn = gr.Button("↺ Clear", variant="secondary", scale=1)

        # Agent Activity & Requirements Displays
        agent_activity_view = gr.HTML(value="")
        agent_reqs_view = gr.HTML(value="")

    # ── Jira Filter Toolbar (Board Header & Manual Form) ───────────────────
    with gr.Group():
        gr.HTML("""
        <div class="jira-board-header">
            <h2 class="jira-board-title">
                🏨 Global Hotel Board
                <span class="jira-badge jira-badge-green">Live API Connected</span>
            </h2>
            <div style="font-size:0.78rem;color:#5A657A;display:flex;align-items:center;gap:12px;">
                <span>Group by: <strong style="color:#8B96A8;">Tier Status</strong></span>
                <span style="color:rgba(255,255,255,0.15);">|</span>
                <span>Sprint: <strong style="color:#8B96A8;">Q4-Travel-Ops</strong></span>
            </div>
        </div>
        """)

        with gr.Row():
            destination = gr.Textbox(
                label="📍 DESTINATION (QUICK SEARCH)",
                value="Tokyo",
                placeholder="Search city e.g. Tokyo, Paris, London, Mumbai …",
                scale=3,
            )
            check_in = gr.Textbox(
                label="📅 CHECK-IN",
                value="2025-12-01",
                scale=1,
            )
            check_out = gr.Textbox(
                label="📅 CHECK-OUT",
                value="2025-12-07",
                scale=1,
            )

        with gr.Row():
            min_price = gr.Number(
                label="💰 MIN PRICE ($/NIGHT)",
                value=80, minimum=0, maximum=10000, step=10,
                scale=1,
            )
            max_price = gr.Number(
                label="💰 MAX PRICE ($/NIGHT)",
                value=400, minimum=0, maximum=10000, step=10,
                scale=1,
            )
            adults = gr.Slider(
                label="👤 GUESTS / ADULTS",
                minimum=1, maximum=8, value=2, step=1,
                scale=1,
            )
            stars = gr.Dropdown(
                label="⭐ STAR TIER",
                choices=["Any", "1", "2", "3", "4", "5"],
                value="Any",
                scale=1,
            )

        with gr.Row():
            search_btn = gr.Button("⚡ Find & Dispatch Hotels to Board", variant="primary", scale=3)

    # ── Status Bar ────────────────────────────────────────────────────────
    status_msg = gr.Textbox(
        label="",
        value="⚡ Board synced: Ready for queries (Groq AI active)",
        interactive=False,
        show_label=False,
        container=False,
        lines=1,
    )

    # ── Jira Smart AI Recommendation Card ─────────────────────────────────
    with gr.Group():
        gr.HTML("""
        <div class="jira-ai-header">
            <span style="font-size:1.1em;">🤖</span>
            <span>Groq AI Intelligence</span>
            <span class="jira-badge jira-badge-blue" style="margin-left:auto;">Smart Recommendation</span>
        </div>
        """)
        recommendation = gr.Textbox(
            label="",
            value="Click '⚡ Find & Dispatch Hotels to Board' or ask the AI Concierge above to trigger Groq AI analysis.",
            lines=3,
            interactive=False,
            show_label=False,
        )

    # ── Board & Backlog Tabs ──────────────────────────────────────────────
    with gr.Tabs():
        with gr.TabItem("📋 Kanban Board View"):
            board_output = gr.HTML(
                value="""
                <div class="jira-empty-state" style="padding:60px 24px;">
                    <div class="jira-empty-state-icon">🗂️</div>
                    <div class="jira-empty-state-title">Board is Empty</div>
                    <div class="jira-empty-state-desc">
                        Ask the <strong>AI Booking Concierge</strong> above or click <strong>⚡ Find & Dispatch Hotels to Board</strong> to search hotels and populate Jira Issue cards across tier columns.
                    </div>
                </div>
                """
            )

        with gr.TabItem("📑 Backlog / List View"):
            results_table = gr.Dataframe(
                headers=["Key", "Name", "Stars", "Price/night", "Rating", "Address"],
                datatype=["str", "str", "str", "str", "str", "str"],
                wrap=True,
                show_label=False,
                interactive=False,
            )

    # ── Quick Booking & Reservation Desk ──────────────────────────────────
    with gr.Group():
        gr.HTML("""
        <div class="jira-ai-header" style="margin-top:14px;border-radius:12px 12px 0 0;">
            <span style="font-size:1.1em;">⚡</span>
            <span>Interactive Reservation Desk</span>
            <span class="jira-badge jira-badge-green" style="margin-left:auto;">Instant Booking Engine</span>
        </div>
        """)
        with gr.Row():
            hotel_dropdown = gr.Dropdown(
                label="SELECT HOTEL FROM BOARD",
                choices=["No hotels loaded yet"],
                value="No hotels loaded yet",
                scale=3,
            )
            room_dropdown = gr.Dropdown(
                label="ROOM CATEGORY",
                choices=["Standard Room", "Deluxe King", "Premium Suite"],
                value="Standard Room",
                scale=2,
            )
        with gr.Row():
            review_btn = gr.Button("📋 Review Cost Breakdown & Policies", variant="secondary", scale=2)
            confirm_btn = gr.Button("⚡ Confirm & Issue Reservation Ticket", variant="primary", scale=2)

        booking_output = gr.HTML(value="")

    # ── Footer ────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="jira-footer">
        Partian Jira Design System &nbsp;·&nbsp;
        Groq AI (gpt-oss-120b Agentic Function Calling) &nbsp;·&nbsp;
        RapidAPI Hotels Integration &nbsp;·&nbsp;
        Automated Reservation Verification
    </div>
    """)

    # ── Wire Up Event Handlers ────────────────────────────────────────────

    # Agent Chat submission
    agent_submit_btn.click(
        fn=on_agent_submit,
        inputs=[
            agent_input, session_state, history_state,
            destination, check_in, check_out, min_price, max_price, adults, stars
        ],
        outputs=[
            agent_input, conversation_view, agent_activity_view, agent_reqs_view,
            board_output, results_table, recommendation, status_msg, hotel_dropdown,
            destination, check_in, check_out, min_price, max_price, adults, stars,
            booking_output, session_state, history_state
        ],
    )
    agent_input.submit(
        fn=on_agent_submit,
        inputs=[
            agent_input, session_state, history_state,
            destination, check_in, check_out, min_price, max_price, adults, stars
        ],
        outputs=[
            agent_input, conversation_view, agent_activity_view, agent_reqs_view,
            board_output, results_table, recommendation, status_msg, hotel_dropdown,
            destination, check_in, check_out, min_price, max_price, adults, stars,
            booking_output, session_state, history_state
        ],
    )

    # Agent Reset
    agent_reset_btn.click(
        fn=on_reset_session,
        inputs=[],
        outputs=[agent_input, conversation_view, agent_activity_view, agent_reqs_view, session_state, history_state],
    )

    # Manual Form Search
    search_btn.click(
        fn=find_hotels,
        inputs=[destination, check_in, check_out, min_price, max_price, adults, stars, session_state],
        outputs=[board_output, results_table, recommendation, status_msg, hotel_dropdown],
    )

    # Review Booking
    review_btn.click(
        fn=on_review_booking,
        inputs=[hotel_dropdown, room_dropdown, check_in, check_out, adults, session_state],
        outputs=[booking_output],
    )

    # Confirm Booking
    confirm_btn.click(
        fn=on_confirm_booking,
        inputs=[hotel_dropdown, room_dropdown, check_in, check_out, adults, session_state],
        outputs=[booking_output],
    )


# ── Export top-level variables for WSGI / ASGI / Serverless runners (Vercel, Render, Spaces) ─
app = gr.mount_gradio_app(
    FastAPI(title="Partian Hotels · Jira Work Management"),
    demo,
    path="/",
    theme=jira_theme,
    css=css,
)
application = app
handler = app


if __name__ == "__main__":
    import os
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        css=css,
        theme=jira_theme,
    )
