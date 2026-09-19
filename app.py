"""
app.py  –  Hotel Booking Assistant (Jira / Partian Enterprise Edition)
──────────────────────────────────────────────────────────────────────────
Gradio 6 + Groq AI (gpt-oss-120b) + RapidAPI Hotels with Jira UI System
"""

import os
import json
import pandas as pd
import gradio as gr
from gradio.themes import Base
from dotenv import load_dotenv
from groq import Groq
from hotel_api import search_hotels
from prompt_templates import build_search_prompt, build_summary_prompt

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
CSS_PATH = os.path.join(os.path.dirname(__file__), "ui.css")
css = open(CSS_PATH).read() if os.path.exists(CSS_PATH) else ""


# ─── Jira / Partian Design System Theme ───────────────────────────────────
jira_theme = Base(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    # Backgrounds
    background_fill_primary="#F7F8F9",
    background_fill_secondary="#FFFFFF",
    # Blocks
    block_background_fill="#FFFFFF",
    block_border_width="1px",
    block_border_color="#DFE1E6",
    block_shadow="0 1px 1px rgba(9, 30, 66, 0.25), 0 0 1px rgba(9, 30, 66, 0.31)",
    block_radius="4px",
    block_padding="14px 16px",
    block_label_background_fill="transparent",
    block_label_text_color="#44546F",
    block_label_text_size="xs",
    block_label_text_weight="700",
    # Inputs
    input_background_fill="#FAFBFC",
    input_border_color="#DFE1E6",
    input_border_width="1.5px",
    input_border_color_focus="#0C66E4",
    input_shadow_focus="0 0 0 2px #85B8FF",
    input_radius="3px",
    input_text_size="sm",
    # Primary Button (Jira Blue)
    button_primary_background_fill="#0C66E4",
    button_primary_background_fill_hover="#0052CC",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="transparent",
    button_primary_shadow="0 1px 2px rgba(9, 30, 66, 0.2)",
    button_large_radius="3px",
    button_large_padding="9px 18px",
    button_large_text_size="sm",
    button_large_text_weight="600",
    # Text
    body_text_color="#172B4D",
    body_text_color_subdued="#626F86",
    body_text_size="sm",
)


def _render_jira_kanban_cards(hotels: list[dict], destination: str) -> str:
    """Generate Jira Kanban board HTML with Issue Tickets categorized by tier."""
    if not hotels:
        return "<div style='padding:30px;text-align:center;color:#626F86;'>No issues found on the board.</div>"

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

        # Status badge config
        if stars_count >= 5:
            badge_class = "jira-badge-purple"
            badge_text = "LUXURY"
            priority_icon = "🔺 Highest"
            tier_class = "jira-card-luxury"
        elif stars_count == 4:
            badge_class = "jira-badge-blue"
            badge_text = "PREMIUM"
            priority_icon = "🔸 High"
            tier_class = "jira-card-premium"
        else:
            badge_class = "jira-badge-green"
            badge_text = "VALUE"
            priority_icon = "🔹 Standard"
            tier_class = "jira-card-value"

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
                <span class="jira-issue-meta-item">📍 {address}</span>
            </div>
            <div class="jira-issue-footer">
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


# ─── Core Logic ─────────────────────────────────────────────────────────────
def find_hotels(destination, check_in, check_out, min_price, max_price, adults, stars):
    """
    1. Groq LLM: user intent → structured params
    2. Hotel API: search results
    3. Groq LLM: smart recommendation
    4. Return (kanban_html, results_df, recommendation_text, status_label)
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
        raw = resp.choices[0].message.content.strip()
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

    if not hotels:
        empty_df = pd.DataFrame(columns=["Key", "Name", "Stars", "Price/night", "Rating", "Address"])
        return (
            "<div style='padding:20px;color:#626F86;'>No hotel tickets found for this query.</div>",
            empty_df,
            "⚠️ No matching hotels found. Try adjusting the price range or filters.",
            "Board updated · 0 issues",
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
        rec_text = rec_resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[app] Groq summary note: {exc}")
        rec_text = "✅ Hotels located and dispatched to the Jira Board."

    # ── 4. Kanban & Dataframe outputs ─────────────────────────────────────
    kanban_html = _render_jira_kanban_cards(hotels, dest_str)

    df_rows = []
    for idx, h in enumerate(hotels, start=101):
        df_rows.append({
            "Key": f"HTL-{idx}",
            "Name": h.get("name", "N/A"),
            "Stars": "★" * int(h.get("stars", 0)) or "—",
            "Price/night": f"${h.get('price', '?')}",
            "Rating": f"{h.get('rating', '—')} / 10",
            "Address": h.get("address", "—"),
        })
    df = pd.DataFrame(df_rows)
    status = f"⚡ Board synced: {len(hotels)} issues found for {dest_str.title()} (Groq AI active)"

    return kanban_html, df, rec_text, status


# ─── UI Layout (Jira Enterprise Style) ──────────────────────────────────────
with gr.Blocks() as demo:

    # ── Top Jira Navigation Bar ───────────────────────────────────────────
    gr.HTML("""
    <div class="jira-top-navbar">
        <div class="jira-nav-left">
            <div class="jira-logo">
                <div class="jira-logo-icon">🔷</div>
                <span>Partian Jira</span>
            </div>
            <div class="jira-breadcrumbs">
                <span>Projects</span> / <span>Travel Operations</span> / <strong>Hotel Booking Hub</strong>
            </div>
        </div>
        <div class="jira-nav-right">
            <span class="jira-badge jira-badge-blue">⚡ Groq LLM: Online</span>
            <div class="jira-user-pill">
                <div class="jira-avatar">PT</div>
                <span>Concierge Lead</span>
            </div>
        </div>
    </div>
    """)

    # ── Jira Filter Toolbar (Board Header) ────────────────────────────────
    with gr.Group():
        gr.HTML("""
        <div class="jira-board-header">
            <h2 class="jira-board-title">
                🏨 Global Hotel Board
                <span class="jira-badge jira-badge-green">Live API Connected</span>
            </h2>
            <div style="font-size:0.8rem;color:#626F86;">
                Group by: <strong>Tier Status</strong> &nbsp;|&nbsp; Sprint: <strong>Q4-Travel-Ops</strong>
            </div>
        </div>
        """)

        with gr.Row():
            destination = gr.Textbox(
                label="📍 DESTINATION (QUICK SEARCH)",
                value="Tokyo",
                placeholder="Search city e.g. Tokyo, Paris, London …",
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
            <span>🤖 Jira Intelligence</span>
            <span class="jira-badge jira-badge-blue">Smart AI Summary</span>
        </div>
        """)
        recommendation = gr.Textbox(
            label="",
            value="Click 'Find & Dispatch Hotels to Board' to trigger Groq AI analysis.",
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
                        Click <strong>⚡ Find & Dispatch Hotels to Board</strong> above to search hotels and populate Jira Issue cards across tier columns.
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

    # ── Wire up ───────────────────────────────────────────────────────────
    search_btn.click(
        fn=find_hotels,
        inputs=[destination, check_in, check_out, min_price, max_price, adults, stars],
        outputs=[board_output, results_table, recommendation, status_msg],
    )

    # ── Footer ────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="jira-footer">
        Partian Jira Design System &nbsp;·&nbsp;
        Groq AI (gpt-oss-120b) &nbsp;·&nbsp;
        RapidAPI Hotels Integration
    </div>
    """)

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
