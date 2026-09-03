# 🏨 Hotel Booking Assistant (Partian Jira Edition)

An automated AI-powered hotel booking assistant with a **Partian Jira-inspired Enterprise UI**, powered by **Groq LLM (`openai/gpt-oss-120b`)** and integrated with the **RapidAPI Hotels4 API**.

---

## 🌟 Features

- **Jira-Inspired UI**: Custom Atlassian Design System tokens, Jira Top Navbar, Project Breadcrumbs, and Quick Filters.
- **Interactive Jira Kanban Board**: Hotels organized into 3 tier columns:
  - 🌟 **5-Star Luxury**
  - 💎 **4-Star Premium**
  - 🎯 **3-Star Smart Value**
  - Each card formatted like a Jira Issue Ticket (`HTL-101`, status badges, priority tags, price pills, booking action).
- **📑 Backlog / List View**: High-density Jira table for quick comparison.
- **🤖 Jira Intelligence (Smart AI Summary)**: Groq LLM analyzes hotel search results in real-time to recommend the best value and top luxury options.
- **Live Search & Filters**: Destination search, check-in/out dates, min/max price sliders, guests count, and star tier selector.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/parthtiwaskar/agentic_project.git
cd agentic_project
```

### 2. Set up virtual environment & install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Add your API keys to `.env`:
```ini
GROQ_API_KEY=gsk_your_groq_api_key
RAPIDAPI_KEY=your_rapidapi_key
```

### 4. Run the application
```bash
python app.py
```
Open **`http://127.0.0.1:7860`** in your browser.

---

## 📂 Project Structure

```
├── app.py               # Main Gradio application & Jira layout
├── hotel_api.py         # RapidAPI Hotels4 wrapper with dynamic fallback
├── prompt_templates.py  # Groq LLM prompt builders
├── ui.css               # Jira / Partian Design System styling
├── requirements.txt     # Python package dependencies
├── .env.example         # Example environment variables
└── README.md            # Project documentation
```

---

## 🛠️ Tech Stack

- **UI Framework**: Gradio 6.x
- **LLM Engine**: Groq API (`openai/gpt-oss-120b`)
- **Hotel Data**: RapidAPI Hotels4
- **Styling**: Vanilla CSS (Atlassian / Partian Design Tokens)
- **Language**: Python 3.10+
