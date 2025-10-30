# Personal AI Analyst

A Streamlit web app that lets users upload CSV/XLSX files, perform data cleaning, interactive exploratory data analysis, and get AI-generated natural language summaries and answers using OpenAI.

## Features

- CSV/XLSX file upload with preview
- Data cleaning: handle nulls, remove duplicates
- Exploratory Data Analysis: summary statistics
- Interactive visualizations: histogram, bar, line (Plotly)
- AI-generated insights using OpenAI GPT (requires API key)
- AI-generated insights using OpenAI, OpenRouter, Gemini, or local Ollama
- Optional: Chat with Data (ask questions about your dataset)
- Optional: Forecasting / PDF reports (placeholders / optional dependencies)

## Quickstart (local)

1. Create a Python environment (recommended):

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
# or for Git Bash / WSL: source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your AI provider keys (choose one):

```bash
# OpenAI
export OPENAI_API_KEY="your_api_key_here"

# OpenRouter (cheaper multi-provider routing)
export OPENROUTER_API_KEY="your_openrouter_key"

# Ollama (local, offline). Install Ollama and pull a model first:
#   https://ollama.com/  | Example: ollama run llama3.1:8b
# No key needed.

# Gemini (Google Generative AI)
export GEMINI_API_KEY="your_gemini_key"
```

4. Run the app:

```bash
streamlit run app.py
```

## Deployment

- You can deploy the app to Streamlit Cloud. Add your `OPENAI_API_KEY` to the Secrets (Settings) in Streamlit Cloud.

## Files

- `app.py` — Streamlit app entrypoint
- `utils/data_cleaning.py` — data cleaning helpers
- `utils/visualization.py` — Plotly visualization helpers
- `utils/ai_summary.py` — OpenAI integration
- `utils/ai_summary.py` — AI integration (OpenAI, OpenRouter, Gemini, Ollama)
- `datasets/sample.csv` — sample dataset
- `reports/` — place generated reports here

## Notes & Next steps

- Forecasting with Prophet and PDF export are optional; add UI and extra dependencies if needed.
- Improve security by using Streamlit Secrets for API keys.
- You can place keys in a `.env` file at project root; the app loads it automatically.
- Add unit tests and CI for quality gates.
