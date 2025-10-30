"""
AI integration helpers to generate natural language summaries and answer questions.

Supported providers:
- OpenAI (default): requires OPENAI_API_KEY
- OpenRouter: OPENROUTER_API_KEY, OpenAI-compatible API
- Ollama (local): no key required; assumes Ollama running at http://localhost:11434

Both functions accept a `provider` argument to match calls from app.py.
"""
import os
import json
from typing import Optional

import pandas as pd
from openai import OpenAI
try:
    import google.generativeai as genai
except Exception:  # optional import; only needed for provider="gemini"
    genai = None


def _get_client_for_provider(provider: str, api_key: Optional[str] = None):
    """Return an OpenAI-compatible client for the selected provider.

    provider:
        - "openai": uses OPENAI_API_KEY
        - "openrouter": uses OPENROUTER_API_KEY and OpenRouter base_url
        - "ollama": uses local Ollama server (no real key required)
    """
    p = (provider or "openai").lower()
    if p == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set in environment")
        return OpenAI(api_key=key)
    if p == "openrouter":
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment")
        return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    if p == "ollama":
        # Ollama exposes an OpenAI-compatible API at localhost:11434
        # A dummy API key is acceptable; some clients require a non-empty key
        key = api_key or os.getenv("OLLAMA_API_KEY", "ollama")
        return OpenAI(api_key=key, base_url="http://localhost:11434/v1")
    if p == "gemini":
        if genai is None:
            raise ImportError("google-generativeai is not installed. Run: pip install google-generativeai")
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set in environment")
        genai.configure(api_key=key)
        # For Gemini we return the module itself; call sites branch on provider.
        return genai
    raise ValueError(f"Unsupported provider: {provider}")


def generate_summary(df: pd.DataFrame, model: str = "gpt-4o-mini", provider: str = "openai") -> str:
    """Generate a concise plain-English summary of the dataframe.

    Includes: dataset shape, first 5 rows (CSV), and summary statistics (CSV).
    """
    client = _get_client_for_provider(provider)
    rows, cols = df.shape
    sample_csv = df.head(5).to_csv(index=False)
    desc = df.describe(include='all').transpose().fillna("")
    desc_text = desc.to_csv()

    prompt = (
        f"You are a data analyst assistant. Provide a concise, plain-English summary of the dataset.\n"
        f"Dataset shape: {rows} rows x {cols} columns.\n"
        f"Here are the first 5 rows as CSV:\n{sample_csv}\n"
        f"Here is a CSV of summary statistics:\n{desc_text}\n"
        f"Give: (1) three top insights, (2) data quality issues to look at, (3) suggested next visualizations or analyses. Keep it short and actionable."
    )

    try:
        if provider == "gemini":
            try:
                gmodel = client.GenerativeModel(model)
                resp = gmodel.generate_content(prompt)
            except Exception as e:
                # Some accounts only expose the "-latest" aliases; try a fallback
                msg = str(e).lower()
                if "404" in msg and not model.endswith("-latest"):
                    gmodel = client.GenerativeModel(model + "-latest")
                    resp = gmodel.generate_content(prompt)
                else:
                    raise
            return getattr(resp, "text", str(resp)).strip()
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        # Fallback summary if provider is unavailable or auth fails
        msg = str(e).lower()
        if any(k in msg for k in ["insufficient_quota", "api key", "authentication", "connection refused", "failed to establish a new connection"]):
            missing = df.isna().mean().sort_values(ascending=False).head(5)
            top_missing = ", ".join([f"{c}: {p:.1%}" for c, p in missing.items()])
            cols_list = ", ".join(df.columns[:10]) + ("..." if df.shape[1] > 10 else "")
            return (
                "AI service unavailable. Here's a basic summary instead:\n"
                f"- Shape: {rows} rows x {cols} columns\n"
                f"- Columns: {cols_list}\n"
                f"- Most missing (top 5): {top_missing if len(missing)>0 else 'none'}\n"
                "- Tip: ensure the correct provider is selected and API key or local server is set."
            )
    raise RuntimeError(f"AI request failed: {e}")


def answer_question(question: str, df: pd.DataFrame, model: str = "gpt-4o-mini", provider: str = "openai") -> str:
    """Answer a user question about the dataframe using a small in-context sample."""
    client = _get_client_for_provider(provider)
    cols = df.columns.tolist()
    sample = df.head(10).to_csv(index=False)
    prompt = (
        f"You are a data-savvy assistant. The dataset has columns: {cols}.\n"
        f"Here are the first 10 rows as CSV:\n{sample}\n"
        f"Answer this question about the dataset: {question}\n"
        f"If the question requires computation, provide the answer and a short explanation. "
        f"If it's ambiguous, ask a clarifying question."
    )
    try:
        if provider == "gemini":
            try:
                gmodel = client.GenerativeModel(model)
                resp = gmodel.generate_content(prompt)
            except Exception as e:
                msg = str(e).lower()
                if "404" in msg and not model.endswith("-latest"):
                    gmodel = client.GenerativeModel(model + "-latest")
                    resp = gmodel.generate_content(prompt)
                else:
                    raise
            return getattr(resp, "text", str(resp)).strip()
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ["insufficient_quota", "api key", "authentication", "connection refused", "failed to establish a new connection"]):
            return (
                "AI chat unavailable (provider/auth). Set the appropriate API key or ensure Ollama is running. "
                "Meanwhile, use the summary stats and visualizations in the app to explore the data."
            )
        raise RuntimeError(f"AI request failed: {e}")


def suggest_chart(question: str, df: pd.DataFrame, model: str = "gpt-4o-mini", provider: str = "openai") -> dict:
    """Return a structured chart suggestion as a dict.

    Expected schema:
    {
      "chart": "histogram|bar|line|scatter",
      "x": "column_name_or_null",
      "y": "column_name_or_null",
      "color": "column_name_or_null",
      "nbins": int or null,
      "top_n": int or null
    }

    If the model fails, a heuristic fallback is returned.
    """
    cols = df.columns.tolist()
    dtypes = {c: str(df[c].dtype) for c in cols}
    numeric = [c for c in cols if str(df[c].dtype).startswith(("int", "float", "Int", "Float"))]
    categorical = [c for c in cols if c not in numeric]

    client = _get_client_for_provider(provider)
    sample = df.head(20).to_csv(index=False)
    schema = (
        "Respond with ONLY valid JSON matching this schema without explanations: "
        "{chart: 'histogram'|'bar'|'line'|'scatter', x: string|null, y: string|null, color: string|null, nbins: number|null, top_n: number|null}. "
        "Use only column names provided."
    )
    prompt = (
        f"You are a data visualization planner. Columns and dtypes: {dtypes}.\n"
        f"User request: {question}.\n"
        f"First 20 rows CSV:\n{sample}\n"
        f"Choose a chart and fields best matching the request and data. {schema}"
    )
    try:
        if provider == "gemini":
            gmodel = client.GenerativeModel(model)
            resp = gmodel.generate_content(prompt)
            text = getattr(resp, "text", "{}")
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful visualization planner."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.1,
            )
            text = resp.choices[0].message.content or "{}"
        # Attempt to extract JSON
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        out = json.loads(text)
        if not isinstance(out, dict):
            raise ValueError("Invalid JSON output")
        # Minimal validation and defaults
        out.setdefault("chart", "histogram")
        out.setdefault("x", None)
        out.setdefault("y", None)
        out.setdefault("color", None)
        out.setdefault("nbins", None)
        out.setdefault("top_n", 20)
        return out
    except Exception:
        # Heuristic fallback: pick a simple chart
        if numeric:
            return {"chart": "histogram", "x": numeric[0], "y": None, "color": None, "nbins": 30, "top_n": 20}
        if categorical:
            return {"chart": "bar", "x": categorical[0], "y": None, "color": None, "nbins": None, "top_n": 20}
        return {"chart": "histogram", "x": cols[0] if cols else None, "y": None, "color": None, "nbins": 30, "top_n": 20}
