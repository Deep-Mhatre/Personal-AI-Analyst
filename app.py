import os
try:
    from dotenv import load_dotenv  # type: ignore
    _DOTENV_AVAILABLE = True
except Exception:  # fallback if python-dotenv is missing
    def load_dotenv(*args, **kwargs):
        return False
    _DOTENV_AVAILABLE = False
import streamlit as st
import pandas as pd
import importlib

from utils.data_cleaning import detect_types, remove_duplicates, handle_nulls, clean_dataframe
from utils.visualization import plot_histogram, plot_bar, plot_line, plot_scatter
import utils.ai_summary as ai_summary
import hashlib

load_dotenv()  # load .env if present (no-op if package missing)
st.set_page_config(page_title="Personal AI Analyst", layout="wide")

# App title
st.title("Personal AI Analyst")
st.markdown("Upload a CSV or Excel file, clean it, explore it with interactive charts, and get AI-generated insights.")

# Sidebar: file upload and options
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
preview_rows = st.sidebar.number_input("Preview rows", min_value=1, max_value=1000, value=5)

# Cleaning options
st.sidebar.header("Data Cleaning")
handle_null_strategy = st.sidebar.selectbox("Null handling strategy", ["leave", "drop_rows", "fill_mean", "fill_median", "fill_mode"]) 
remove_dup = st.sidebar.checkbox("Remove duplicate rows", value=True)

# Visualization options
st.sidebar.header("Visualization")
chart_type = st.sidebar.selectbox("Chart type", ["Histogram", "Bar", "Line"]) 
hist_bins = st.sidebar.slider("Histogram bins", min_value=5, max_value=100, value=30)
bar_topn = st.sidebar.slider("Bar: top N", min_value=5, max_value=100, value=20)

# AI options
st.sidebar.header("AI")
provider_friendly = st.sidebar.selectbox(
    "Provider",
    ["OpenAI", "OpenRouter", "Ollama (local)", "Gemini"]
)
provider_map = {
    "OpenAI": "openai",
    "OpenRouter": "openrouter",
    "Ollama (local)": "ollama",
    "Gemini": "gemini",
}
provider = provider_map[provider_friendly]

if provider == "openai":
    model_choice = st.sidebar.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o-mini-2024-07-18", "gpt-3.5-turbo"],
        index=0,
    )
elif provider == "openrouter":
    st.sidebar.caption("Uses OpenRouter (set OPENROUTER_API_KEY). Model ids vary; see their model list.")
    model_choice = st.sidebar.text_input(
        "Model id",
        value="openrouter/openai/gpt-4o-mini",
        help="Example: openrouter/openai/gpt-4o-mini, anthropic/claude-3-haiku, mistralai/mistral-7b-instruct"
    )
elif provider == "ollama":
    st.sidebar.caption("Local LLM via Ollama. Install Ollama and run the model locally.")
    model_choice = st.sidebar.text_input("Model id", value="llama3.1:8b", help="Example: llama3.1:8b, mistral:7b")
else:  # gemini
    st.sidebar.caption("Google Gemini API. Set GEMINI_API_KEY in your environment.")
    # Optional: dynamically fetch available models from your account
    if "gemini_models" not in st.session_state:
        st.session_state["gemini_models"] = None
    if st.sidebar.button("Refresh Gemini models"):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            st.sidebar.error("GEMINI_API_KEY not set")
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=key)
                models = genai.list_models()
                options = []
                for m in models:
                    supported = getattr(m, "supported_generation_methods", []) or getattr(m, "generation_methods", [])
                    if any("generateContent" in s or "generate_content" in s for s in supported):
                        name = getattr(m, "name", "")
                        if name.startswith("models/"):
                            name = name.split("/", 1)[1]
                        if name:
                            options.append(name)
                # Deduplicate and sort for usability
                options = sorted(set(options))
                if options:
                    st.session_state["gemini_models"] = options
                    st.sidebar.success(f"Loaded {len(options)} models")
                else:
                    st.sidebar.warning("No compatible Gemini models found; using defaults.")
            except Exception as e:
                st.sidebar.error(f"Failed to list models: {e}")

    default_models = ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]
    gemini_models = st.session_state.get("gemini_models") or default_models
    model_choice = st.sidebar.selectbox("Model", gemini_models, index=0)
openai_key = os.getenv("OPENAI_API_KEY")
if provider == "openai" and not openai_key:
    st.sidebar.warning("Set OPENAI_API_KEY in your environment for AI features.")
if provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
    st.sidebar.warning("Set OPENROUTER_API_KEY to use OpenRouter.")
if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
    st.sidebar.warning("Set GEMINI_API_KEY to use Gemini.")

# Load data
df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

if df is not None:
    st.subheader("Data preview")
    st.dataframe(df.head(preview_rows))

    # Detect types
    with st.expander("Column types and basic info"):
        types = detect_types(df)
        st.table(types)

    # Data cleaning
    if st.button("Run cleaning"):
        st.info("Running cleaning...")
        try:
            cleaned = clean_dataframe(
                df.copy(), null_strategy=handle_null_strategy, drop_duplicates=remove_dup
            )
            st.success("Cleaning complete. Preview below.")
            st.dataframe(cleaned.head(preview_rows))
            df = cleaned
        except Exception as e:
            st.error(f"Cleaning failed: {e}")

    # Visualization column options (only show after data is loaded)
    cols = df.columns.tolist()
    vis_col = st.sidebar.selectbox(
        "Select column (after upload)", options=["-- None --"] + cols, index=0, key="vis_col"
    )
    # For line chart, optional x-axis
    index_col = None
    if chart_type == "Line":
        index_col_choice = st.sidebar.selectbox(
            "X-axis (optional)", options=["Row index"] + cols, index=0, key="x_axis"
        )
        index_col = None if index_col_choice == "Row index" else index_col_choice

    # EDA: summary statistics
    st.subheader("Exploratory Data Analysis")
    try:
        st.write(df.describe(include='all').transpose())
    except Exception as e:
        st.warning(f"Could not compute full describe: {e}")

    # Visualization
    st.subheader("Visualization")
    if vis_col and vis_col != "-- None --":
        try:
            if chart_type == "Histogram":
                fig = plot_histogram(df, vis_col, nbins=hist_bins)
            elif chart_type == "Bar":
                fig = plot_bar(df, vis_col, top_n=bar_topn)
            else:
                fig = plot_line(df, vis_col, index_col=index_col)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to create chart: {e}")

    # AI-assisted visualization
    st.subheader("AI-assisted visualization")
    ai_viz_q = st.text_input("Describe the chart you want (e.g., 'scatter plot of sales vs profit, color by region')")
    if st.button("AI: Generate chart") and df is not None:
        if provider == "openai" and not openai_key:
            st.error("OpenAI API key not found. Set OPENAI_API_KEY in your environment.")
        elif provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
            st.error("OpenRouter selected but OPENROUTER_API_KEY is not set.")
        elif provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            st.error("Gemini selected but GEMINI_API_KEY is not set.")
        else:
            with st.spinner("Asking AI for a chart suggestion..."):
                try:
                    ai_summary = importlib.reload(ai_summary)
                    plan = ai_summary.suggest_chart(ai_viz_q or "Suggest the most informative chart.", df, model=model_choice, provider=provider)
                    st.caption("AI chart plan")
                    st.json(plan)
                    chart = str(plan.get("chart", "")).lower()
                    if chart == "histogram" and plan.get("x"):
                        nb = plan.get("nbins") or hist_bins
                        fig = plot_histogram(df, plan["x"], nbins=int(nb))
                    elif chart == "bar" and plan.get("x"):
                        tn = plan.get("top_n") or bar_topn
                        fig = plot_bar(df, plan["x"], top_n=int(tn))
                    elif chart == "line" and (plan.get("y") or plan.get("x")):
                        ycol = plan.get("y") or vis_col or df.columns[0]
                        xcol = plan.get("x")
                        fig = plot_line(df, ycol, index_col=xcol)
                    elif chart == "scatter" and plan.get("x") and plan.get("y"):
                        fig = plot_scatter(df, plan["x"], plan["y"], color=plan.get("color"))
                    else:
                        # fallback to a quick histogram suggestion
                        target = plan.get("x") or vis_col or df.columns[0]
                        fig = plot_histogram(df, target, nbins=hist_bins)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"AI visualization failed: {e}")

    # Helpers for light caching to reduce token usage
    def _df_signature(frame: pd.DataFrame) -> str:
        try:
            h = hashlib.md5()
            h.update(str(frame.shape).encode())
            h.update(",".join(list(map(str, frame.columns))).encode())
            h.update(frame.head(100).to_csv(index=False).encode())
            return h.hexdigest()
        except Exception:
            return str(frame.shape)

    if "ai_cache" not in st.session_state:
        st.session_state["ai_cache"] = {}
    ai_cache = st.session_state["ai_cache"]

    # AI-generated summary
    st.subheader("AI-generated insights")
    if st.button("Generate AI summary"):
        # Basic provider checks
        if provider == "openai" and not openai_key:
            st.error("OpenAI API key not found. Set OPENAI_API_KEY in your environment.")
        elif provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
            st.error("OpenRouter selected but OPENROUTER_API_KEY is not set.")
        else:
            try:
                with st.spinner("Generating summary..."):
                    sig = _df_signature(df)
                    cache_key = f"summary::{provider}::{model_choice}::{sig}"
                    if cache_key in ai_cache:
                        summary = ai_cache[cache_key]
                    else:
                        # ensure latest version of ai_summary is used when file changes
                        ai_summary = importlib.reload(ai_summary)
                        summary = ai_summary.generate_summary(df, model=model_choice, provider=provider)
                        ai_cache[cache_key] = summary
                    st.markdown(summary)
            except Exception as e:
                st.error(f"AI summary failed: {e}")

    # Chat with data (optional)
    st.subheader("Chat with your data")
    question = st.text_input("Ask a question about the data (e.g., 'Which column has the most missing values?')")
    if st.button("Ask") and question:
        if provider == "openai" and not openai_key:
            st.error("OpenAI API key not found. Set OPENAI_API_KEY in your environment.")
        elif provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
            st.error("OpenRouter selected but OPENROUTER_API_KEY is not set.")
        else:
            try:
                with st.spinner("Asking the model..."):
                    sig = _df_signature(df)
                    cache_key = f"q::{provider}::{model_choice}::{sig}::{hashlib.md5(question.encode()).hexdigest()}"
                    if cache_key in ai_cache:
                        ans = ai_cache[cache_key]
                    else:
                        ai_summary = importlib.reload(ai_summary)
                        ans = ai_summary.answer_question(question, df, model=model_choice, provider=provider)
                        ai_cache[cache_key] = ans
                    st.markdown(ans)
            except Exception as e:
                st.error(f"AI question failed: {e}")

else:
    st.info("Upload a CSV/XLSX file to get started. A sample dataset is included in `datasets/sample.csv`.")
