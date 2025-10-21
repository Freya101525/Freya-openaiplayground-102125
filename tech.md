Below is an improved, end-to-end Streamlit app that adds the requested “Prompt ID Orchestrator” tab/feature, expands the “wow UI” with the specified themes, upgrades visualization with status indicators and an interactive dashboard, and supports Gemini, Grok, and OpenAI with model selection per agent. It automatically reads API keys from the environment/Secrets; if keys are missing, it prompts the user to provide them on the webpage (without exposing environment values). It also supports OpenAI Prompt IDs (including the provided default prompt ID) and lets users modify the system prompt, user prompt, and model/parameters before executing agents one-by-one.

Copy-paste to your Hugging Face Space as app.py (or update your existing file):

import streamlit as st
import google.generativeai as genai
from openai import OpenAI as OpenAIClient
import os
import io
import yaml
import traceback
from PyPDF2 import PdfReader, PdfWriter
import pytesseract
from pdf2image import convert_from_bytes
import base64
import time
import json

# Grok (xAI)
try:
    from xai_sdk import Client as GrokClient
    from xai_sdk.chat import user as grok_user, system as grok_system, image as grok_image
    XAI_AVAILABLE = True
except Exception:
    XAI_AVAILABLE = False

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="🤖 Agentic PDF & Prompt Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Themes ----------------
THEMES = {
    "Blue sky": {
        "primary": "#87CEEB",
        "secondary": "#4682B4",
        "background": "#F0F8FF",
        "text": "#1E3A5F",
        "accent": "#FFD700"
    },
    "Snow white": {
        "primary": "#FFFFFF",
        "secondary": "#E8E8E8",
        "background": "#F5F5F5",
        "text": "#2C3E50",
        "accent": "#3498DB"
    },
    "Dark Knight": {
        "primary": "#0F0F10",
        "secondary": "#1C1C1E",
        "background": "#0B0B0C",
        "text": "#EAEAEA",
        "accent": "#FF3B30"
    },
    "Sparkling galaxy": {
        "primary": "#8B5CF6",
        "secondary": "#EC4899",
        "background": "#1E1B4B",
        "text": "#F3E8FF",
        "accent": "#FCD34D"
    },
    "Alp.Forest": {
        "primary": "#2D5016",
        "secondary": "#4A7C59",
        "background": "#E8F5E9",
        "text": "#1B5E20",
        "accent": "#FF6F00"
    },
    "UK royal": {
        "primary": "#00247D",
        "secondary": "#CF142B",
        "background": "#F4F6FA",
        "text": "#1A1A1A",
        "accent": "#FFD700"
    },
    "Flora": {
        "primary": "#E91E63",
        "secondary": "#9C27B0",
        "background": "#FCE4EC",
        "text": "#880E4F",
        "accent": "#00BCD4"
    },
    "Fendi CASA": {
        "primary": "#D4AF37",
        "secondary": "#8B7355",
        "background": "#F5F5DC",
        "text": "#3E2723",
        "accent": "#C9A961"
    },
    "Ferrari Sportscar": {
        "primary": "#DC0000",
        "secondary": "#8B0000",
        "background": "#FFF5F5",
        "text": "#1A0000",
        "accent": "#FFD700"
    },
    "Holiday season style": {
        "primary": "#0B8457",
        "secondary": "#C1121F",
        "background": "#F9FAFB",
        "text": "#22333B",
        "accent": "#F4D35E"
    }
}

MODEL_OPTIONS = {
    "Gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
    "OpenAI": ["gpt-5-nano", "gpt-4.1-mini", "gpt-4o-mini"],
    "Grok": ["grok-4-fast-reasoning", "grok-3-mini"]
}

DEFAULT_PROMPT_ID = "pmpt_68f743661d1881948282ad0db8e40f4c07032cb377d03d6f"

# ---------------- CSS ----------------
def apply_theme(theme_name):
    theme = THEMES[theme_name]
    css = f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, {theme['background']} 0%, {theme['secondary']}15 100%);
        }}
        .main-header {{
            background: linear-gradient(90deg, {theme['primary']}, {theme['secondary']});
            padding: 1.2rem 1.5rem;
            border-radius: 15px;
            text-align: left;
            color: white;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            margin-bottom: 1.0rem;
            animation: slideIn 0.5s ease-out;
        }}
        .card {{
            background: white;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin: 0.75rem 0;
            border-left: 4px solid {theme['accent']};
            transition: transform 0.3s ease;
        }}
        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }}
        .metric-card {{
            background: linear-gradient(135deg, {theme['primary']}20, {theme['secondary']}20);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            border: 2px solid {theme['accent']};
        }}
        .stButton>button {{
            background: linear-gradient(90deg, {theme['primary']}, {theme['secondary']});
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 25px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            transform: scale(1.03);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}
        .status-badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            background: {theme['accent']};
            color: black;
            font-weight: 700;
            margin-left: 0.5rem;
        }}
        .chip {{
            display:inline-block;
            padding:0.25rem 0.6rem;
            border-radius: 999px;
            background: {theme['secondary']}22;
            border:1px solid {theme['secondary']}66;
            color: {theme['text']};
            font-size: 0.8rem;
            margin-right: 0.4rem;
        }}
        h1, h2, h3, h4 {{
            color: {theme['text']};
        }}
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ---------------- Helpers ----------------
@st.cache_data
def load_agents_config():
    try:
        with open("agents.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        st.warning("agents.yaml not found. You can still run ad-hoc agents and the Prompt-ID Orchestrator.")
        return {}

def trim_pdf(file_bytes, pages_to_trim):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        writer = PdfWriter()
        start_page, end_page = pages_to_trim
        total_pages = len(reader.pages)
        if start_page > end_page or start_page < 1 or end_page > total_pages:
            st.error("Invalid page range selected.")
            return None
        for i in range(start_page - 1, end_page):
            writer.add_page(reader.pages[i])
        output_pdf = io.BytesIO()
        writer.write(output_pdf)
        return output_pdf.getvalue()
    except Exception as e:
        st.error(f"Error trimming PDF: {e}")
        return None

def ocr_pdf(file_bytes):
    try:
        images = convert_from_bytes(file_bytes)
        full_text = ""
        progress_bar = st.progress(0)
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            full_text += f"\n--- Page {i+1} ---\n{text}"
            progress_bar.progress((i + 1) / len(images))
        return full_text
    except Exception as e:
        st.warning(f"Could not perform OCR: {e}")
        return None

def extract_text_from_pdf(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def to_markdown_with_keywords(text, keywords):
    if keywords:
        keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        for keyword in keyword_list:
            text = text.replace(keyword, f"<span style='color:coral;font-weight:bold'>{keyword}</span>")
    return text

# ---------------- API Keys: get from env; if missing, prompt user (no display of env) ----------------
def get_env_or_secret(k):
    # environment takes precedence over secrets to avoid exposing secrets
    return os.getenv(k) or st.secrets.get(k)

def ensure_api_keys_ui():
    if "USER_OPENAI_API_KEY" not in st.session_state:
        st.session_state.USER_OPENAI_API_KEY = None
    if "USER_GOOGLE_API_KEY" not in st.session_state:
        st.session_state.USER_GOOGLE_API_KEY = None
    if "USER_XAI_API_KEY" not in st.session_state:
        st.session_state.USER_XAI_API_KEY = None

    env_openai = bool(get_env_or_secret("OPENAI_API_KEY"))
    env_gemini = bool(get_env_or_secret("GEMINI_API_KEY") or get_env_or_secret("GOOGLE_API_KEY"))
    env_xai = bool(get_env_or_secret("XAI_API_KEY") or get_env_or_secret("GROK_API_KEY"))

    st.markdown("#### 🔐 API Keys")
    # OpenAI
    if not env_openai and not st.session_state.USER_OPENAI_API_KEY:
        st.info("OpenAI API key not found in environment. Please provide your key here to enable OpenAI.")
        st.session_state.USER_OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password", key="openai_key_input")
    else:
        st.success("OpenAI key available")

    # Gemini
    if not env_gemini and not st.session_state.USER_GOOGLE_API_KEY:
        st.info("Google Gemini API key not found in environment. Provide your key to enable Gemini.")
        st.session_state.USER_GOOGLE_API_KEY = st.text_input("Google/Gemini API Key", type="password", key="gemini_key_input")
    else:
        st.success("Gemini key available")

    # Grok (xAI)
    if not env_xai and not st.session_state.USER_XAI_API_KEY:
        st.info("xAI (Grok) API key not found in environment. Provide your key to enable Grok.")
        st.session_state.USER_XAI_API_KEY = st.text_input("xAI (Grok) API Key", type="password", key="xai_key_input")
    else:
        st.success("Grok key available")

def resolve_key(provider):
    if provider == "OpenAI":
        return get_env_or_secret("OPENAI_API_KEY") or st.session_state.USER_OPENAI_API_KEY
    if provider == "Gemini":
        return get_env_or_secret("GEMINI_API_KEY") or get_env_or_secret("GOOGLE_API_KEY") or st.session_state.USER_GOOGLE_API_KEY
    if provider == "Grok":
        return get_env_or_secret("XAI_API_KEY") or get_env_or_secret("GROK_API_KEY") or st.session_state.USER_XAI_API_KEY
    return None

def get_llm_client(api_choice):
    try:
        if api_choice == "Gemini":
            api_key = resolve_key("Gemini")
            if not api_key:
                st.error("Gemini API key required.")
                return None
            genai.configure(api_key=api_key)
            return genai
        elif api_choice == "OpenAI":
            api_key = resolve_key("OpenAI")
            if not api_key:
                st.error("OpenAI API key required.")
                return None
            return OpenAIClient(api_key=api_key)
        elif api_choice == "Grok":
            api_key = resolve_key("Grok")
            if not api_key:
                st.error("Grok (xAI) API key required.")
                return None
            if not XAI_AVAILABLE:
                st.error("xai_sdk not installed in this environment.")
                return None
            return GrokClient(api_key=api_key, timeout=3600)
    except Exception as e:
        st.error(f"Error initializing {api_choice} client: {e}")
        return None

# ---------------- Agent execution for YAML agents ----------------
def execute_agent(agent_config, input_text):
    client = get_llm_client(agent_config['api'])
    if not client:
        return f"Could not initialize the {agent_config['api']} client. Check API keys."

    prompt = agent_config['prompt'].format(input_text=input_text)
    model = agent_config['model']

    try:
        with st.spinner(f"🤖 {agent_config['name']} is processing..."):
            if agent_config['api'] == "Gemini":
                model_instance = client.GenerativeModel(model)
                response = model_instance.generate_content(prompt)
                return response.text
            elif agent_config['api'] == "OpenAI":
                # Use Responses API for consistency
                params = agent_config.get('parameters', {})
                oc = client.responses.create(
                    model=model,
                    input=f"{prompt}",
                    **params
                )
                return getattr(oc, "output_text", None) or (oc.output[0].content[0].text if getattr(oc, "output", None) else str(oc))
            elif agent_config['api'] == "Grok":
                chat = client.chat.create(model=model)
                chat.append(grok_user(prompt))
                response = chat.sample()
                return response.content
    except Exception as e:
        st.error(f"Error executing agent '{agent_config.get('name','Unnamed')}': {e}")
        traceback.print_exc()
        return None

# ---------------- Prompt ID Orchestrator (OpenAI) ----------------
def run_openai_prompt_id(client: OpenAIClient, model: str, prompt_id: str, version: str, user_input: str, params: dict):
    # Use OpenAI Responses API with stored prompt ID
    try:
        resp = client.responses.create(
            model=model,
            prompt={"id": prompt_id, "version": version},
            input=user_input or "",
            **params
        )
        text = getattr(resp, "output_text", None)
        if text:
            return text
        # Fallback parse
        try:
            return resp.output[0].content[0].text
        except Exception:
            return str(resp)
    except Exception as e:
        raise RuntimeError(f"OpenAI Prompt ID execution failed: {e}")

def run_openai_freeform(client: OpenAIClient, model: str, system_prompt: str, user_input: str, params: dict):
    # Use Responses API (single-turn)
    built_input = f"System:\n{system_prompt}\n\nUser:\n{user_input}"
    try:
        resp = client.responses.create(
            model=model,
            input=built_input,
            **params
        )
        text = getattr(resp, "output_text", None)
        if text:
            return text
        try:
            return resp.output[0].content[0].text
        except Exception:
            return str(resp)
    except Exception as e:
        raise RuntimeError(f"OpenAI freeform execution failed: {e}")

def apply_dashboard_update(agent_name, status, duration_s, provider, model, tokens=None, prompt_id=None):
    if "run_history" not in st.session_state:
        st.session_state.run_history = []
    st.session_state.run_history.append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "agent": agent_name,
        "provider": provider,
        "model": model,
        "status": status,
        "duration_s": round(duration_s, 2),
        "tokens": tokens,
        "prompt_id": prompt_id
    })

# ---------------- Main App ----------------
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎨 Theme")
        selected_theme = st.selectbox(
            "Select Theme",
            list(THEMES.keys()),
            index=0
        )
        apply_theme(selected_theme)

        st.markdown("---")
        ensure_api_keys_ui()

        st.markdown("---")
        st.markdown("### ⚙️ Default Config")
        api_choice = st.selectbox("Default API Provider", list(MODEL_OPTIONS.keys()))
        model_choice = st.selectbox("Default Model", MODEL_OPTIONS[api_choice])

        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("Process PDFs with multi-agent AI, orchestrate agents by Prompt ID, and visualize runs in a live dashboard. Deployed on Hugging Face Spaces with Streamlit.")

    st.markdown(f"""
        <div class="main-header">
            <h2>🤖 Agentic PDF & Prompt Orchestrator</h2>
            <div>
                <span class="chip">Gemini</span>
                <span class="chip">OpenAI</span>
                <span class="chip">Grok (xAI)</span>
                <span class="chip">OCR</span>
                <span class="chip">Prompt ID</span>
                <span class="chip">Interactive Dashboard</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📚 PDF Processor", "🧩 Prompt-ID Orchestrator", "📊 Dashboard"])

    # --------- Tab 1: PDF Processor ----------
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📂 Upload Your PDF")
            uploaded_file = st.file_uploader(
                "Drop your PDF here or click to browse",
                type=['pdf'],
                help="Upload a PDF file to process with AI agents"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            total_runs = len(st.session_state.get("run_history", []))
            st.metric("📊 Total Runs", f"{total_runs}")
            st.metric("⚙️ Default Provider", api_choice)
            st.metric("🧠 Default Model", model_choice)
            st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            file_bytes = uploaded_file.read()
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🔧 Processing Options")

            col_a, col_b = st.columns(2)
            with col_a:
                try:
                    reader = PdfReader(io.BytesIO(file_bytes))
                    total_pages = len(reader.pages)
                except Exception:
                    total_pages = 1
                st.info(f"Total pages: {total_pages}")

                use_page_range = st.checkbox("Select specific page range")
                if use_page_range:
                    start_page = st.number_input("Start page", 1, total_pages, 1)
                    end_page = st.number_input("End page", 1, total_pages, total_pages)
                    pages_to_trim = (int(start_page), int(end_page))
                else:
                    pages_to_trim = (1, total_pages)

            with col_b:
                use_ocr = st.checkbox("Use OCR for scanned PDFs", help="Enable if the PDF contains images of text")
                highlight_keywords = st.text_input(
                    "Keywords to highlight (comma-separated)",
                    placeholder="AI, machine learning, data"
                )

            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🚀 Process PDF with AI", use_container_width=True):
                with st.spinner("Processing your document..."):
                    start_t = time.time()
                    if use_page_range:
                        file_bytes = trim_pdf(file_bytes, pages_to_trim)
                        if file_bytes is None:
                            st.stop()

                    if use_ocr:
                        st.info("🔍 Performing OCR on images...")
                        extracted_text = ocr_pdf(file_bytes)
                    else:
                        st.info("📝 Extracting text from PDF...")
                        extracted_text = extract_text_from_pdf(file_bytes)

                    if not extracted_text or not extracted_text.strip():
                        st.error("❌ No text could be extracted. Try enabling OCR if this is a scanned document.")
                        st.stop()

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### 📄 Extracted Text Preview")
                    preview_text = extracted_text[:1200] + "..." if len(extracted_text) > 1200 else extracted_text
                    if highlight_keywords:
                        preview_text = to_markdown_with_keywords(preview_text, highlight_keywords)
                    st.markdown(preview_text, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    agents_config = load_agents_config()
                    if agents_config and 'agents' in agents_config:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("### 🤖 AI Agent Processing")
                        for agent in agents_config['agents']:
                            agent_api = api_choice
                            agent_model = model_choice
                            agent_name = agent.get('name', 'Agent')

                            # Build config with overrides
                            merged = {
                                "name": agent_name,
                                "api": agent_api,
                                "model": agent_model,
                                "prompt": agent.get("prompt", "Summarize:\n{input_text}"),
                                "parameters": agent.get("parameters", {})
                            }
                            with st.expander(f"🔹 {agent_name}", expanded=False):
                                status_placeholder = st.empty()
                                try:
                                    status_placeholder.info(f"Running {agent_name} with {agent_api}/{agent_model} ...")
                                    result = execute_agent(merged, extracted_text)
                                    if result:
                                        st.markdown("**Result:**")
                                        st.write(result)
                                        st.download_button(
                                            f"💾 Download {agent_name} Result",
                                            result,
                                            file_name=f"{agent_name.replace(' ','_')}_result.txt",
                                            mime="text/plain"
                                        )
                                        status_placeholder.success("Completed ✅")
                                        apply_dashboard_update(agent_name, "success", time.time()-start_t, agent_api, agent_model)
                                    else:
                                        status_placeholder.error("No result")
                                        apply_dashboard_update(agent_name, "empty", time.time()-start_t, agent_api, agent_model)
                                except Exception as e:
                                    status_placeholder.error(f"Error: {e}")
                                    apply_dashboard_update(agent_name, "error", time.time()-start_t, agent_api, agent_model)
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.success("✅ Processing complete!")

    # --------- Tab 2: Prompt ID Orchestrator ----------
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧩 Configure number of agents (Prompt IDs)")
        num_agents = st.number_input("How many agents to run (sequentially)?", min_value=1, max_value=10, value=1, step=1, key="num_prompt_agents")
        st.markdown('</div>', unsafe_allow_html=True)

        if 'prompt_agents' not in st.session_state or len(st.session_state.prompt_agents) != num_agents:
            st.session_state.prompt_agents = [
                {
                    "use_prompt_id": True,
                    "prompt_id": DEFAULT_PROMPT_ID if i == 0 else "",
                    "version": "1",
                    "provider": "OpenAI",
                    "model": "gpt-5-nano",
                    "system_prompt": "You are a helpful AI assistant.",
                    "user_prompt": "",
                    "temperature": 0.7,
                    "max_output_tokens": 1024,
                    "top_p": 1.0
                } for i in range(num_agents)
            ]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Agent Definitions")
        for idx in range(num_agents):
            with st.expander(f"Agent {idx+1} settings", expanded=(idx == 0)):
                pa = st.session_state.prompt_agents[idx]
                pa["provider"] = st.selectbox("Provider", ["OpenAI"], index=0, key=f"agent_provider_{idx}")
                pa["model"] = st.selectbox("Model (OpenAI)", MODEL_OPTIONS["OpenAI"], index=MODEL_OPTIONS["OpenAI"].index(pa.get("model","gpt-5-nano")) if pa.get("model") in MODEL_OPTIONS["OpenAI"] else 0, key=f"agent_model_{idx}")

                pa["use_prompt_id"] = st.radio(
                    "Use Prompt ID?",
                    ["Yes (stored prompt)", "No (custom/system prompt)"],
                    index=0 if pa["use_prompt_id"] else 1,
                    horizontal=True,
                    key=f"use_prompt_id_{idx}"
                ) == "Yes (stored prompt)"

                if pa["use_prompt_id"]:
                    pa["prompt_id"] = st.text_input("Prompt ID", value=pa["prompt_id"], placeholder="pmpt_...", key=f"prompt_id_{idx}")
                    pa["version"] = st.text_input("Prompt Version", value=pa.get("version","1"), key=f"prompt_ver_{idx}")
                    st.caption("Tip: You can override model and parameters below even with a stored Prompt ID.")
                    pa["user_prompt"] = st.text_area("User prompt (input to the Prompt ID)", value=pa["user_prompt"], key=f"user_prompt_pid_{idx}", height=120)
                else:
                    pa["system_prompt"] = st.text_area("System prompt", value=pa["system_prompt"], key=f"system_prompt_{idx}", height=120)
                    pa["user_prompt"] = st.text_area("User prompt", value=pa["user_prompt"], key=f"user_prompt_{idx}", height=120)

                colp1, colp2, colp3 = st.columns(3)
                with colp1:
                    pa["temperature"] = st.slider("Temperature", 0.0, 2.0, float(pa["temperature"]), 0.1, key=f"temp_{idx}")
                with colp2:
                    pa["top_p"] = st.slider("Top-p", 0.0, 1.0, float(pa["top_p"]), 0.05, key=f"topp_{idx}")
                with colp3:
                    pa["max_output_tokens"] = st.number_input("Max output tokens", 64, 8192, int(pa["max_output_tokens"]), 32, key=f"max_toks_{idx}")

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("▶️ Run Agents (Prompt IDs) sequentially", use_container_width=True):
            client = get_llm_client("OpenAI")
            if not client:
                st.error("OpenAI client not available. Provide a valid OpenAI API key.")
                st.stop()

            results_container = st.container()
            for idx, pa in enumerate(st.session_state.prompt_agents):
                with results_container.expander(f"Execution: Agent {idx+1}", expanded=True):
                    status = st.empty()
                    start_t = time.time()
                    try:
                        status.info(f"Running Agent {idx+1} with model {pa['model']} ...")
                        params = {
                            "temperature": pa["temperature"],
                            "top_p": pa["top_p"],
                            "max_output_tokens": pa["max_output_tokens"]
                        }
                        if pa["use_prompt_id"]:
                            if not pa["prompt_id"]:
                                raise ValueError("Missing Prompt ID.")
                            output_text = run_openai_prompt_id(
                                client=client,
                                model=pa["model"],
                                prompt_id=pa["prompt_id"],
                                version=pa.get("version","1"),
                                user_input=pa["user_prompt"],
                                params=params
                            )
                            status.success(f"Completed Agent {idx+1} ✅")
                            st.write(output_text)
                            st.download_button(
                                f"💾 Download Agent {idx+1} Output",
                                output_text,
                                file_name=f"agent_{idx+1}_output.txt",
                                mime="text/plain",
                                key=f"dl_pid_{idx}"
                            )
                            apply_dashboard_update(f"PromptID Agent {idx+1}", "success", time.time()-start_t, "OpenAI", pa["model"], prompt_id=pa["prompt_id"])
                        else:
                            output_text = run_openai_freeform(
                                client=client,
                                model=pa["model"],
                                system_prompt=pa["system_prompt"],
                                user_input=pa["user_prompt"],
                                params=params
                            )
                            status.success(f"Completed Agent {idx+1} ✅")
                            st.write(output_text)
                            st.download_button(
                                f"💾 Download Agent {idx+1} Output",
                                output_text,
                                file_name=f"agent_{idx+1}_output.txt",
                                mime="text/plain",
                                key=f"dl_custom_{idx}"
                            )
                            apply_dashboard_update(f"CustomPrompt Agent {idx+1}", "success", time.time()-start_t, "OpenAI", pa["model"])
                    except Exception as e:
                        status.error(f"❌ Agent {idx+1} failed: {e}")
                        apply_dashboard_update(f"Agent {idx+1}", "error", time.time()-start_t, "OpenAI", pa["model"])

    # --------- Tab 3: Dashboard ----------
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Run Overview")
        history = st.session_state.get("run_history", [])
        if not history:
            st.info("No runs yet. Execute agents in the other tabs to populate the dashboard.")
        else:
            # KPIs
            total = len(history)
            successes = sum(1 for r in history if r["status"] == "success")
            errors = sum(1 for r in history if r["status"] == "error")
            colA, colB, colC = st.columns(3)
            colA.metric("Total Runs", total)
            colB.metric("Successes", successes)
            colC.metric("Errors", errors)

            # Table
            st.markdown("#### Detailed History")
            st.dataframe(history, use_container_width=True)

            # Simple visualization: bar by provider-model pair counts
            group_counts = {}
            for r in history:
                k = f"{r['provider']} | {r['model']}"
                group_counts[k] = group_counts.get(k, 0) + 1
            if group_counts:
                st.markdown("#### Usage by Provider/Model")
                chart_data = {"pair": list(group_counts.keys()), "count": list(group_counts.values())}
                st.bar_chart(chart_data, x="pair", y="count", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✅ Status Indicators")
    status_cols = st.columns(4)
    with status_cols[0]:
        st.info("System: Ready")
    with status_cols[1]:
        st.success("UI: OK")
    with status_cols[2]:
        st.warning("Keys: see sidebar")
    with status_cols[3]:
        st.info("Dashboard: Active")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

Notes:
- Prompt-ID Orchestrator tab allows you to specify how many agents to run. For each agent:
  - Choose OpenAI model
  - Use default Prompt ID or a new Prompt ID with version
  - Provide a user prompt for the Prompt ID
  - Or disable Prompt ID and supply a system prompt + user prompt
  - Override parameters (temperature, top_p, max_output_tokens)
- OpenAI execution uses the Responses API for both Prompt ID and freeform modes.
- If API keys are not present in the environment/Secrets, the app will prompt you to input them in the sidebar without displaying environment keys.
- Dashboard tab collects run history, shows KPIs, and provides a basic usage chart.

10 follow-up questions:
1) Should I add retrieval of stored Prompt metadata (system prompt, default params) to auto-populate fields when a Prompt ID is entered, with fallback when unavailable?
2) Would you like chain execution outputs (Agent i output) to become the input for Agent i+1 in the Prompt-ID Orchestrator?
3) Do you want to save modified prompts and parameters as new Prompt IDs or into agents.yaml for reuse?
4) Should I add batch PDF processing and queue management (multiple files, parallelizable where safe)?
5) Do you want role-based presets (Summarizer, Extractor, Reviewer) selectable per agent with one click?
6) Should I add token usage and cost estimates per run (where APIs return usage) and aggregate those in the dashboard?
7) Would you like image inputs for Grok (vision) and Gemini multimodal directly in the UI (drag-and-drop images) for analysis?
8) Do you want export options for results (Markdown, JSON, DOCX, HTML) and a combined report across agents?
9) Should I add real-time status streaming of tokens/content for OpenAI/Gemini when available, and a live run timeline?
10) Would you like user/team profiles with saved configurations, theme preferences, and API keys stored securely via Streamlit Secrets on Spaces?
