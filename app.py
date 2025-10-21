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
import time
import json

# Grok (xAI)
try:
    from xai_sdk import Client as GrokClient
    from xai_sdk.chat import user as grok_user, system as grok_system
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
    "Gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
    "OpenAI": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-nano"],
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
        .chain-indicator {{
            background: linear-gradient(90deg, {theme['primary']}40, {theme['accent']}40);
            padding: 0.5rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 3px solid {theme['accent']};
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
        with open("agents.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except FileNotFoundError:
        st.warning("agents.yaml not found. Using default configuration.")
        return {}
    except Exception as e:
        st.error(f"Error loading agents.yaml: {e}")
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

# ---------------- API Keys ----------------
def get_env_or_secret(k):
    return os.getenv(k) or st.secrets.get(k, "")

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
    
    if not env_openai and not st.session_state.USER_OPENAI_API_KEY:
        st.info("OpenAI API key not found. Provide key to enable OpenAI.")
        st.session_state.USER_OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password", key="openai_key_input")
    else:
        st.success("✅ OpenAI key available")

    if not env_gemini and not st.session_state.USER_GOOGLE_API_KEY:
        st.info("Google Gemini API key not found. Provide key to enable Gemini.")
        st.session_state.USER_GOOGLE_API_KEY = st.text_input("Google/Gemini API Key", type="password", key="gemini_key_input")
    else:
        st.success("✅ Gemini key available")

    if not env_xai and not st.session_state.USER_XAI_API_KEY:
        st.info("xAI (Grok) API key not found. Provide key to enable Grok.")
        st.session_state.USER_XAI_API_KEY = st.text_input("xAI (Grok) API Key", type="password", key="xai_key_input")
    else:
        st.success("✅ Grok key available")

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
                st.error("xai_sdk not installed.")
                return None
            return GrokClient(api_key=api_key, timeout=3600)
    except Exception as e:
        st.error(f"Error initializing {api_choice} client: {e}")
        return None

# ---------------- Agent execution ----------------
def execute_agent(agent_config, input_text):
    client = get_llm_client(agent_config['api'])
    if not client:
        return f"Could not initialize {agent_config['api']} client."

    prompt = agent_config['prompt'].format(input_text=input_text)
    model = agent_config['model']

    try:
        if agent_config['api'] == "Gemini":
            model_instance = client.GenerativeModel(model)
            response = model_instance.generate_content(prompt)
            return response.text
        elif agent_config['api'] == "OpenAI":
            params = agent_config.get('parameters', {})
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            return response.choices[0].message.content
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
    try:
        # Note: Adjust based on actual OpenAI Prompt ID API
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_input}],
            **params
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"OpenAI Prompt ID execution failed: {e}")

def run_openai_freeform(client: OpenAIClient, model: str, system_prompt: str, user_input: str, params: dict):
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **params
        )
        return response.choices[0].message.content
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
    # Initialize session state for chained outputs
    if "chain_outputs" not in st.session_state:
        st.session_state.chain_outputs = {}
    if "current_chain_input" not in st.session_state:
        st.session_state.current_chain_input = ""
    
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
        st.info("Process PDFs with multi-agent AI, orchestrate agents with chained execution, and visualize runs in a live dashboard.")

    st.markdown(f"""
        <div class="main-header">
            <h2>🤖 Agentic PDF & Prompt Orchestrator</h2>
            <div>
                <span class="chip">Gemini</span>
                <span class="chip">OpenAI</span>
                <span class="chip">Grok (xAI)</span>
                <span class="chip">Chain Execution</span>
                <span class="chip">Output Editing</span>
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
                use_ocr = st.checkbox("Use OCR for scanned PDFs", help="Enable if PDF contains images of text")
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
                        st.error("❌ No text could be extracted. Try enabling OCR.")
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
                            agent_api = agent.get('api', api_choice)
                            agent_model = agent.get('model', model_choice)
                            agent_name = agent.get('name', 'Agent')

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
                                    status_placeholder.info(f"Running {agent_name}...")
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
                                        status_placeholder.success("✅ Completed")
                                        apply_dashboard_update(agent_name, "success", time.time()-start_t, agent_api, agent_model)
                                    else:
                                        status_placeholder.error("No result")
                                        apply_dashboard_update(agent_name, "empty", time.time()-start_t, agent_api, agent_model)
                                except Exception as e:
                                    status_placeholder.error(f"Error: {e}")
                                    apply_dashboard_update(agent_name, "error", time.time()-start_t, agent_api, agent_model)
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.success("✅ Processing complete!")

    # --------- Tab 2: Prompt ID Orchestrator with Chaining ----------
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧩 Configure Agent Chain")
        
        col_config1, col_config2 = st.columns(2)
        with col_config1:
            num_agents = st.number_input(
                "Number of agents to chain", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.get('num_agents', 1), 
                step=1, 
                key="num_prompt_agents"
            )
            st.session_state.num_agents = num_agents
        
        with col_config2:
            enable_chaining = st.checkbox(
                "Enable chain execution (output → input)", 
                value=True,
                help="When enabled, output from Agent N becomes input to Agent N+1"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

        if 'prompt_agents' not in st.session_state or len(st.session_state.prompt_agents) != num_agents:
            st.session_state.prompt_agents = [
                {
                    "use_prompt_id": True,
                    "prompt_id": DEFAULT_PROMPT_ID if i == 0 else "",
                    "version": "1",
                    "provider": "OpenAI",
                    "model": "gpt-4o-mini",
                    "system_prompt": "You are a helpful AI assistant.",
                    "user_prompt": "",
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "top_p": 1.0
                } for i in range(num_agents)
            ]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Agent Definitions")
        for idx in range(num_agents):
            with st.expander(f"Agent {idx+1} Configuration", expanded=(idx == 0)):
                pa = st.session_state.prompt_agents[idx]
                
                # Chain indicator
                if enable_chaining and idx > 0:
                    st.markdown(f"""
                        <div class="chain-indicator">
                            🔗 <strong>Chained Input:</strong> This agent will receive output from Agent {idx}
                        </div>
                    """, unsafe_allow_html=True)
                
                pa["provider"] = st.selectbox(
                    "Provider", 
                    ["OpenAI", "Gemini", "Grok"], 
                    index=0, 
                    key=f"agent_provider_{idx}"
                )
                
                pa["model"] = st.selectbox(
                    "Model", 
                    MODEL_OPTIONS[pa["provider"]], 
                    key=f"agent_model_{idx}"
                )

                pa["use_prompt_id"] = st.radio(
                    "Prompt Type",
                    ["Stored Prompt ID", "Custom Prompt"],
                    index=0 if pa["use_prompt_id"] else 1,
                    horizontal=True,
                    key=f"use_prompt_id_{idx}"
                ) == "Stored Prompt ID"

                if pa["use_prompt_id"]:
                    pa["prompt_id"] = st.text_input(
                        "Prompt ID", 
                        value=pa["prompt_id"], 
                        placeholder="pmpt_...", 
                        key=f"prompt_id_{idx}"
                    )
                    pa["version"] = st.text_input(
                        "Version", 
                        value=pa.get("version","1"), 
                        key=f"prompt_ver_{idx}"
                    )
                    if idx == 0 or not enable_chaining:
                        pa["user_prompt"] = st.text_area(
                            "Initial User Input", 
                            value=pa["user_prompt"], 
                            key=f"user_prompt_pid_{idx}", 
                            height=120
                        )
                    else:
                        st.info("Input will come from previous agent's output")
                else:
                    pa["system_prompt"] = st.text_area(
                        "System Prompt", 
                        value=pa["system_prompt"], 
                        key=f"system_prompt_{idx}", 
                        height=120
                    )
                    if idx == 0 or not enable_chaining:
                        pa["user_prompt"] = st.text_area(
                            "User Prompt", 
                            value=pa["user_prompt"], 
                            key=f"user_prompt_{idx}", 
                            height=120
                        )
                    else:
                        st.info("Input will come from previous agent's output")

                colp1, colp2, colp3 = st.columns(3)
                with colp1:
                    pa["temperature"] = st.slider(
                        "Temperature", 
                        0.0, 2.0, 
                        float(pa.get("temperature", 0.7)), 
                        0.1, 
                        key=f"temp_{idx}"
                    )
                with colp2:
                    pa["top_p"] = st.slider(
                        "Top-p", 
                        0.0, 1.0, 
                        float(pa.get("top_p", 1.0)), 
                        0.05, 
                        key=f"topp_{idx}"
                    )
                with colp3:
                    pa["max_tokens"] = st.number_input(
                        "Max Tokens", 
                        64, 8192, 
                        int(pa.get("max_tokens", 2048)), 
                        32, 
                        key=f"max_toks_{idx}"
                    )

        st.markdown('</div>', unsafe_allow_html=True)

        # Initial input for chain
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 Initial Input (for first agent)")
        initial_input = st.text_area(
            "Enter the starting input for the agent chain",
            value=st.session_state.get("initial_chain_input", ""),
            height=150,
            key="initial_chain_input_widget"
        )
        st.session_state.initial_chain_input = initial_input
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("▶️ Execute Agent Chain", use_container_width=True, type="primary"):
            if not initial_input.strip() and st.session_state.prompt_agents[0].get("user_prompt", "").strip() == "":
                st.error("Please provide initial input for the first agent")
                st.stop()

            st.session_state.chain_outputs = {}
            current_input = initial_input if initial_input.strip() else st.session_state.prompt_agents[0].get("user_prompt", "")
            
            results_container = st.container()
            
            for idx, pa in enumerate(st.session_state.prompt_agents):
                with results_container:
                    st.markdown(f"""
                        <div class="card">
                            <h4>🤖 Agent {idx+1}: Execution</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show current input
                    if enable_chaining and idx > 0:
                        st.markdown(f"""
                            <div class="chain-indicator">
                                📥 <strong>Input from Agent {idx}:</strong>
                            </div>
                        """, unsafe_allow_html=True)
                        st.text_area(
                            f"Input to Agent {idx+1}", 
                            value=current_input, 
                            height=150, 
                            key=f"view_input_{idx}",
                            disabled=True
                        )
                    
                    # Allow editing before execution
                    if enable_chaining and idx > 0:
                        edit_input = st.checkbox(
                            f"Edit input before running Agent {idx+1}",
                            key=f"edit_input_{idx}"
                        )
                        if edit_input:
                            current_input = st.text_area(
                                f"Modified input for Agent {idx+1}",
                                value=current_input,
                                height=150,
                                key=f"edited_input_{idx}"
                            )
                    
                    status = st.empty()
                    output_container = st.empty()
                    start_t = time.time()
                    
                    try:
                        status.info(f"⏳ Running Agent {idx+1} with {pa['provider']}/{pa['model']}...")
                        
                        # Get appropriate client
                        if pa["provider"] == "OpenAI":
                            client = get_llm_client("OpenAI")
                            if not client:
                                raise ValueError("OpenAI client not available")
                            
                            params = {
                                "temperature": pa["temperature"],
                                "top_p": pa["top_p"],
                                "max_tokens": pa["max_tokens"]
                            }
                            
                            if pa["use_prompt_id"]:
                                if not pa["prompt_id"]:
                                    raise ValueError("Missing Prompt ID")
                                output_text = run_openai_prompt_id(
                                    client=client,
                                    model=pa["model"],
                                    prompt_id=pa["prompt_id"],
                                    version=pa.get("version", "1"),
                                    user_input=current_input,
                                    params=params
                                )
                            else:
                                output_text = run_openai_freeform(
                                    client=client,
                                    model=pa["model"],
                                    system_prompt=pa.get("system_prompt", ""),
                                    user_input=current_input,
                                    params=params
                                )
                        
                        elif pa["provider"] == "Gemini":
                            client = get_llm_client("Gemini")
                            if not client:
                                raise ValueError("Gemini client not available")
                            
                            model_instance = client.GenerativeModel(pa["model"])
                            
                            if pa["use_prompt_id"]:
                                prompt = current_input
                            else:
                                prompt = f"{pa.get('system_prompt', '')}\n\n{current_input}"
                            
                            response = model_instance.generate_content(
                                prompt,
                                generation_config={
                                    "temperature": pa["temperature"],
                                    "top_p": pa["top_p"],
                                    "max_output_tokens": pa["max_tokens"]
                                }
                            )
                            output_text = response.text
                        
                        elif pa["provider"] == "Grok":
                            client = get_llm_client("Grok")
                            if not client:
                                raise ValueError("Grok client not available")
                            
                            chat = client.chat.create(model=pa["model"])
                            if pa["use_prompt_id"]:
                                chat.append(grok_user(current_input))
                            else:
                                if pa.get("system_prompt"):
                                    chat.append(grok_system(pa["system_prompt"]))
                                chat.append(grok_user(current_input))
                            
                            response = chat.sample()
                            output_text = response.content
                        
                        else:
                            raise ValueError(f"Unknown provider: {pa['provider']}")
                        
                        # Store output
                        st.session_state.chain_outputs[idx] = output_text
                        
                        # Display output
                        status.success(f"✅ Agent {idx+1} completed successfully!")
                        
                        with output_container.container():
                            st.markdown("**Output:**")
                            st.markdown(output_text)
                            
                            col_dl, col_copy = st.columns([1, 3])
                            with col_dl:
                                st.download_button(
                                    f"💾 Download",
                                    output_text,
                                    file_name=f"agent_{idx+1}_output.txt",
                                    mime="text/plain",
                                    key=f"dl_output_{idx}"
                                )
                        
                        # Update dashboard
                        agent_label = f"Agent {idx+1}"
                        if pa["use_prompt_id"]:
                            agent_label += f" (Prompt ID)"
                        
                        apply_dashboard_update(
                            agent_label, 
                            "success", 
                            time.time() - start_t, 
                            pa["provider"], 
                            pa["model"],
                            prompt_id=pa.get("prompt_id") if pa["use_prompt_id"] else None
                        )
                        
                        # Set up for next agent in chain
                        if enable_chaining and idx < len(st.session_state.prompt_agents) - 1:
                            current_input = output_text
                            st.markdown("""
                                <div class="chain-indicator">
                                    🔗 Output will be passed to next agent
                                </div>
                            """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        status.error(f"❌ Agent {idx+1} failed: {str(e)}")
                        st.error(traceback.format_exc())
                        apply_dashboard_update(
                            f"Agent {idx+1}", 
                            "error", 
                            time.time() - start_t, 
                            pa["provider"], 
                            pa["model"]
                        )
                        if not enable_chaining:
                            continue
                        else:
                            st.warning("Chain execution stopped due to error")
                            break
            
            st.success("🎉 Agent chain execution completed!")

    # --------- Tab 3: Dashboard ----------
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Execution Analytics")
        history = st.session_state.get("run_history", [])
        
        if not history:
            st.info("No execution history yet. Run agents to populate the dashboard.")
        else:
            # KPIs
            total = len(history)
            successes = sum(1 for r in history if r["status"] == "success")
            errors = sum(1 for r in history if r["status"] == "error")
            avg_duration = sum(r["duration_s"] for r in history) / total if total > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Runs", total)
            with col2:
                st.metric("✅ Successes", successes, delta=f"{(successes/total*100):.1f}%")
            with col3:
                st.metric("❌ Errors", errors, delta=f"{(errors/total*100):.1f}%")
            with col4:
                st.metric("⏱️ Avg Duration", f"{avg_duration:.2f}s")
            
            st.markdown("---")
            
            # Detailed history table
            st.markdown("#### 📋 Execution History")
            
            # Add filters
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                filter_provider = st.multiselect(
                    "Filter by Provider",
                    options=list(set(r["provider"] for r in history)),
                    default=list(set(r["provider"] for r in history))
                )
            with col_f2:
                filter_status = st.multiselect(
                    "Filter by Status",
                    options=list(set(r["status"] for r in history)),
                    default=list(set(r["status"] for r in history))
                )
            with col_f3:
                filter_model = st.multiselect(
                    "Filter by Model",
                    options=list(set(r["model"] for r in history)),
                    default=list(set(r["model"] for r in history))
                )
            
            # Apply filters
            filtered_history = [
                r for r in history 
                if r["provider"] in filter_provider 
                and r["status"] in filter_status
                and r["model"] in filter_model
            ]
            
            st.dataframe(filtered_history, use_container_width=True, hide_index=True)
            
            # Visualizations
            st.markdown("#### 📊 Usage Analytics")
            
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                st.markdown("**Executions by Provider**")
                provider_counts = {}
                for r in filtered_history:
                    provider_counts[r["provider"]] = provider_counts.get(r["provider"], 0) + 1
                if provider_counts:
                    st.bar_chart(provider_counts)
            
            with col_v2:
                st.markdown("**Success Rate by Provider**")
                provider_success = {}
                provider_total = {}
                for r in filtered_history:
                    p = r["provider"]
                    provider_total[p] = provider_total.get(p, 0) + 1
                    if r["status"] == "success":
                        provider_success[p] = provider_success.get(p, 0) + 1
                
                success_rates = {
                    p: (provider_success.get(p, 0) / provider_total[p] * 100)
                    for p in provider_total
                }
                if success_rates:
                    st.bar_chart(success_rates)
            
            # Model usage
            st.markdown("**Model Usage Distribution**")
            model_counts = {}
            for r in filtered_history:
                key = f"{r['provider']} - {r['model']}"
                model_counts[key] = model_counts.get(key, 0) + 1
            if model_counts:
                st.bar_chart(model_counts)
            
            # Export options
            st.markdown("---")
            st.markdown("#### 💾 Export Data")
            
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                json_data = json.dumps(filtered_history, indent=2)
                st.download_button(
                    "📥 Download as JSON",
                    json_data,
                    file_name=f"execution_history_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col_e2:
                csv_data = "Time,Agent,Provider,Model,Status,Duration(s),Tokens,PromptID\n"
                for r in filtered_history:
                    csv_data += f"{r['time']},{r['agent']},{r['provider']},{r['model']},{r['status']},{r['duration_s']},{r.get('tokens', 'N/A')},{r.get('prompt_id', 'N/A')}\n"
                st.download_button(
                    "📥 Download as CSV",
                    csv_data,
                    file_name=f"execution_history_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear history option
        if st.button("🗑️ Clear Execution History", type="secondary"):
            st.session_state.run_history = []
            st.rerun()

    # Footer status
    st.markdown("---")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✅ System Status")
    status_cols = st.columns(5)
    with status_cols[0]:
        st.info("🟢 System Ready")
    with status_cols[1]:
        openai_status = "🟢 Connected" if resolve_key("OpenAI") else "🔴 No Key"
        st.info(f"OpenAI: {openai_status}")
    with status_cols[2]:
        gemini_status = "🟢 Connected" if resolve_key("Gemini") else "🔴 No Key"
        st.info(f"Gemini: {gemini_status}")
    with status_cols[3]:
        grok_status = "🟢 Connected" if resolve_key("Grok") else "🔴 No Key"
        st.info(f"Grok: {grok_status}")
    with status_cols[4]:
        chain_status = "🟢 Active" if len(st.session_state.get("chain_outputs", {})) > 0 else "⚪ Idle"
        st.info(f"Chain: {chain_status}")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
