import streamlit as st
import yaml
from PyPDF2 import PdfReader, PdfWriter
from PIL import Image
import pytesseract
import io
import google.generativeai as genai
import openai
import os
from dotenv import load_dotenv
import traceback

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic PDF Processing System",
    page_icon="🤖",
    layout="wide",
)

# Load environment variables for local development
load_dotenv()

# --- Model Definitions ---
MODEL_OPTIONS = {
    "Gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite","gemini-2.0-flash", "gemini-2.0-flash-lite"],
    "OpenAI": ["gpt-5-nano", "gpt-4o-mini"],
    "Grok": ["grok-4-fast-reasoning", "grok-3-mini"] # Models available via xAI SDK
}

# --- Function Definitions ---

def trim_pdf(file_bytes, pages_to_trim):
    """Trims a PDF file based on the specified page range."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        writer = PdfWriter()
        start_page, end_page = pages_to_trim
        if start_page > end_page or start_page < 1 or end_page > len(reader.pages):
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
    """Performs OCR on an image-based PDF."""
    try:
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(file_bytes)
        full_text = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            full_text += f"\n--- Page {i+1} ---\n{text}"
        return full_text
    except ImportError:
        st.error("The 'pdf2image' library is not installed. This is required for OCR.")
        return None
    except Exception as e:
        st.warning(f"Could not perform OCR. Ensure 'poppler' and 'tesseract' are installed. Error: {e}")
        return None

def extract_text_from_pdf(file_bytes):
    """Extracts text from a text-based PDF."""
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
    """Converts text to Markdown and highlights keywords."""
    if keywords:
        keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
        for keyword in keyword_list:
            text = text.replace(keyword, f"<span style='color:coral;'>{keyword}</span>")
    return text

@st.cache_data
def load_agents_config():
    """Loads the agent configurations from agents.yaml."""
    try:
        with open("agents.yaml", 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("agents.yaml not found. Please create it.")
        return {}

def get_llm_client(api_choice):
    """Initializes and returns the appropriate LLM client."""
    try:
        if api_choice == "Gemini":
            api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                st.error("Google Gemini API key is not set in Streamlit secrets.")
                return None
            genai.configure(api_key=api_key)
            return genai
        elif api_choice == "OpenAI":
            api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OpenAI API key is not set in Streamlit secrets.")
                return None
            return openai.OpenAI(api_key=api_key)
        elif api_choice == "Grok":
            # *** BUG FIX & IMPROVEMENT: Using native xai-sdk ***
            try:
                from xai_sdk import Client as GrokClient
            except ImportError:
                st.error("The 'xai-sdk' is required for Grok. Please add it to requirements.txt.")
                return None
                
            api_key = st.secrets.get("GROK_API_KEY") or os.getenv("XAI_API_KEY")
            if not api_key:
                st.error("XAI_API_KEY (for Grok) is not set in Streamlit secrets.")
                return None
            # Use a long timeout as recommended for reasoning models
            return GrokClient(api_key=api_key, timeout=3600)
    except Exception as e:
        st.error(f"Error initializing {api_choice} client: {e}")
        return None
    return None

def execute_agent(agent_config, input_text):
    """Executes a single agent with the given configuration and input."""
    client = get_llm_client(agent_config['api'])
    if not client:
        return f"Could not initialize the {agent_config['api']} client. Check API keys."

    prompt = agent_config['prompt'].format(input_text=input_text)
    model = agent_config['model']
    
    try:
        if agent_config['api'] == "Gemini":
            model_instance = client.GenerativeModel(model)
            response = model_instance.generate_content(prompt)
            return response.text
        elif agent_config['api'] == "OpenAI":
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **agent_config.get('parameters', {})
            )
            return response.choices[0].message.content
        elif agent_config['api'] == "Grok":
            # *** BUG FIX & IMPROVEMENT: Using native xai-sdk workflow ***
            try:
                from xai_sdk.chat import user
            except ImportError:
                return "Could not import 'user' from 'xai_sdk.chat'. Is xai-sdk installed?"
            
            # For our stateless agent system, we create a new chat for each execution.
            chat = client.chat.create(model=model)
            chat.append(user(prompt))
            response = chat.sample()
            return response.content
            
    except Exception as e:
        st.error(f"An error occurred executing agent '{agent_config['name']}': {e}")
        traceback.print_exc()
        return None

# The rest of the app.py file (UI and main logic) remains unchanged.
# It is omitted here for brevity but should be included from the previous correct version.
def load_css(file_name):
    """Loads a CSS file into the Streamlit app."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found.")

# --- Main Application ---
def main():
    st.title("📄✨ Agentic PDF Processing & Analysis System")

    # --- Initialize Session State ---
    if 'processed_texts' not in st.session_state:
        st.session_state.processed_texts = {}
    if 'agent_configs' not in st.session_state:
        st.session_state.agent_configs = {}
    if 'agent_outputs' not in st.session_state:
        st.session_state.agent_outputs = []


    # --- Sidebar for Theme Selection ---
    with st.sidebar:
        st.header("🎨 Theme Selector")
        themes = ["Neat dark", "Simple white", "Alp. forest", "Blue sky", "Deep ocean", "Magic purple", "Beethoven", "Mozart", "J.S.Bach", "Chopin", "Ferrari Sportscar", "NBA", "MLB", "NFL"]
        selected_theme = st.selectbox("Choose a UI theme", themes)
        
        # Sanitize theme name for CSS class
        theme_class = selected_theme.lower().replace(" ", "-").replace(".", "")
        st.markdown(f'<body class="{theme_class}"></body>', unsafe_allow_html=True)


    # --- File Upload and Processing ---
    st.header("1. PDF Upload and Pre-processing")
    uploaded_files = st.file_uploader("📂 Upload your PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.expander(f"Process '{uploaded_file.name}'"):
                file_bytes = uploaded_file.getvalue()
                
                try:
                    reader = PdfReader(io.BytesIO(file_bytes))
                    total_pages = len(reader.pages)
                    pages_to_trim = st.slider(f"Pages to trim for '{uploaded_file.name}'", 1, total_pages, (1, total_pages), key=f"trim_{uploaded_file.name}")
                except Exception:
                    st.error(f"Could not read '{uploaded_file.name}'. The file may be corrupted.")
                    continue

                keywords = st.text_input(f"Keywords to highlight for '{uploaded_file.name}' (comma-separated)", key=f"kw_{uploaded_file.name}")

                if st.button(f"Process '{uploaded_file.name}'", key=f"proc_{uploaded_file.name}"):
                    trimmed_pdf_bytes = trim_pdf(file_bytes, pages_to_trim)
                    if trimmed_pdf_bytes:
                        text = extract_text_from_pdf(trimmed_pdf_bytes)
                        if len(text.strip()) < 100 * (pages_to_trim[1] - pages_to_trim[0] + 1):
                            st.info("Low text content detected, attempting OCR...")
                            ocr_text = ocr_pdf(trimmed_pdf_bytes)
                            if ocr_text:
                                text = ocr_text
                        
                        markdown_text = to_markdown_with_keywords(text, keywords)
                        st.session_state.processed_texts[uploaded_file.name] = markdown_text
                        st.success(f"'{uploaded_file.name}' processed successfully!")

            if uploaded_file.name in st.session_state.processed_texts:
                st.markdown(f"### Processed & Editable Text from '{uploaded_file.name}'")
                edited_text = st.text_area(f"Edit Markdown for '{uploaded_file.name}'", 
                                           st.session_state.processed_texts[uploaded_file.name], 
                                           height=300, key=f"edit_{uploaded_file.name}")
                st.session_state.processed_texts[uploaded_file.name] = edited_text


    # --- Agent Configuration and Execution ---
    if st.session_state.processed_texts:
        st.header("2. Agentic Workflow")
        
        initial_input_text = "\n\n---\n\n".join(st.session_state.processed_texts.values())

        agents_config_yaml = load_agents_config()
        if not agents_config_yaml or 'agents' not in agents_config_yaml:
            st.error("Agent configuration is missing or invalid in agents.yaml.")
            return
            
        num_agents = st.slider("Number of agents to use", 1, len(agents_config_yaml.get('agents', [])), 1)
        
        # Ensure agent_outputs list is the correct size
        if len(st.session_state.agent_outputs) != num_agents:
            st.session_state.agent_outputs = [None] * num_agents
        
        # Use a separate variable for the input to the next agent
        next_input = initial_input_text
        
        for i in range(num_agents):
            st.subheader(f"Agent {i+1}")
            
            # Allow modification of previous agent's output before it becomes input
            if i > 0 and st.session_state.agent_outputs[i-1]:
                st.markdown(f"**Input for Agent {i+1} (Editable Output from Agent {i})**")
                edited_input = st.text_area(f"Edit input for Agent {i+1}", st.session_state.agent_outputs[i-1], height=200, key=f"input_edit_{i}")
                next_input = edited_input
            elif i == 0:
                next_input = initial_input_text

            agent_options = [agent['name'] for agent in agents_config_yaml['agents']]
            selected_agent_name = st.selectbox(f"Select Agent {i+1}", agent_options, key=f"agent_select_{i}")
            
            # Initialize the agent's config in session state if it doesn't exist
            if f'agent_{i}_config' not in st.session_state or st.session_state[f'agent_{i}_config']['name'] != selected_agent_name:
                default_config = next((agent for agent in agents_config_yaml['agents'] if agent['name'] == selected_agent_name), None)
                st.session_state[f'agent_{i}_config'] = default_config.copy()

            current_config = st.session_state[f'agent_{i}_config']
            
            with st.container(border=True):
                current_config['prompt'] = st.text_area(f"Prompt for Agent {i+1}", current_config['prompt'], height=150, key=f"prompt_{i}")
                
                col1, col2 = st.columns(2)
                with col1:
                    api_choice = st.selectbox("API", list(MODEL_OPTIONS.keys()), key=f"api_{i}", index=list(MODEL_OPTIONS.keys()).index(current_config['api']))
                    current_config['api'] = api_choice
                with col2:
                    model_choice = st.selectbox("Model", MODEL_OPTIONS[api_choice], key=f"model_{i}", index=MODEL_OPTIONS[api_choice].index(current_config['model']) if current_config['model'] in MODEL_OPTIONS[api_choice] else 0)
                    current_config['model'] = model_choice

                with st.expander("Advanced Parameters"):
                     params_str = st.text_area(f"Parameters (YAML format)", yaml.dump(current_config.get('parameters', {})), key=f"params_{i}")
                     try:
                        current_config['parameters'] = yaml.safe_load(params_str)
                     except yaml.YAMLError as e:
                        st.error(f"Invalid YAML in parameters: {e}")

            if st.button(f"Execute Agent {i+1}", key=f"exec_{i}"):
                with st.spinner(f"Agent {i+1} ({current_config['name']}) is processing..."):
                    output = execute_agent(current_config, next_input)
                    st.session_state.agent_outputs[i] = output if output is not None else "Execution failed."
            
            if st.session_state.agent_outputs[i]:
                st.markdown(f"**Output from Agent {i+1}**")
                st.text_area("Result", st.session_state.agent_outputs[i], height=300, key=f"output_display_{i}", disabled=True)

if __name__ == "__main__":
    main()