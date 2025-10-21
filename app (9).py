import streamlit as st
import google.generativeai as genai
import openai
from xai_sdk import Client as GrokClient
from xai_sdk.chat import user as grok_user
import os
import io
import yaml
import traceback
from PyPDF2 import PdfReader, PdfWriter
import pytesseract
from pdf2image import convert_from_bytes
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="🤖 Agentic PDF Processing System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme Configurations ---
THEMES = {
    "Blue Sky": {
        "primary": "#87CEEB",
        "secondary": "#4682B4",
        "background": "#F0F8FF",
        "text": "#1E3A5F",
        "accent": "#FFD700"
    },
    "Snow White": {
        "primary": "#FFFFFF",
        "secondary": "#E8E8E8",
        "background": "#F5F5F5",
        "text": "#2C3E50",
        "accent": "#3498DB"
    },
    "Deep Ocean": {
        "primary": "#006994",
        "secondary": "#003049",
        "background": "#001219",
        "text": "#E0FBFC",
        "accent": "#F77F00"
    },
    "Sparkling Galaxy": {
        "primary": "#8B5CF6",
        "secondary": "#EC4899",
        "background": "#1E1B4B",
        "text": "#F3E8FF",
        "accent": "#FCD34D"
    },
    "Alp Forest": {
        "primary": "#2D5016",
        "secondary": "#4A7C59",
        "background": "#E8F5E9",
        "text": "#1B5E20",
        "accent": "#FF6F00"
    },
    "Flora": {
        "primary": "#E91E63",
        "secondary": "#9C27B0",
        "background": "#FCE4EC",
        "text": "#880E4F",
        "accent": "#00BCD4"
    },
    "Ferrari": {
        "primary": "#DC0000",
        "secondary": "#8B0000",
        "background": "#FFF5F5",
        "text": "#1A0000",
        "accent": "#FFD700"
    },
    "Fendi Casa": {
        "primary": "#D4AF37",
        "secondary": "#8B7355",
        "background": "#F5F5DC",
        "text": "#3E2723",
        "accent": "#C9A961"
    }
}

# --- Model Definitions ---
MODEL_OPTIONS = {
    "Gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
    "OpenAI": ["gpt-5-nano", "gpt-4.1-mini", "gpt-4o-mini"],
    "Grok": ["grok-4-fast-reasoning", "grok-3-mini"]
}

# --- Custom CSS Function ---
def apply_theme(theme_name):
    theme = THEMES[theme_name]
    css = f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, {theme['background']} 0%, {theme['secondary']}15 100%);
        }}
        .main-header {{
            background: linear-gradient(90deg, {theme['primary']}, {theme['secondary']});
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            margin-bottom: 2rem;
            animation: slideIn 0.5s ease-out;
        }}
        .card {{
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin: 1rem 0;
            border-left: 4px solid {theme['accent']};
            transition: transform 0.3s ease;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }}
        .metric-card {{
            background: linear-gradient(135deg, {theme['primary']}20, {theme['secondary']}20);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            border: 2px solid {theme['accent']};
        }}
        .success-message {{
            background: #D4EDDA;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #28A745;
        }}
        .error-message {{
            background: #F8D7DA;
            color: #721C24;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #DC3545;
        }}
        h1, h2, h3 {{
            color: {theme['text']};
        }}
        .stButton>button {{
            background: linear-gradient(90deg, {theme['primary']}, {theme['secondary']});
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 25px;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}
        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(-20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        .highlight {{
            color: {theme['accent']};
            font-weight: bold;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Helper Functions ---
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
            text = text.replace(keyword, f"<span style='color:coral;font-weight:bold'>{keyword}</span>")
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
                st.error("Google Gemini API key not found. Please set GEMINI_API_KEY in Secrets.")
                return None
            genai.configure(api_key=api_key)
            return genai
        elif api_choice == "OpenAI":
            api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OpenAI API key not found. Please set OPENAI_API_KEY in Secrets.")
                return None
            return openai.OpenAI(api_key=api_key)
        elif api_choice == "Grok":
            api_key = st.secrets.get("GROK_API_KEY") or os.getenv("XAI_API_KEY")
            if not api_key:
                st.error("Grok API key not found. Please set GROK_API_KEY or XAI_API_KEY in Secrets.")
                return None
            return GrokClient(api_key=api_key, timeout=3600)
    except Exception as e:
        st.error(f"Error initializing {api_choice} client: {e}")
        return None

def execute_agent(agent_config, input_text):
    """Executes a single agent with the given configuration and input."""
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
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    **agent_config.get('parameters', {})
                )
                return response.choices[0].message.content
            elif agent_config['api'] == "Grok":
                chat = client.chat.create(model=model)
                chat.append(grok_user(prompt))
                response = chat.sample()
                return response.content
    except Exception as e:
        st.error(f"Error executing agent '{agent_config['name']}': {e}")
        traceback.print_exc()
        return None

# --- Main Application ---
def main():
    # Sidebar - Theme Selection
    with st.sidebar:
        st.markdown("### 🎨 Choose Your Theme")
        selected_theme = st.selectbox(
            "Select Theme",
            list(THEMES.keys()),
            index=0
        )
        apply_theme(selected_theme)
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        
        # API Selection
        api_choice = st.selectbox("Select API Provider", list(MODEL_OPTIONS.keys()))
        model_choice = st.selectbox("Select Model", MODEL_OPTIONS[api_choice])
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("This intelligent system processes PDFs using multiple AI agents with OCR capabilities.")
    
    # Apply theme
    apply_theme(selected_theme)
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🤖 Agentic PDF Processing System</h1>
            <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                Transform your documents with AI-powered intelligence
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main Content
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
        st.metric("📄 Files Processed", "0", "+0")
        st.metric("🤖 Agents Active", "0", "Ready")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        file_bytes = uploaded_file.read()
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔧 Processing Options")
        
        col_a, col_b = st.columns(2)
        with col_a:
            reader = PdfReader(io.BytesIO(file_bytes))
            total_pages = len(reader.pages)
            st.info(f"Total pages: **{total_pages}**")
            
            use_page_range = st.checkbox("Select specific page range")
            if use_page_range:
                start_page = st.number_input("Start page", 1, total_pages, 1)
                end_page = st.number_input("End page", 1, total_pages, total_pages)
                pages_to_trim = (int(start_page), int(end_page))
            else:
                pages_to_trim = (1, total_pages)
        
        with col_b:
            use_ocr = st.checkbox("Use OCR for scanned PDFs", help="Enable if your PDF contains images of text")
            highlight_keywords = st.text_input("Keywords to highlight (comma-separated)", placeholder="AI, machine learning, data")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Process PDF with AI", use_container_width=True):
            with st.spinner("Processing your document..."):
                # Trim PDF if needed
                if use_page_range:
                    file_bytes = trim_pdf(file_bytes, pages_to_trim)
                    if file_bytes is None:
                        st.stop()
                
                # Extract or OCR text
                if use_ocr:
                    st.info("🔍 Performing OCR on images...")
                    extracted_text = ocr_pdf(file_bytes)
                else:
                    st.info("📝 Extracting text from PDF...")
                    extracted_text = extract_text_from_pdf(file_bytes)
                
                if not extracted_text or not extracted_text.strip():
                    st.error("❌ No text could be extracted. Try enabling OCR if this is a scanned document.")
                    st.stop()
                
                # Display extracted text
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 📄 Extracted Text Preview")
                preview_text = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
                if highlight_keywords:
                    preview_text = to_markdown_with_keywords(preview_text, highlight_keywords)
                st.markdown(preview_text, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Load and execute agents
                agents_config = load_agents_config()
                if agents_config and 'agents' in agents_config:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### 🤖 AI Agent Processing")
                    
                    for agent in agents_config['agents']:
                        # Override with user selections
                        agent['api'] = api_choice
                        agent['model'] = model_choice
                        
                        with st.expander(f"🔹 {agent['name']}", expanded=True):
                            result = execute_agent(agent, extracted_text)
                            if result:
                                st.markdown(f"**Result:**")
                                st.markdown(result)
                                st.download_button(
                                    f"💾 Download {agent['name']} Result",
                                    result,
                                    file_name=f"{agent['name'].replace(' ', '_')}_result.txt",
                                    mime="text/plain"
                                )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.success("✅ Processing complete!")
    
    # Follow-up Questions Section
    st.markdown("---")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💡 Follow-up Questions")
    
    questions = [
        "Would you like to add **batch processing** for multiple PDFs simultaneously?",
        "Should I implement **AI-powered summarization** with adjustable length controls?",
        "Would you like to add **export options** (Markdown, JSON, DOCX) for processed results?",
        "Should I create **custom agent workflows** where you can chain multiple agents together?",
        "Would you like **real-time collaboration features** where multiple users can process documents together?"
    ]
    
    for i, question in enumerate(questions, 1):
        st.markdown(f"{i}. {question}")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
