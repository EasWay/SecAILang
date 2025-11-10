import streamlit as st
import requests
import json
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Welfare Secretary AI",
    page_icon="🏛️",
    layout="wide"
)

# Custom CSS for mobile-friendly design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #4285f4, #34a853, #fbbc04, #ea4335);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #4285f4;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #1976d2;
    }
    .ai-message {
        background-color: #f1f8e9;
        border-left-color: #388e3c;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🏛️ Welfare Secretary AI</h1><p>Your AI assistant for welfare committee management</p></div>', unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message ai-message"><strong>AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            
            # Add download buttons for AI responses
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("📄 PDF", key=f"pdf_{len(st.session_state.messages)}"):
                    # Create PDF download
                    pdf_data = create_pdf_download(message["content"])
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name=f"welfare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
            with col2:
                if st.button("📝 Word", key=f"word_{len(st.session_state.messages)}"):
                    # Create Word download
                    docx_data = create_docx_download(message["content"])
                    st.download_button(
                        label="Download Word",
                        data=docx_data,
                        file_name=f"welfare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

# Chat input
if prompt := st.chat_input("Ask about welfare committee activities..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get AI response (you'll need to integrate your AI logic here)
    with st.spinner("AI is thinking..."):
        # For now, using a placeholder - integrate your actual AI logic
        response = get_ai_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()

# Suggestion buttons
st.markdown("### Quick Actions")
col1, col2 = st.columns(2)
with col1:
    if st.button("📊 Generate Welfare Report"):
        st.session_state.messages.append({"role": "user", "content": "Generate a comprehensive welfare report"})
        st.rerun()
    if st.button("💰 Financial Summary"):
        st.session_state.messages.append({"role": "user", "content": "What are the total finances collected?"})
        st.rerun()

with col2:
    if st.button("📅 Recent Events"):
        st.session_state.messages.append({"role": "user", "content": "Tell me about recent events"})
        st.rerun()
    if st.button("🤝 Meeting Summaries"):
        st.session_state.messages.append({"role": "user", "content": "What meetings have been held?"})
        st.rerun()

def get_ai_response(query):
    # Integrate your existing AI logic here
    # For now, returning a placeholder
    return "This is where your AI response would go. Integrate your existing generate_response() function here."

def create_pdf_download(content):
    # Integrate your PDF creation logic
    return b"PDF content would go here"

def create_docx_download(content):
    # Integrate your Word doc creation logic
    return b"DOCX content would go here"