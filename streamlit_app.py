import streamlit as st
import pandas as pd
import os
from datetime import datetime
import io
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Set page config
st.set_page_config(
    page_title="Welfare Secretary AI",
    page_icon="🏛️",
    layout="wide"
)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Custom CSS
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

# Initialize AI components
@st.cache_resource
def initialize_ai():
    if not api_key:
        return None, None, None
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.7,
        google_api_key=api_key
    )
    
    # Load Excel data
    excel_path = "SECRETARY FORM(1-6).xlsx"
    if not os.path.exists(excel_path):
        return None, None, None
    
    try:
        df = pd.read_excel(excel_path)
        
        # Prepare document content
        def prepare_document_content(row):
            content_parts = []
            
            if pd.notna(row.get('Name')):
                content_parts.append(f"Name: {row['Name']}")
            if pd.notna(row.get('Category')):
                content_parts.append(f"Category: {row['Category']}")
            if pd.notna(row.get('Total Collected (Amount)')):
                content_parts.append(f"Total Collected: {row['Total Collected (Amount)']}")
            if pd.notna(row.get('Total Spent (Amount)')):
                content_parts.append(f"Total Spent: {row['Total Spent (Amount)']}")
            if pd.notna(row.get('Total Remaining (Amount)')):
                content_parts.append(f"Total Remaining: {row['Total Remaining (Amount)']}")
            if pd.notna(row.get('Event Name')):
                content_parts.append(f"Event: {row['Event Name']}")
            if pd.notna(row.get('Location')):
                content_parts.append(f"Location: {row['Location']}")
            if pd.notna(row.get('Attendance')):
                content_parts.append(f"Attendance: {row['Attendance']}")
            if pd.notna(row.get('Key Outcomes')):
                content_parts.append(f"Key Outcomes: {row['Key Outcomes']}")
            if pd.notna(row.get('Agenda')):
                content_parts.append(f"Agenda: {row['Agenda']}")
            if pd.notna(row.get('Decisions Made')):
                content_parts.append(f"Decisions Made: {row['Decisions Made']}")
            if pd.notna(row.get('Issue Title')):
                content_parts.append(f"Issue: {row['Issue Title']}")
            if pd.notna(row.get('Description')):
                content_parts.append(f"Description: {row['Description']}")
            if pd.notna(row.get('Comments')):
                content_parts.append(f"Comments: {row['Comments']}")
            
            return " | ".join(content_parts) if content_parts else f"Record ID: {row.get('ID', 'Unknown')}"
        
        df['document_content'] = df.apply(prepare_document_content, axis=1)
        
        # Create embeddings and vector store
        loader = DataFrameLoader(df, page_content_column='document_content')
        documents = loader.load()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()
        
        # Create prompt templates
        report_template = """
        You are the Welfare Committee Secretary.
        Using ONLY the information provided, write a comprehensive and expressive report
        in the first person, covering activities, meetings, events, and finances.
        Make it flow naturally and sound human—formal but simple, intelligent, and warm.
        Avoid robotic or AI-sounding language.

        User request: {question}

        Relevant data:
        {context}
        """
        
        normal_template = """
        Answer this question directly and briefly using only the Welfare Committee data.

        User request: {question}

        Relevant data:
        {context}
        """
        
        report_prompt = PromptTemplate(template=report_template, input_variables=["context", "question"])
        normal_prompt = PromptTemplate(template=normal_template, input_variables=["context", "question"])
        
        # Create QA chains
        report_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": report_prompt}
        )
        
        normal_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": normal_prompt}
        )
        
        return report_chain, normal_chain, df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def generate_response(query):
    report_chain, normal_chain, df = initialize_ai()
    
    if report_chain is None:
        return "Please ensure GOOGLE_API_KEY is set and SECRETARY FORM(1-6).xlsx file is available."
    
    if "report" in query.lower():
        chain_to_use = report_chain
    else:
        chain_to_use = normal_chain
    
    try:
        response = chain_to_use.run(query)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

def create_pdf_download(content):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor='#1a365d',
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=14,
        alignment=TA_JUSTIFY,
        leftIndent=0,
        rightIndent=0,
        leading=16
    )
    
    story = []
    story.append(Paragraph("Welfare Committee Report", title_style))
    story.append(Spacer(1, 30))
    
    clean_content = content.replace("I am", "The committee").replace("I have", "The committee has")
    clean_content = clean_content.replace("my role", "the secretary's role")
    
    paragraphs = clean_content.split('\n\n')
    for para in paragraphs:
        if para.strip():
            clean_para = para.strip().replace('\n', ' ')
            story.append(Paragraph(clean_para, body_style))
            story.append(Spacer(1, 8))
    
    story.append(Spacer(1, 40))
    current_date = datetime.now().strftime("%B %d, %Y")
    story.append(Paragraph(f"Report Date: {current_date}", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def create_docx_download(content):
    buffer = io.BytesIO()
    doc = Document()
    
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
    
    title = doc.add_heading('Welfare Committee Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    
    clean_content = content.replace("I am", "The committee").replace("I have", "The committee has")
    clean_content = clean_content.replace("my role", "the secretary's role")
    
    paragraphs = clean_content.split('\n\n')
    for para in paragraphs:
        if para.strip():
            clean_para = para.strip().replace('\n', ' ')
            p = doc.add_paragraph(clean_para)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    doc.add_paragraph()
    doc.add_paragraph()
    current_date = datetime.now().strftime("%B %d, %Y")
    footer_para = doc.add_paragraph(f"Report Date: {current_date}")
    footer_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

# Header
st.markdown('<div class="main-header"><h1>🏛️ Welfare Secretary AI</h1><p>Your AI assistant for welfare committee management</p></div>', unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# File upload section
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
if uploaded_file is not None:
    # Save uploaded file
    with open("SECRETARY FORM(1-6).xlsx", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("File uploaded successfully!")
    # Clear cache to reload data
    st.cache_resource.clear()

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
                pdf_data = create_pdf_download(message["content"])
                st.download_button(
                    label="📄 PDF",
                    data=pdf_data,
                    file_name=f"welfare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            with col2:
                docx_data = create_docx_download(message["content"])
                st.download_button(
                    label="📝 Word",
                    data=docx_data,
                    file_name=f"welfare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

# Chat input
if prompt := st.chat_input("Ask about welfare committee activities..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get AI response
    with st.spinner("AI is thinking..."):
        response = generate_response(prompt)
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

# Status information
st.sidebar.header("📊 Status")
report_chain, normal_chain, df = initialize_ai()
if df is not None:
    st.sidebar.success(f"✅ Data loaded: {len(df)} records")
else:
    st.sidebar.warning("⚠️ No data loaded. Please upload Excel file and set GOOGLE_API_KEY.")