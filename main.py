import os
import pandas as pd
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- Initialization ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.5,  # Balanced for natural, flowing language while staying factual
    google_api_key=api_key,
    max_output_tokens=4096  # Allow very long, comprehensive reports with all details
)

# Load Excel data
excel_path = "SECRETARY FORM(1-6).xlsx"
df = pd.read_excel(excel_path)

# Clean and prepare data for document loading
def prepare_document_content(row):
    """Convert DataFrame row to meaningful text content"""
    content_parts = []
    
    # Add basic info
    if pd.notna(row.get('Name')):
        content_parts.append(f"Name: {row['Name']}")
    if pd.notna(row.get('Category')):
        content_parts.append(f"Category: {row['Category']}")
    
    # Add financial information
    if pd.notna(row.get('Total Collected (Amount)')):
        content_parts.append(f"Total Collected: {row['Total Collected (Amount)']}")
    if pd.notna(row.get('Total Spent (Amount)')):
        content_parts.append(f"Total Spent: {row['Total Spent (Amount)']}")
    if pd.notna(row.get('Total Remaining (Amount)')):
        content_parts.append(f"Total Remaining: {row['Total Remaining (Amount)']}")
    
    # Add event information
    if pd.notna(row.get('Event Name')):
        content_parts.append(f"Event: {row['Event Name']}")
    if pd.notna(row.get('Location')):
        content_parts.append(f"Location: {row['Location']}")
    if pd.notna(row.get('Attendance')):
        content_parts.append(f"Attendance: {row['Attendance']}")
    if pd.notna(row.get('Key Outcomes')):
        content_parts.append(f"Key Outcomes: {row['Key Outcomes']}")
    
    # Add meeting information
    if pd.notna(row.get('Agenda')):
        content_parts.append(f"Agenda: {row['Agenda']}")
    if pd.notna(row.get('Decisions Made')):
        content_parts.append(f"Decisions Made: {row['Decisions Made']}")
    
    # Add issues and comments
    if pd.notna(row.get('Issue Title')):
        content_parts.append(f"Issue: {row['Issue Title']}")
    if pd.notna(row.get('Description')):
        content_parts.append(f"Description: {row['Description']}")
    if pd.notna(row.get('Comments')):
        content_parts.append(f"Comments: {row['Comments']}")
    
    # Join all parts or return a default message
    return " | ".join(content_parts) if content_parts else f"Record ID: {row.get('ID', 'Unknown')}"

# Create a new column with combined content
df['document_content'] = df.apply(prepare_document_content, axis=1)

# Convert Excel to documents using the prepared content
loader = DataFrameLoader(df, page_content_column='document_content')
documents = loader.load()

# Create embeddings and store (using free local embeddings)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# --- Prompt Templates ---
report_template = """
You are the Welfare Committee Secretary writing a warm, human, and inclusive report.

CRITICAL INSTRUCTIONS:
1. Use ONLY the data provided in the context below - DO NOT use general knowledge or make assumptions
2. Write in first person as the secretary, using a warm and inclusive tone
3. Write in flowing paragraphs and essay form - NO bullet points or lists
4. Make it read like a story that connects all the welfare activities together
5. Include EVERY SINGLE item from the data - do not skip anything
6. Mention ALL specific details: names, amounts, dates, locations, attendance, outcomes, descriptions, comments
7. If there are general updates, announcements, or miscellaneous items, include them all

STRUCTURE YOUR REPORT AS FLOWING PARAGRAPHS (INCLUDE EVERYTHING):

Opening: Start with a warm introduction about the welfare committee's mission and period covered.

Financial Overview: Write a detailed narrative paragraph about EVERY financial transaction. Mention each collection source, every expenditure item, all amounts, and what remains. Include who contributed, what was purchased, and why. Make it conversational but include ALL the numbers and details.

Events and Activities: Write in essay form about EVERY SINGLE event in the data. For each event, describe:
- The event name and purpose
- When and where it was held
- Who attended (mention attendance numbers and any specific names)
- What happened during the event
- Key outcomes and achievements
- Any comments or feedback
Do not skip any event, no matter how small.

Meetings and Decisions: Write flowing paragraphs about EVERY meeting held. For each meeting:
- Discuss the full agenda
- Weave in all decisions made
- Mention who attended
- Include any action items or follow-ups
- Explain how these decisions impact member welfare
Make it read like you're telling a colleague about important discussions.

General Updates and Announcements: If there are any general updates, announcements, news, or miscellaneous items in the data, discuss them all in a warm, informative way. Don't leave anything out.

Issues, Concerns, and Progress: If there are ANY issues, concerns, problems, or ongoing matters in the data, discuss each one:
- Describe the issue clearly
- Mention any progress or status
- Show empathy and commitment to resolution
- Include any comments or descriptions

Other Items: If there are ANY other categories or items in the data not covered above, include them all. Check for columns like: updates, notes, remarks, status, progress, or any other information.

Closing: End with a warm, forward-looking statement about the committee's continued commitment to member welfare.

TONE: Warm, professional, inclusive, human, and caring. Avoid corporate jargon. Write as if you're sharing good news with friends while maintaining professionalism.

REMEMBER: Include EVERY SINGLE item from the data. Do not summarize or skip anything. If it's in the data, it must be in your report.

User request: {question}

Welfare Committee Data:
{context}

Write your report in flowing essay form with connected paragraphs. NO bullet points. Include EVERYTHING from the data.
"""

normal_template = """
You are the Welfare Committee Secretary answering a specific question in a warm, human way.

CRITICAL INSTRUCTIONS:
1. Use ONLY the data provided in the context below
2. Write in a conversational, friendly tone while being professional
3. Include specific details from the data: names, amounts, dates, locations
4. Write in flowing sentences, not bullet points
5. If the data doesn't contain the answer, say "I don't have that information in our current records"
6. Make your response feel personal and caring, as befits a welfare committee

User request: {question}

Welfare Committee Data:
{context}

Answer warmly and conversationally using only the data above:
"""

report_prompt = PromptTemplate(template=report_template, input_variables=["context", "question"])
normal_prompt = PromptTemplate(template=normal_template, input_variables=["context", "question"])

# --- Create QA Chains ---
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


def generate_response(query):
    if "report" in query.lower():
        chain_to_use = report_chain
    else:
        chain_to_use = normal_chain

    try:
        response = chain_to_use.run(query)
        return response
    except Exception as e:
        print(f"Error during chain invocation: {e}")
        return "An error occurred while generating the response."


print("Welfare Secretary AI (Classic) is ready. Type 'exit' or 'quit' to stop.")
while True:
    query = input("\nAsk the Welfare Secretary AI: ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    print("\n" + generate_response(query))