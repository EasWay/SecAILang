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
    temperature=0.3,  # Lower temperature for more focused, factual responses
    google_api_key=api_key,
    max_output_tokens=2048
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
You are the Welfare Committee Secretary writing an official report.

CRITICAL INSTRUCTIONS:
1. Use ONLY the data provided in the context below - DO NOT use general knowledge or make assumptions
2. If the context doesn't contain information to answer the question, say "This information is not available in the welfare committee records"
3. Write in first person as the secretary
4. Cover ALL items found in the data systematically
5. Include specific details: names, amounts, dates, locations, attendance numbers, outcomes
6. Organize the report with clear sections for:
   - Financial Summary (collections, expenditures, remaining balance)
   - Events and Activities (with attendance and outcomes)
   - Meetings (with agendas and decisions)
   - Issues and Concerns (with descriptions and status)
7. Use actual numbers and names from the data
8. Make it formal but readable

User request: {question}

Welfare Committee Data:
{context}

Remember: Only use information from the data above. Do not add general knowledge about welfare committees.
"""

normal_template = """
You are the Welfare Committee Secretary answering a specific question.

CRITICAL INSTRUCTIONS:
1. Use ONLY the data provided in the context below
2. If the context doesn't contain the answer, say "This information is not available in the welfare committee records"
3. Be specific and cite actual data (names, amounts, dates, etc.)
4. Keep your answer brief and focused on the question
5. DO NOT use general knowledge or make assumptions

User request: {question}

Welfare Committee Data:
{context}

Answer based only on the data above:
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