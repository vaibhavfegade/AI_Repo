import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt.chat_agent_executor import create_tool_calling_executor
from langgraph.prebuilt.tool_node import create_tool
from pypdf import PdfReader



# Load environment variables from .env file (e.g. GROQ_API_KEY)
load_dotenv()
print("[DEBUG] Environment variables loaded")

groq_api_key = os.getenv("GROQ_API_KEY")
PDF_DIR = Path("./agentic_rag/data")
print(f"[DEBUG] PDF directory: {PDF_DIR}")
print(f"[DEBUG] GROQ API key available: {bool(groq_api_key)}")


@st.cache_resource
def load_pdf_documents(directory: Path) -> list[Document]:
    """Load all PDF files from a directory and split their text into document chunks. (Cached)."""
    print(f"[DEBUG] Starting PDF loading from {directory}")
    documents: list[Document] = []
    
    if not directory.exists():
        print(f"[DEBUG] Directory {directory} does not exist")
        return documents
    
    pdf_files = list(directory.glob("*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files")
    
    for pdf_path in sorted(pdf_files):
        print(f"[DEBUG] Processing PDF: {pdf_path.name}")
        reader = PdfReader(pdf_path)
        pages = len(reader.pages)
        print(f"[DEBUG]   - Pages in {pdf_path.name}: {pages}")
        
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        text = text.strip()
        if not text:
            print(f"[DEBUG]   - Warning: No text extracted from {pdf_path.name}")
            continue
        
        print(f"[DEBUG]   - Text length: {len(text)} characters")
        
        # Convert the extracted text into smaller chunks for embedding/search
        chunks = split_text(text)
        print(f"[DEBUG]   - Created {len(chunks)} chunks")
        
        documents.extend(
            Document(
                page_content=chunk,
                metadata={"source": str(pdf_path.name), "chunk": idx},
            )
            for idx, chunk in enumerate(chunks, start=1)
        )
    
    print(f"[DEBUG] Total documents loaded: {len(documents)}")
    return documents


def split_text(text: str, chunk_size: int = 5000, chunk_overlap: int = 200) -> list[str]:
    """Split raw text into overlapping chunks for retrieval."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - chunk_overlap
    
    print(f"[DEBUG] Text split: {len(text)} chars → {len(chunks)} chunks")
    return [chunk for chunk in chunks if chunk]


@st.cache_resource
def build_retriever(documents: list[Document]):
    """Create an in-memory vector store retriever using HuggingFace embeddings. (Cached)."""
    print("[DEBUG] Starting retriever setup")
    print("[DEBUG] Initializing HuggingFace embeddings model: all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"[DEBUG] Creating vector store with {len(documents)} documents")
    vector_store = InMemoryVectorStore.from_documents(documents, embeddings)
    
    print("[DEBUG] Creating retriever with k=3 search results")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    print("[DEBUG] Retriever setup complete")
    return retriever


# Load PDF documents and prepare the retriever
print("\n[DEBUG] ===== STARTUP PHASE =====")
pdf_documents = load_pdf_documents(PDF_DIR)
if not pdf_documents:
    print("[DEBUG] WARNING: No PDF documents loaded")
    st.warning(f"No PDF documents found in {PDF_DIR}. Add PDFs and refresh.")

retriever = build_retriever(pdf_documents) if pdf_documents else None
print(f"[DEBUG] Retriever initialized: {retriever is not None}")


@create_tool(
    "pdf_search",
    description=(
        "Search the loaded PDF documents for cybersecurity and AI framework information. "
        "Use this tool when answering user questions based on the PDF content."
    ),
)
def pdf_search(query: str) -> str:
    """Tool function for retrieving relevant PDF chunks from the vector store."""
    print(f"\n[DEBUG] ===== PDF_SEARCH CALLED =====")
    print(f"[DEBUG] Query: {query}")
    
    if retriever is None:
        print("[DEBUG] ERROR: Retriever is None")
        return "No documents are loaded. Please add PDFs to the data folder."

    print("[DEBUG] Invoking retriever...")
    docs = retriever.invoke(query)
    print(f"[DEBUG] Retrieved {len(docs)} documents")
    
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown") if hasattr(doc, 'metadata') else "unknown"
        content_len = len(doc.page_content) if hasattr(doc, 'page_content') else 0
        print(f"[DEBUG]   Doc {i+1}: {source} ({content_len} chars)")
    
    if not docs:
        print("[DEBUG] No relevant documents found")
        return "No relevant content found in the PDFs."
    
    result = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)
    print(f"[DEBUG] Combined result length: {len(result)} characters")
    return result


tools = [pdf_search]
print(f"[DEBUG] Tools registered: {[t.name if hasattr(t, 'name') else str(t) for t in tools]}")

# Streamlit UI setup
st.title("CyberSecurity Chatbot using Groq + LangGraph")

# Initialize the Groq chat model with the API key and model name
@st.cache_resource
def initialize_llm():
    """Initialize ChatGroq LLM (Cached)."""
    print("[DEBUG] Initializing ChatGroq model...")
    llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")
    print("[DEBUG] ChatGroq model initialized")
    return llm

llm = initialize_llm()

# Template for how the agent should answer questions with retrieved context
@st.cache_resource
def initialize_agent():
    """Initialize the agent executor (Cached)."""
    print("[DEBUG] Creating LangGraph executor...")
    # create_tool_calling_executor handles prompt construction internally
    # It will use the LLM's built-in instructions for tool calling
    agent_executor = create_tool_calling_executor(llm, tools)
    print("[DEBUG] LangGraph executor created")
    return agent_executor

agent_executor = initialize_agent()

# Streamlit input and response rendering
query = st.text_input("Enter your cybersecurity query here")
if st.button("Get Answer") or query:
    if not query:
        st.warning("Please enter a query before submitting.")
    else:
        print(f"\n[DEBUG] ===== USER QUERY RECEIVED =====")
        print(f"[DEBUG] Query text: {query}")
        
        start_overall = time.time()
        start_llm = time.process_time()
        
        try:
            print("[DEBUG] Invoking agent_executor...")
            # create_tool_calling_executor expects a list of messages
            response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
            print(f"[DEBUG] Agent response received (type: {type(response).__name__})")
            
            response_time_overall = time.time() - start_overall
            response_time_llm = time.process_time() - start_llm
            
            print(f"[DEBUG] Response times - Overall: {response_time_overall:.2f}s, LLM: {response_time_llm:.2f}s")
            
            # Extract output from response (which contains messages list)
            if isinstance(response, dict) and "messages" in response:
                # Get the last message content
                messages = response.get("messages", [])
                output = messages[-1].content if messages else "No response generated"
            else:
                output = getattr(response, "content", None) or str(response)
            
            print(f"[DEBUG] Output length: {len(str(output))} characters")
            
            st.write(output)
            st.markdown(
                f"<p style='color:blue;'>Overall Response Time: {response_time_overall:.2f} seconds</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='color:blue;'>LLM Response Time: {response_time_llm:.2f} seconds</p>",
                unsafe_allow_html=True,
            )
            
            # Debug info in expander
            with st.expander("🐛 Debug Information"):
                st.write(f"**Query:** {query}")
                st.write(f"**Response Type:** {type(response).__name__}")
                st.write(f"**Total Response Time:** {response_time_overall:.3f} seconds")
                st.write(f"**LLM CPU Time:** {response_time_llm:.3f} seconds")
                st.write(f"**Output Length:** {len(str(output))} characters")
            
            print("[DEBUG] Response successfully rendered to UI")
            
        except Exception as e:
            print(f"[DEBUG] ERROR occurred: {type(e).__name__}: {str(e)}")
            import traceback
            print("[DEBUG] Traceback:")
            traceback.print_exc()
            st.error(f"An error occurred: {e}")
else:
    st.info("Enter a question and click Get Answer.")

print("[DEBUG] ===== SESSION RENDERED =====\n")

