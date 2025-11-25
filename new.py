import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
##from langchain.chat_models import ChatOpenAI
from sqlalchemy import create_engine, text
from langchain_groq import ChatGroq
import os

# ---------------- CONFIGURATION ----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_INDEX_PATH = "pdf_knowledge.faiss"
DB_CONNECTION_STRING = "mysql+pymysql://root:password@localhost/mydb"  # update for your DB
LLM_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0

# ---------------- INITIALIZATION ----------------
##llm = ChatOpenAI(model_name=LLM_MODEL, temperature=TEMPERATURE)
llm = ChatGroq(
        api_key="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",
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512,
    )
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ---------------- VECTORSTORE FUNCTIONS ----------------
def build_vector_store(pdf_files):
    """Load PDFs and create FAISS vector store"""
    docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_INDEX_PATH)
    return vectorstore

def load_vector_store():
    """Load FAISS vectorstore from disk"""
    if os.path.exists(VECTOR_INDEX_PATH):
        return FAISS.load_local(VECTOR_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

# ---------------- KNOWLEDGE SEARCH FUNCTIONS ----------------
def pdf_search(query, vectorstore):
    """Search PDF vector database"""
    if not vectorstore:
        return None
    docs = vectorstore.similarity_search(query, k=2)
    if not docs:
        return None
    return "\n".join([d.page_content for d in docs])

##def query_database(user_query):
    ##"""Query fallback database if PDF search fails"""
   ## engine = create_engine(DB_CONNECTION_STRING)
   ## with engine.connect() as conn:
   ##     sql_query = f"SELECT answer FROM knowledge_table WHERE question LIKE '%{user_query}%'"
   ##     result = conn.execute(text(sql_query)).fetchall()
   ## if result:
   ##     return result[0][0]
   ## return None

def get_response(user_query, vectorstore):
    """Main Q&A logic"""
    pdf_context = pdf_search(user_query, vectorstore)
    if pdf_context:
        prompt = f"Based only on this context, answer clearly :\n{pdf_context}\n\nUser: {user_query}"
        return llm.invoke(prompt).content
    else:
                    return "No relevant information found in PDF or database."

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="💬 Enterprise Knowledge Chatbot", layout="centered")

st.title("💬 FG HR Chatbot")
st.caption("Currently Leaves and Travel Policies Available")

# Sidebar
with st.sidebar:
    st.header("📄 Knowledge Base Setup")
    uploaded_pdfs = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)
    if uploaded_pdfs:
        with st.spinner("Processing and indexing PDFs..."):
            pdf_paths = []
            for file in uploaded_pdfs:
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.read())
                pdf_paths.append(temp_path)
            vectorstore = build_vector_store(pdf_paths)
        st.success("✅ Knowledge base updated!")
    else:
        vectorstore = load_vector_store()
        if vectorstore:
            st.info("Loaded existing knowledge base.")
        else:
            st.warning("No knowledge base found. Please upload PDFs.")

# ---------------- CONVERSATION MEMORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input from user
if prompt := st.chat_input("Ask your question here..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        response = get_response(prompt, vectorstore)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})






