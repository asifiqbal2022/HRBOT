import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from openai import OpenAI

# ---------------- OPENAI CLIENT ----------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

class OpenAIWrapper:
    def __init__(self, model="gpt-4.1-mini", temperature=1, max_tokens=512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        # ✅ Make sure this return is inside the function
        return response.choices[0].message['content']

# ---------------- CONFIGURATION ----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_INDEX_PATH = "pdf_knowledge.faiss"

LLM_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0

# ---------------- LLM INITIALIZATION ----------------
llm = OpenAIWrapper(
    model="gpt-4.1-mini",
    temperature=1,
    max_tokens=512
)

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ---------------- VECTORSTORE FUNCTIONS ----------------
def build_vector_store(pdf_files):
    docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_INDEX_PATH)
    return vectorstore

def load_vector_store():
    if os.path.exists(VECTOR_INDEX_PATH):
        return FAISS.load_local(VECTOR_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

# ---------------- INTENT CLASSIFIER ----------------
def classify_intent(user_query):
    keywords = {
        "leave": ["leave", "chutti", "vacation", "casual", "annual", "medical"],
        "policy": ["policy", "rule", "regulation", "procedure"],
        "salary": ["salary", "pay", "wages"],
        "loan": ["loan", "advance"],
        "attendance": ["attendance", "time in", "late", "biometric"]
    }
    for intent, keys in keywords.items():
        if any(k in user_query.lower() for k in keys):
            return intent
    return "general"

# ---------------- PDF SEARCH ----------------
def pdf_search(query, vectorstore):
    if not vectorstore:
        return None
    docs = vectorstore.similarity_search(query, k=2)
    if not docs:
        return None
    return "\n".join([d.page_content for d in docs])

def get_response(user_query, vectorstore):
    # Collect last 6 messages from chat history for memory
    chat_memory = ""
    if "chat_history" in st.session_state:
        recent = st.session_state.chat_history[-6:]
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_memory += f"{role}: {msg['content']}\n"

    pdf_context = pdf_search(user_query, vectorstore)
    pdf_score = 3 if pdf_context else 0

    total_score = pdf_score

    # Case 1 — PDF Only
    if pdf_score > 0:
        prompt = f"""
Use ONLY the PDF text below to answer.

CONTEXT MEMORY:
{chat_memory}

PDF Content:
{pdf_context}

User Question: {user_query}
"""
        return llm.invoke(prompt).content

    # Case 2 — Nothing found
    return (
        "PDF میں اس سوال کا جواب موجود نہیں۔ "
        "براہ کرم سوال کو مزید واضح انداز میں پوچھیں۔"
    )

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="💬 Enterprise Knowledge Chatbot", layout="centered")

st.title("💬 Enterprise Knowledge Chatbot (Phase 2 Enhanced)")
st.caption("PDF Search • No Database • No Hallucination")

with st.sidebar:
    st.header("📄 Knowledge Base Setup")
    uploaded_pdfs = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

    if uploaded_pdfs:
        with st.spinner("Processing PDFs..."):
            pdf_paths = []
            for file in uploaded_pdfs:
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.read())
                pdf_paths.append(temp_path)
            vectorstore = build_vector_store(pdf_paths)
        st.success("Knowledge base updated!")
    else:
        vectorstore = load_vector_store()
        if vectorstore:
            st.info("Existing knowledge base loaded.")
        else:
            st.warning("Upload PDFs to generate knowledge base.")

# Conversation Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question here..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        response = get_response(prompt, vectorstore)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})



