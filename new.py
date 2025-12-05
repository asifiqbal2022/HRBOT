import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text
import os
import oracledb
from openai import OpenAI

# ---------------- OPENAI CLIENT ----------------
client = OpenAI(api_key="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")

class OpenAIWrapper:
    def __init__(self, model="gpt-4.1-mini", temperature=1, max_tokens=512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt):
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message

# ---------------- CONFIGURATION ----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_INDEX_PATH = "pdf_knowledge.faiss"

# Oracle connection parameters
ORACLE_USERNAME = "apps"
ORACLE_PASSWORD = "fagrKu3arTap"
ORACLE_HOST = "10.0.1.153"
ORACLE_PORT = "1521"
ORACLE_SERVICE = "PROD"

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

# ---------------- ORACLE DB FUNCTIONS ----------------
def create_oracle_connection():
    try:
        oracledb.init_oracle_client(lib_dir=r"E:\OracleAI\client\instantclient_21_19")
        connection = oracledb.connect(
            user=ORACLE_USERNAME,
            password=ORACLE_PASSWORD,
            dsn=f"{ORACLE_HOST}:{ORACLE_PORT}/{ORACLE_SERVICE}"
        )
        return connection
    except Exception as e:
        st.error(f"Oracle connection error: {str(e)}")
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

# ---------------- ORACLE QUERY ----------------
def query_oracle_database(user_query):
    try:
        import re
        match = re.search(r'\b\d+\b', user_query)
        if not match:
            return None
        
        employee_id = match.group()

        connection = create_oracle_connection()
        if not connection:
            return None
        
        cursor = connection.cursor()

        sql = """
            SELECT balance 
            FROM CUST_EMPLOYEES_LEAVES
            WHERE employee_id = :emp_id
        """

        cursor.execute(sql, {"emp_id": employee_id})
        row = cursor.fetchone()

        if row:
            return f"Employee ID {employee_id} has a leave balance of {row[0]}."

        return None

    except Exception as e:
        st.error(f"Database query error: {str(e)}")
        return None
        
    finally:
        try:
            if connection:
                connection.close()
        except:
            pass

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

    db_answer = query_oracle_database(user_query)
    db_score = 2 if db_answer else 0

    total_score = pdf_score + db_score

    # Case 1 — PDF + DB (Strong match)
    if total_score >= 4:
        prompt = f"""
You are an accurate HR Assistant bot.

CONTEXT MEMORY (Recent Conversation):
{chat_memory}

INSTRUCTIONS:
- First priority → PDF text
- Second → DB
- If something was discussed earlier, maintain consistency using the above chat memory.
- Never guess anything.
- If user writes Hindi, respond in Urdu.

--- PDF Content ---
{pdf_context}

--- Database Record ---
{db_answer}

User Question: {user_query}
"""
        return llm.invoke(prompt).content

    # Case 2 — PDF Only
    if pdf_score > db_score:
        prompt = f"""
Use ONLY the PDF text below to answer.

CONTEXT MEMORY:
{chat_memory}

PDF Content:
{pdf_context}

User Question: {user_query}

If not fully available in PDF, say:
"More information is needed from the database."
"""
        return llm.invoke(prompt).content

    # Case 3 — DB Only
    if db_score > pdf_score:
        prompt = f"""
Use ONLY the following database information.

CONTEXT MEMORY:
{chat_memory}

Database Record:
{db_answer}

User Question: {user_query}
"""
        return llm.invoke(prompt).content

    # Case 4 — Nothing found
    return (
        "PDFs اور ڈیٹا بیس دونوں میں اس سوال کا جواب موجود نہیں۔ "
        "براہ کرم سوال کو مزید واضح انداز میں پوچھیں۔"
    )

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="💬 Enterprise Knowledge Chatbot", layout="centered")

st.title("💬 Enterprise Knowledge Chatbot (Phase 2 Enhanced)")
st.caption("PDF + Oracle Hybrid Search • Weighted Scoring • No Hallucination")

with st.sidebar:
    st.header("📄 Knowledge Base Setup")
    uploaded_pdfs = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

    st.subheader("🔗 Database Configuration")
    st.write(f"**Host:** {ORACLE_HOST}")
    st.write(f"**Port:** {ORACLE_PORT}")
    st.write(f"**Service:** {ORACLE_SERVICE}")
    st.write(f"**Username:** {ORACLE_USERNAME}")

    if st.button("Test Database Connection"):
        try:
            conn = create_oracle_connection()
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM DUAL")
            st.success("Connection OK")
        except Exception as e:
            st.error(str(e))

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
