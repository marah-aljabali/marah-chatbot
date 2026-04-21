import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from tavily import TavilyClient

load_dotenv()

st.set_page_config(
    page_title="Marah – University Assistant",
    page_icon="🎓",
    layout="wide",
)

# ───────────── LOAD ─────────────
@st.cache_resource
def load_components():
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    db = Chroma(
        persist_directory="university_db_app",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        streaming=True
    )

    return retriever, llm

retriever, llm = load_components()

# ───────────── UI ─────────────
st.title("🎓 Marah Assistant")

# عرض آخر تحديث
if os.path.exists("last_update.txt"):
    with open("last_update.txt") as f:
        st.caption("Last Update: " + f.read())

# ───────────── CHAT ─────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.chat_history.add_ai_message("مرحبًا 👋أنا مرح، المساعد الذكي للجامعة الإسلامية. \n كيف يمكنني مساعدتك؟")

for msg in st.session_state.chat_history.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

question = st.chat_input("Ask something...")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    formatted = ""
    for m in history.messages:
        if m.type == "human":
            formatted += f"User: {m.content}\n"
        else:
            formatted += f"AI: {m.content}\n"
    return formatted

if question:
    st.session_state.chat_history.add_user_message(question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        db_docs = retriever.invoke(question)
        db_context = format_docs(db_docs)

        web_context = ""
        try:
            tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            result = tavily.search(query=question)
            web_context = "\n".join([r["content"] for r in result["results"]])
        except:
            pass

        final_context = db_context + "\n" + web_context

        prompt = ChatPromptTemplate.from_template("""
        Answer clearly based on context.

        Context:
        {context}

        Question:
        {question}
        """)

        chain = prompt | llm | StrOutputParser()

        for chunk in chain.stream({
            "context": final_context,
            "question": question
        }):
            full_response += chunk
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)
        st.session_state.chat_history.add_ai_message(full_response)
