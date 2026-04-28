import streamlit as st
import os
import time
import shutil
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from tavily import TavilyClient

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Marah – University Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

/* ── Design tokens ── */
:root {
    --navy:   #0f1f3d;
    --teal:   #0d9488;
    --amber:  #f59e0b;
    --cream:  #f8f7f4;
    --shadow: 0 4px 24px rgba(15,31,61,.10);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--cream);
    color: var(--navy);
}

[data-testid="stSidebar"] {
    background: var(--navy) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.block-container { padding-top: 2rem !important; }

.page-hdr {
    background: linear-gradient(120deg, #0f1f3d 0%, #1a3560 60%, #0f5f5a 100%);
    border-radius: 16px;
    padding: 34px 40px;
    margin-bottom: 32px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: var(--shadow);
}
.page-hdr-icon { font-size: 3rem; }
.page-hdr-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #fff;
    line-height: 1.2;
    margin: 0;
}
.page-hdr-sub { color: #94d5cf; font-size: 1rem; margin-top: 6px; font-weight: 300; }

[data-testid="stChatMessage"] { background-color: transparent !important; }

/* User Bubble */
[data-testid="stChatMessage"]:has([data-testid="chat-avatar-user"]) {
    display: flex;
    justify-content: flex-end;
}
[data-testid="stChatMessage"]:has([data-testid="chat-avatar-user"]) .stMarkdown {
    background-color: var(--navy);
    color: #fff;
    padding: 12px 20px;
    border-radius: 16px 16px 0 16px;
    box-shadow: var(--shadow);
}

/* Assistant Bubble */
[data-testid="stChatMessage"]:has([data-testid="chat-avatar-assistant"]) .stMarkdown {
    background-color: #fff;
    color: var(--navy);
    padding: 14px 22px;
    border-radius: 16px 16px 16px 0;
    box-shadow: var(--shadow);
    line-height: 1.6;
}

.stButton > button {
    background: var(--teal) !important;
    color: #fff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Loader */
.splash-screen {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: linear-gradient(135deg, #f8f7f4 0%, #e2e8f0 100%);
    z-index: 9999;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
}
.splash-icon { font-size: 5rem; margin-bottom: 20px; animation: float 3s ease-in-out infinite; }
.splash-text { font-family: 'DM Serif Display', serif; font-size: 2.5rem; color: var(--navy); }
@keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-20px); } }

.typing-cursor {
    display: inline-block; width: 6px; height: 20px;
    background-color: var(--teal); margin-left: 4px;
    animation: blink 1s step-end infinite; vertical-align: middle;
}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

.sidebar-footer { border-top: 1px solid rgba(255,255,255,0.1); text-align: center; margin-top: auto; padding-bottom: 1rem; }
.sidebar-footer h4 { color: var(--teal); font-family: 'DM Serif Display', serif; font-size: 1rem; margin-bottom: 10px; }
.sidebar-footer p { font-size: 0.8rem; color: #94a3b8; margin: 5px 0; }
.sidebar-footer span.highlight { color: #fff; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOADING RESOURCES
# ─────────────────────────────────────────────────────────────────────────────

load_overlay = st.empty()
load_overlay.markdown("""
<div class="splash-screen">
    <div class="splash-icon">🎓</div>
    <div class="splash-text">Marah</div>
    <div class="splash-sub">University Assistant</div>
    <div style="margin-top: 20px; font-size: 0.9rem; color: #0d9488;">Loading AI Models...</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_components():
    # تحميل نموذج التضمين (يجب أن يكون نفس المستخدم في البناء)
    # نستخدم نموذج يدعم العربية بجودة مقبولة وسريع
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # تحميل قاعدة البيانات
    db = Chroma(persist_directory="university_db_app", embedding_function=embeddings)
    
    # نزيد عدد النتائج المسترجعة (k=4) لتعطي الذكاء الاصطناعي خيارات أكثر
    retriever = db.as_retriever(search_kwargs={"k": 4})
    
    # تفعيل الستريمنج
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", # أو gemini-pro حسب ما هو متاح لك
        temperature=0.1, # قللتها قليلاً لتكون أكثر دقة
        streaming=True   
    )
    return retriever, llm

try:
    retriever, llm = load_components()
    load_overlay.empty()
except Exception as e:
    load_overlay.empty()
    st.error(f"Initialization Failed: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def format_docs(docs):
    # تنظيف النصوص المسترجعة
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    formatted = ""
    # نأخذ آخر 4 رسائل فقط للسياق لتقليل التكلفة والتشتت
    recent_history = list(history.messages)[-4:]
    for m in recent_history:
        if m.type == "human":
            formatted += f"Student: {m.content}\n"
        else:
            formatted += f"Marah: {m.content}\n"
    return formatted

# ─────────────────────────────────────────────────────────────────────────────
# UI: HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-hdr">
  <div class="page-hdr-icon">🎓</div>
  <div>
    <div class="page-hdr-title">Marah - University Assistant</div>
    <div class="page-hdr-sub">Ask questions about university courses, departments, and regulations.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ System Status")
    if os.path.exists("university_db_app"):
        st.success("✅ Database is connected.")
    else:
        st.error("❌ Database not found.")
    
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-footer">
      <h4>Marah Assistant</h4>
      <p><span class="highlight">Marah Ahmed Aljabali</span></p>
      <p>© All Rights Reserved 2026.</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.chat_history.add_ai_message("مرحبًا!👋 أنا 'مرح'، المساعد الجامعي الذكي للجامعة الإسلامية.\nأنا مستعدة للإجابة عن استفساراتك.")

# عرض الرسائل السابقة
for msg in st.session_state.chat_history.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

question = st.chat_input("Ask your question here...")

if question:
    st.session_state.chat_history.add_user_message(question)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # ==========================================
        # 🎯 منطق البحث الثابت (Fix Retrieval)
        # ==========================================
        
        # 1. البحث في قاعدة البيانات:
        # المفتاح هنا: نبحث باستخدام السؤال فقط، وليس التاريخ الكامل
        db_docs = retriever.invoke(question)
        db_context = format_docs(db_docs)

        # 2. البحث في الويب (كاحتياطي)
        web_context = ""
        try:
            tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            result = tavily.search(query=question, search_depth="basic")
            if "results" in result:
                web_context = "\n\n".join([r["content"] for r in result["results"][:2]]) # نأخذ أول نتيجتين فقط لتوفير المكان
        except Exception as e:
            print(f"Tavily Error: {e}")
            pass

        # تجميع السياق
        final_context = f"--- معلومات من قاعدة بيانات الجامعة ---\n{db_context}\n\n--- معلومات من الويب ---\n{web_context}"
        
        # 3. تحضير الذاكرة للفهم فقط (وليس للبحث)
        history_text = format_history(st.session_state.chat_history)

        # 4. الـ Prompt المحسن
        prompt = ChatPromptTemplate.from_template("""
        أنت مساعد جامعي ذكي اسمك "مرح" وتعمل في الجامعة الإسلامية - غزة.
        مهمتك الأساسية هي مساعدة الطلاب.

        تعليمات هامة جداً:
        1. أجب على السؤال اعتماداً على "قاعدة بيانات الجامعة" أولاً.
        2. إذا لم تجد الإجابة في قاعدة بيانات الجامعة، استخدم "معلومات الويب".
        3. إذا لم تجد الإجابة في المصدرين، قل بوضوح: "عذراً، لم أجد هذه المعلومة في المصادر المتوفرة حالياً." لا تخترع إجابات.
        4. تكلم بنبرة ودودة وجامعية واحترافية.
        5. إذا كان السؤال بالعربية أجب بالعربية، وإذا كان بالإنجليزية أجب بالإنجليزية.

        المحتوى المتوفر (السياق):
        {context}

        المحادثة السابقة (للفهم السياق العام):
        {history}

        سؤال الطالب:
        {question}

        الإجابة:
        """)

        chain = prompt | llm | StrOutputParser()

        try:
            for chunk in chain.stream({"context": final_context, "question": question, "history": history_text}):
                full_response += chunk
                message_placeholder.markdown(full_response + '<span class="typing-cursor"></span>', unsafe_allow_html=True)
            
            message_placeholder.markdown(full_response)
            st.session_state.chat_history.add_ai_message(full_response)

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
