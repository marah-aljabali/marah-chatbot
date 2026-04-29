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

load_dotenv()

# استدعاء دالة البناء
try:
    from build_db_app import build_database
except ImportError:
    build_database = None

# ─────────────────────────────────────────────────────────────────────────────
# 🚀 UPDATE LOGIC (يجب أن يكون في البداية قبل تحميل أي شيء)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.get("trigger_update", False):
    st.markdown("""
    <div class="fullscreen-overlay">
        <div class="building-anim">
            <div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div>
        </div>
        <div class="overlay-text">Updating Database...</div>
        <div class="overlay-sub">Connecting to University & Files...</div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        db_path = "university_db_app"
        
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            time.sleep(1)
        
        if build_database:
            build_database()
        else:
            raise Exception("Build module not found.")
        
        st.session_state.trigger_update = False
        st.success("✅ Updated Successfully!")
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"Update Failed: {e}")
        time.sleep(3)
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Marah – University Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

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
    display: flex;
    flex-direction: column;
    justify-content: space-between;
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
    margin: 0;
}
.page-hdr-sub { color: #94d5cf; font-size: 1rem; margin-top: 6px; font-weight: 300; }

[data-testid="stChatMessage"] { background-color: transparent !important; }

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
[data-testid="stChatMessage"]:has([data-testid="chat-avatar-assistant"]) .stMarkdown {
    background-color: #fff;
    color: var(--navy);
    padding: 14px 22px;
    border-radius: 16px 16px 16px 0;
    box-shadow: var(--shadow);
    margin-bottom: 10px;
    line-height: 1.6;
}

[data-testid="stChatInput"] {
    background-color: #fff;
    border-radius: 16px;
    padding: 10px;
    box-shadow: var(--shadow);
    border: 1px solid #e2e8f0;
}

.stButton > button {
    background: var(--teal) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
.stButton > button:hover { opacity: .85 !important; }

.splash-screen {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background: linear-gradient(135deg, #f8f7f4 0%, #e2e8f0 100%);
    z-index: 9999;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
    animation: fadeIn 0.8s ease-in-out;
}
.splash-icon { font-size: 5rem; margin-bottom: 20px; animation: float 3s ease-in-out infinite; }
.splash-text {
    font-family: 'DM Serif Display', serif;
    font-size: 2.5rem;
    color: var(--navy);
    margin-bottom: 10px;
}
@keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-20px); } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

.fullscreen-overlay {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background-color: #ffffff;
    z-index: 9999;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
    animation: fadeIn 0.5s ease-out;
}
.building-anim { display: flex; align-items: flex-end; justify-content: center; gap: 10px; height: 60px; margin-bottom: 30px; }
.bar {
    width: 15px; background-color: var(--navy);
    animation: stack 1.5s infinite ease-in-out; border-radius: 4px 4px 0 0;
}
.bar:nth-child(1) { height: 20px; animation-delay: 0.0s; background-color: var(--navy); }
.bar:nth-child(2) { height: 35px; animation-delay: 0.2s; background-color: var(--teal); }
.bar:nth-child(3) { height: 50px; animation-delay: 0.4s; background-color: var(--amber); }
.bar:nth-child(4) { height: 25px; animation-delay: 0.6s; background-color: var(--navy); }
.bar:nth-child(5) { height: 40px; animation-delay: 0.8s; background-color: var(--teal); }
@keyframes stack { 0%, 100% { transform: scaleY(1); } 50% { transform: scaleY(0.4); } }

.overlay-text {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--navy);
    margin-bottom: 10px;
}
.overlay-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #64748b;
}

.typing-cursor {
    display: inline-block; width: 6px; height: 20px;
    background-color: var(--teal); margin-left: 4px;
    animation: blink 1s step-end infinite; vertical-align: middle;
}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

.warn-box {
    background: #fefce8;
    border-left: 4px solid var(--amber);
    border-radius: 0 10px 10px 0;
    padding: 10px 14px; 
    font-size: .8rem; 
    color: #92400e;
    margin-bottom: 10px;
}

.sidebar-footer {
    border-top: 1px solid rgba(255,255,255,0.1);
    text-align: center;
    margin-top: auto; 
    padding-bottom: 1rem;
}
.sidebar-footer h4 {
    color: var(--teal);
    font-family: 'DM Serif Display', serif;
    margin-bottom: 10px;
    font-size: 1rem;
}
.sidebar-footer p {
    font-size: 0.75rem;
    color: #94a3b8;
    margin: 5px 0;
}
.sidebar-footer span.highlight {
    color: #fff;
    font-weight: bold;
}
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
    # تصحيح اسم المودل إلى الإصدار المتاح والمستقر
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0, 
        streaming=True,
        safety_settings=None # قد يكون مطلوباً لبعض المحتويات الجامعية
    )
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    db = Chroma(persist_directory="university_db_app", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4}) # زيادة عدد النتائج قليلاً لضمان الدقة
    return retriever, llm

try:
    retriever, llm = load_components()
    load_overlay.empty()
except Exception as e:
    load_overlay.empty()
    st.error(f"Initialization Failed: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# LOGIC & HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    formatted = ""
    for m in history.messages:
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
    <div class="page-hdr-sub">I answer based on official university files and website only.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR & BUTTON LOGIC
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ System Status")
    
    db_exists = os.path.exists("university_db_app")
    last_update = "Unknown"
    needs_update = False 
    
    if db_exists:
        if os.path.exists("last_update.txt"):
            with open("last_update.txt", "r", encoding="utf-8") as f:
                last_update_str = f.read().strip()
                last_update = last_update_str
                try:
                    last_date = datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S")
                    days_diff = (datetime.now() - last_date).days
                    if days_diff >= 7:
                        needs_update = True
                except:
                    pass
    else:
        needs_update = True

    st.info(f"📅 **{last_update}**")

    if needs_update and db_exists:
        st.markdown("""
        <div class="warn-box">
        ⚠️ Old DB (7+ days).<br>Update recommended.
        </div>
        """, unsafe_allow_html=True)
    elif not needs_update and db_exists:
        st.success("✅ DB is fresh.")
    
    st.markdown("---")
    st.markdown("### 🔄 Database")
    
    if db_exists:
        st.metric("Status", "Ready")
    else:
        st.metric("Status", "Missing", delta_color="inverse")

    update_clicked = st.button(
        "Update Now", 
        use_container_width=True, 
        disabled=not needs_update,
        help="Click to refresh database knowledge."
    )
    
    if update_clicked:
        st.session_state.trigger_update = True
        st.rerun()

    if not needs_update and db_exists:
        st.caption("DB is fresh. Update disabled until 7 days.")

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
    st.session_state.chat_history.add_ai_message("مرحبًا!👋 أنا 'مرح'، المساعد الجامعي الذكي.\nأجيب على أسئلتك بناءً على الملفات الرسمية وموقع الجامعة فقط.")

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
        
        # 1. البحث في قاعدة البيانات (الملفات + الموقع الجامعي فقط)
        contextual_query = f"Current Question: {question}\nContext: {format_history(st.session_state.chat_history)}"
        db_docs = retriever.invoke(contextual_query)
        db_context = format_docs(db_docs)

        # 2. الأمر الـ Prompt الصارم لمنع الهلوسة
        prompt = ChatPromptTemplate.from_template("""
        You are "Marah", a rigorous and accurate university assistant for the Islamic University of Gaza.
        
        CRITICAL INSTRUCTIONS:
        1. Answer ONLY using the context provided below.
        2. The context consists of data from official PDF files and the university website (iugaza.edu.ps).
        3. If the answer is NOT found in the context, strictly say: "عذراً، لم أجد إجابة لهذا السؤال في الملفات أو الموقع الرسمي حالياً." (Sorry, I couldn't find an answer...).
        4. DO NOT use any outside knowledge, general web search, or hallucinate information.
        5. Be concise and helpful.
        6. Answer in the same language as the user's question (Arabic or English).
        
        Context:
        {context}

        Conversation History:
        {history}

        User Question:
        {question}

        Answer:
        """)

        chain = prompt | llm | StrOutputParser()
        history_text = format_history(st.session_state.chat_history)

        try:
            for chunk in chain.stream({"context": db_context, "question": question, "history": history_text}):
                full_response += chunk
                message_placeholder.markdown(full_response + '<span class="typing-cursor"></span>', unsafe_allow_html=True)
            
            message_placeholder.markdown(full_response)
            st.session_state.chat_history.add_ai_message(full_response)

        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")
