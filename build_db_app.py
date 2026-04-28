import os
import re
import time
import shutil
import datetime
from dotenv import load_dotenv

# مكتبات Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# مكتبات LangChain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from bs4 import BeautifulSoup

# ========= إعدادات =========
load_dotenv()

DATA_PATH = "data/pdfs"
DB_PATH = "university_db_app"
UNIVERSITY_BASE_URL = "https://www.iugaza.edu.ps"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2" 

# ========= دالة السكرابنج (محسنة مع انتظار ذكي) =========
def scrape_with_selenium(urls):
    print(f"🌐 Initializing Selenium for {len(urls)} pages (Smart Wait Mode)...")
    
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")

    driver = None
    documents = []

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(45) # زدنا وقت التحميل

        for i, url in enumerate(urls):
            try:
                print(f"🔍 Scraping [{i+1}/{len(urls)}]: {url}")
                driver.get(url)
                
                # ⏱️ إستراتيجية الانتظار:
                # ننتظر 10 ثواني، إذا اكتشفنا صفحة تحقق ننتظر 15 ثانية إضافية
                time.sleep(10)
                
                # فحص سريع هل الصفحة تقول "Enable JS" أو "Checking browser"؟
                if "Enable JavaScript" in driver.page_source or "Checking your browser" in driver.page_source or "Attention Required" in driver.page_source:
                    print("   ⏳ Detected Challenge/Loading page... Waiting extra 15s.")
                    time.sleep(15)

                # جلب الكود
                html_content = driver.page_source
                soup = BeautifulSoup(html_content, 'html.parser')

                # تنظيف السكريبتات
                for element in soup(["script", "style", "noscript", "iframe"]):
                    element.decompose()

                # استخراج البيانات كما في السابق (Links + Text)
                extracted_parts = []
                
                for a_tag in soup.find_all('a'):
                    text = a_tag.get_text(strip=True)
                    if text: extracted_parts.append(f"Link: {text}")
                
                for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                    text = tag.get_text(strip=True)
                    if text: extracted_parts.append(text)

                full_text = "\n".join(extracted_parts)

                # 🐞 التشخيص: طباعة أول 150 حرف لنرى ماذا جلب
                print(f"   📝 Content Sample: {full_text[:150]}...")

                if full_text and len(full_text.strip()) > 50:
                    doc = Document(page_content=full_text, metadata={"source": url})
                    documents.append(doc)
                    print(f"   ✅ Success: Extracted {len(full_text)} chars.")
                else:
                    print(f"   ⚠️ Skipped: Content too short or invalid.")

            except Exception as e:
                print(f"   ❌ Error scraping {url}: {e}")
                continue

    except Exception as e:
        print(f"❌ Selenium/BS4 Error: {e}")
    finally:
        if driver:
            driver.quit()
            
    print(f"🎉 Scraping finished. Collected {len(documents)} documents.")
    return documents

# ========= إعدادات الروابط =========
def get_urls():
    return [
        f"{UNIVERSITY_BASE_URL}/",
            f"{UNIVERSITY_BASE_URL}/aboutiug/",
            f"{UNIVERSITY_BASE_URL}/facalties/",
            f"{UNIVERSITY_BASE_URL}/division/",
            f"{UNIVERSITY_BASE_URL}/e3lan/",
            f"{UNIVERSITY_BASE_URL}/eservices/",
            f"{UNIVERSITY_BASE_URL}/newstd/",
            f"{UNIVERSITY_BASE_URL}/eservices/",
            f"{UNIVERSITY_BASE_URL}/أخبار-الجامعة/"
    ]

# ========= بناء قاعدة البيانات =========
def build_database():
    print("🚀 Starting Database Construction (Smart Wait)...")

    all_documents = []

    # ===== 🌐 تحميل الموقع =====
    urls = get_urls()
    web_documents = scrape_with_selenium(urls)
    all_documents.extend(web_documents)

    # ===== 📄 تحميل PDF =====
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_data_path = os.path.join(current_dir, DATA_PATH)

    if os.path.exists(absolute_data_path):
        print("📥 Loading PDF files...")
        try:
            pdf_loader = DirectoryLoader(absolute_data_path, loader_cls=PyPDFLoader, silent_errors=True)
            pdf_docs = pdf_loader.load()
            print(f"✅ Loaded {len(pdf_docs)} pages from PDF files.")
            all_documents.extend(pdf_docs)
        except Exception as e:
            print(f"❌ PDF Loading Error: {e}")

    if not all_documents:
        print("❌ No documents to process!")
        return

    # ===== ✂️ تقسيم النصوص =====
    print("✂️ Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ". ", "،", ""]
    )
    chunks = splitter.split_documents(all_documents)
    print(f"✅ Final chunks count: {len(chunks)}")

    # ===== 🧠 Embeddings & Save =====
    print(f"🧠 Loading Embedding Model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("💾 Building Vector Database...")
    
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        
        with open("last_update.txt", "w", encoding="utf-8") as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
        print(f"🎉 Database Updated Successfully!")
    except Exception as e:
        print(f"❌ Error saving database: {e}")

if __name__ == "__main__":
    build_database()
