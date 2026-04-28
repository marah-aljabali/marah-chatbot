import os
import requests
import re
from bs4 import BeautifulSoup, SoupStrainer
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import hashlib
import datetime
import shutil

# ========= إعدادات =========
load_dotenv()

DATA_PATH = "data/pdfs"
DB_PATH = "university_db_app"
SITEMAP_URL = "https://www.iugaza.edu.ps/wp-sitemap.xml"
UNIVERSITY_BASE_URL = "https://www.iugaza.edu.ps"
# نموذج يدعم العربية والإنجليزية بشكل جيد وسرعة مقبولة
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2" 

# ========= دالة تنظيف الويب (HTML Cleaning) =========
def clean_html_text(text):
    """
    دالة لإزالة الضوضاء من نصوص الويب (قوائم، فوترات، كلمات غير مفيدة)
    لضمان أن قاعدة البيانات تحتوي على معلومات دراسية فقط.
    """
    if not text:
        return ""
    
    # قائمة كلمات وجمل شائعة في المواقع الإلكترونية نريد التخلص منها
    noise_patterns = [
        r"من نحن", r"اتصل بنا", r"الرئيسية", r"إعلان", r"تسجيل الدخول", r"بحث", 
        r"القائمة الرئيسية", r"Toggle navigation", r"القائمة", r"Scroll to top", 
        r"جميع الحقوق محفوظة", r"حقوق النشر", r"Facebook", r"Twitter", r"Instagram",
        r"بريد الالكتروني", r"رابط", r"انقر هنا", r"للمزيد", r"تابعنا على", r"English", r"العربية"
    ]
    
    # إزالة الأنماط
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # إزالة الأسطر الفارغة المتكررة
    text = re.sub(r'\n+', '\n', text)
    # إزالة المسافات الزائدة
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

# ========= جلب الروابط =========
def get_website_urls_from_sitemap(sitemap_url):
    print("🗺️ Fetching Sitemap...")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(sitemap_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]
        
        valid_urls = [url for url in urls if url.startswith(UNIVERSITY_BASE_URL)]
        
        # 🔍 فلترة: نستثني صفحات الوسائط الكبيرة والملفات لتركز على المحتوى
        valid_urls = [url for url in valid_urls if not any(x in url.lower() for x in ['.jpg', '.png', '.pdf', 'video', 'attachment'])]
        
        print(f"✅ Found {len(valid_urls)} valid URLs.")
        return valid_urls
    except Exception as e:
        print(f"❌ Sitemap Error: {e}")
        # قائمة احتياطية صغيرة في حال فشل السايت ماب
        return [
            f"{UNIVERSITY_BASE_URL}/",
            f"{UNIVERSITY_BASE_URL}/aboutiug/",
            f"{UNIVERSITY_BASE_URL}/facalties/",
            f"{UNIVERSITY_BASE_URL}/division/"
        ]

# ========= بناء قاعدة البيانات =========
def build_database():
    print("🚀 Starting Database Construction (Web + PDF)...")

    all_documents = []

    # ===== 🌐 تحميل الموقع =====
    urls = get_website_urls_from_sitemap(SITEMAP_URL)
    
    # ⚠️ في بيئة GitHub Actions، حاول ألا تحمل آلاف الصفحات لتتجاوز Timeout
    # 150 رابط عادة يكفي لتغطية الدوريات والمعلومات الأساسية
    # يمكنك زيادة الرقم إذا كانت الـ Action وقتها كافياً
    urls = urls[:150] 

    if urls:
        print(f"📥 Loading content from {len(urls)} web pages...")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }

        try:
            # استخدام continue_on_failure لعدم توقف العملية بفشل صفحة واحدة
            web_loader = WebBaseLoader(
                urls,
                continue_on_failure=True, 
                requests_per_second=1, # نحترم الموقع حتى لا يتم حظرنا
                requests_kwargs={"headers": headers},
                bs_kwargs={"parse_only": SoupStrainer("body")}
            )
            web_documents = web_loader.load()
            
            # ✨ عملية التنظيف الجوهرية
            cleaned_web_docs = []
            for doc in web_documents:
                # تنفيذ دالة التنظيف
                cleaned_text = clean_html_text(doc.page_content)
                # نتجاهل النصوص القصيرة جداً (عناوين القوائم مثلاً)
                if len(cleaned_text) > 100: 
                    doc.page_content = cleaned_text
                    doc.metadata["source"] = "website"
                    cleaned_web_docs.append(doc)
            
            print(f"✅ Loaded & Cleaned {len(cleaned_web_docs)} web documents.")
            all_documents.extend(cleaned_web_docs)
        except Exception as e:
            print(f"❌ Web Loading Error: {e}")

    # ===== 📄 تحميل PDF =====
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_data_path = os.path.join(current_dir, DATA_PATH)

    if os.path.exists(absolute_data_path):
        print("📥 Loading PDF files...")
        try:
            pdf_loader = DirectoryLoader(absolute_data_path, loader_cls=PyPDFLoader, silent_errors=True)
            pdf_docs = pdf_loader.load()

            for doc in pdf_docs:
                doc.metadata["source"] = "pdf"

            print(f"✅ Loaded {len(pdf_docs)} pages from PDF files.")
            all_documents.extend(pdf_docs)
        except Exception as e:
            print(f"❌ PDF Loading Error: {e}")
    else:
        print("⚠️ Warning: data/pdfs directory not found.")

    if not all_documents:
        print("❌ No documents to process!")
        return

    # ===== ✂️ تقسيم النصوص (Chunking) =====
    print("✂️ Splitting text into chunks...")
    
    # تعديل الاعدادات لزيادة الدقة:
    # chunk_size=1000: لاستيعاب معلومة كاملة
    # chunk_overlap=200: لضمان عدم فقدان السياق بين الجمل
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ". ", "،", ""] # فواصل مناسبة للعربية
    )
    chunks = splitter.split_documents(all_documents)
    print(f"✅ Final chunks count: {len(chunks)}")

    # ===== 🧠 Embeddings & Save =====
    print(f"🧠 Loading Embedding Model: {EMBEDDING_MODEL}...")
    # نستخدم device='cpu' للتأكد من عدم حدوث مشاكل في GitHub Actions إذا لم يتوفر CUDA
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("💾 Building Vector Database...")
    
    if os.path.exists(DB_PATH):
        print(f"🧹 Cleaning old database at {DB_PATH}...")
        shutil.rmtree(DB_PATH)

    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        
        # كتابة ملف للتأريخ
        update_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("last_update.txt", "w", encoding="utf-8") as f:
            f.write(update_time)
            
        print(f"🎉 Database Updated Successfully! ({update_time})")
    except Exception as e:
        print(f"❌ Error saving database: {e}")

def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

if __name__ == "__main__":
    build_database()
