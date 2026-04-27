import os
import requests
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
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ========= جلب روابط السايت ماب =========
def get_website_urls_from_sitemap(sitemap_url):
    print("🗺️ جاري جلب خريطة الموقع...")
    try:
        # نستخدم هنا أيضاً User-Agent في حال حظر السايت ماب نفسه
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(sitemap_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]
        
        # فلترة الروابط
        valid_urls = [url for url in urls if url.startswith(UNIVERSITY_BASE_URL)]
        
        print(f"✅ تم العثور على {len(valid_urls)} رابط صالح.")
        return valid_urls
    except Exception as e:
        print(f"❌ خطأ في جلب خريطة الموقع: {e}")
        print("🔄 استخدام قائمة روابط احتياطية...")
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
    print("🚀 بدء بناء قاعدة المعرفة (الموقع والـ PDF)...")

    all_documents = []

    # ===== 🌐 تحميل الموقع (المهم جداً) =====
    urls = get_website_urls_from_sitemap(SITEMAP_URL)
    
    # تحديد عدد الروابط لتجنب التوقيت الطويل جداً في الووركفلو
    # يمكنك زيادة الرقم لاحقاً، لكن 100-200 رابط جيد للتجربة
    urls = urls[:100] 

    if urls:
        print(f"📥 جاري تحميل المحتوى من {len(urls)} صفحة ويب...")
        
        # ⚠️ هذا هو الحل لمشكلة الـ 403: تعريف هوية المتصفح (User-Agent)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }

        try:
            web_loader = WebBaseLoader(
                urls,
                continue_on_failure=True, 
                requests_per_second=1,
                requests_kwargs={"headers": headers}, # تمرير الهيدرز هنا
                bs_kwargs={"parse_only": SoupStrainer("body")}
            )
            web_documents = web_loader.load()
            print(f"✅ تم تحميل {len(web_documents)} وثيقة من الموقع الإلكتروني.")
            
            # إضافة مصدر للموقع للتمييز
            for doc in web_documents:
                doc.metadata["source"] = "website"
                
            all_documents.extend(web_documents)
        except Exception as e:
            print(f"❌ خطأ أثناء تحميل الموقع: {e}")

    # ===== 📄 تحميل PDF =====
    # التأكد من المسار المطلق مرة أخرى لضمان عمله في الووركفلو
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_data_path = os.path.join(current_dir, DATA_PATH)

    if os.path.exists(absolute_data_path):
        print("📥 جاري تحميل ملفات PDF...")
        try:
            # استخدام silent_errors لتجنب توقف العملية بسبب ملف واحد تالف
            pdf_loader = DirectoryLoader(absolute_data_path, loader_cls=PyPDFLoader, silent_errors=True)
            pdf_docs = pdf_loader.load()

            for doc in pdf_docs:
                doc.metadata["source"] = "pdf"

            print(f"✅ تم تحميل {len(pdf_docs)} صفحة من ملفات PDF")
            all_documents.extend(pdf_docs)
        except Exception as e:
            print(f"❌ خطأ أثناء قراءة ملفات PDF: {e}")

    if not all_documents:
        print("❌ لا توجد مستندات لمعالجتها!")
        return

    # ===== ✂️ تقسيم =====
    print("✂️ تقسيم النصوص...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(all_documents)
    print(f"✅ عدد الأجزاء النهائية: {len(chunks)}")

    # ===== 🧠 Embeddings =====
    print(f"🧠 تحميل نموذج التضمين: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # ===== 💾 بناء DB =====
    print("💾 بناء وحفظ قاعدة البيانات...")
    
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    for chunk in chunks:
        chunk.metadata["hash"] = get_hash(chunk.page_content)

    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        
        with open("last_update.txt", "w", encoding="utf-8") as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
        print("🎉 تم تحديث قاعدة البيانات من الموقع وملفات PDF بنجاح!")
    except Exception as e:
        print(f"❌ خطأ أثناء حفظ قاعدة البيانات: {e}")

def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

if __name__ == "__main__":
    build_database()

