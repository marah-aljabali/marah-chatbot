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

# ========= جلب كل روابط السايت ماب =========
def get_website_urls_from_sitemap(sitemap_url):
    """
    دالة لجلب كل الروابط من ملف sitemap.xml الخاص بالموقع.
    """
    print(f"🗺️ جاري جلب خريطة الموقع من: {sitemap_url}")
    try:
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]
        
        # فلترة الروابط للتأكد من أنها تابعة للموقع الرئيسي فقط
        valid_urls = [url for url in urls if url.startswith(UNIVERSITY_BASE_URL)]
        
        # استبعاد صفحات غير مفيدة (مثل الوسوم والمؤلفين)
        skip_keywords = ["tag", "author", "feed", "comment", "category"]
        final_urls = [u for u in valid_urls if not any(k in u for k in skip_keywords)]
        
        print(f"✅ تم العثور على {len(final_urls)} رابط صالح.")
        return final_urls
    except Exception as e:
        print(f"❌ خطأ في جلب خريطة الموقع: {e}")
        return []

def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# ========= بناء قاعدة البيانات =========
def build_database():
    print("🚀 بدء بناء قاعدة المعرفة...")
    
    all_documents = []

    # ===== 🌐 1. تحميل الموقع الجامعي فقط =====
    urls = get_website_urls_from_sitemap(SITEMAP_URL)
    
    if urls:
        print(f"📥 جاري تحميل {len(urls)} صفحة من الموقع الرسمي...")
        # استخدام WebBaseLoader مع إعدادات لطيفة لتجنب الحظر
        web_loader = WebBaseLoader(
            urls,
            continue_on_failure=True,
            requests_per_second=1,   # إبطاء الطلبات
            bs_kwargs={"parse_only": SoupStrainer(["main", "article", "h1", "h2", "h3", "p", "li"])} # استخراج المحتوى الرئيسي فقط
        )
        try:
            web_documents = web_loader.load()
            print(f"✅ تم تحميل {len(web_documents)} مستند من الويب.")
            all_documents.extend(web_documents)
        except Exception as e:
            print(f"⚠️ حدث خطأ أثناء تحميل الويب: {e}")

    # ===== 📄 2. تحميل ملفات PDF المرفقة =====
    if os.path.exists(DATA_PATH):
        print("📥 جاري تحميل ملفات PDF...")
        try:
            pdf_loader = DirectoryLoader(DATA_PATH, loader_cls=PyPDFLoader)
            pdf_docs = pdf_loader.load()
            print(f"✅ تم تحميل {len(pdf_docs)} ملف PDF.")
            all_documents.extend(pdf_docs)
        except Exception as e:
            print(f"⚠️ حدث خطأ أثناء تحميل الـ PDFs: {e}")
    else:
        print("⚠️ مجلد data/pdfs غير موجود، سيتم الاعتماد على الموقع فقط.")

    if not all_documents:
        print("❌ لا يوجد بيانات للمعالجة!")
        return

    # ===== ✂️ 3. تقسيم النصوص =====
    print("✂️ جاري تقسيم النصوص...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # زيادة الحجم قليلاً للحفاظ على السياق
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_documents(all_documents)
    print(f"✅ عدد الأجزاء النصية: {len(chunks)}")

    # ===== 🧠 4. إنشاء التضمينات (Embeddings) =====
    print("🧠 جاري تحميل نموذج التضمين...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'} # التأكد من العمل على الـ CPU في Actions
    )

    # ===== 💾 5. بناء وحفظ قاعدة البيانات =====
    if os.path.exists(DB_PATH):
        print("🧹 تنظيف قاعدة البيانات القديمة...")
        shutil.rmtree(DB_PATH)

    print("💾 بناء قاعدة البيانات الجديدة...")

    # إضافة مصدر للبيانات للمساعدة في المرجعية
    for chunk in chunks:
        chunk.metadata["hash"] = get_hash(chunk.page_content)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    # ===== 🕒 حفظ وقت التحديث =====
    with open("last_update.txt", "w", encoding="utf-8") as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("🎉 تم بناء قاعدة البيانات بنجاح!")

if __name__ == "__main__":
    build_database()
