import os
import requests
from bs4 import BeautifulSoup, SoupStrainer
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import datetime
import shutil
import re

# ========= إعدادات =========
load_dotenv()

DATA_PATH = "data/pdfs"
DB_PATH = "university_db_app"
SITEMAP_URL = "https://www.iugaza.edu.ps/wp-sitemap.xml"
UNIVERSITY_BASE_URL = "https://www.iugaza.edu.ps"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ========= دالة لتنظيف النص =========
def clean_text(text):
    # إزالة المسافات الزائدة والأسطر الفارغة المتكررة
    text = re.sub(r'\s+', ' ', text).strip()
    # يمكن إضافة فلترات خاصة هنا إذا لزم الأمر
    return text

# ========= جلب الروابط (الصفحات + أحدث الأخبار) =========
def get_website_urls_from_sitemap(sitemap_url):
    print("🗺️ جاري جلب خريطة الموقع (فهرس ذكي)...")
    
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    all_urls = []

    try:
        # 1. جلب الـ Sitemap الرئيسي (الفهرس)
        print(f"⬇️ تحميل الفهرس الرئيسي...")
        response = session.get(sitemap_url, headers=headers, timeout=20)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'xml')
        sitemap_tags = soup.find_all('loc')
        
        target_sitemaps = []

        # 2. تحديد الخرائط الفرعية المهمة
        for loc in sitemap_tags:
            url = loc.text
            
            # الحالة أ: الصفحات الثابتة (الأنظمة، الكليات) - نأخذها كلها
            if "wp-sitemap-posts-page" in url:
                target_sitemaps.append(url)
            
            # الحالة ب: الأخبار - نأخذ أول خريطة فقط (تحتوي الأحدث عادة)
            # هذا يمنع دخول أخبار 2018 ويحتفظ بالأخبار الجديدة
            if "wp-sitemap-posts-post-1.xml" in url:
                target_sitemaps.append(url)

        if not target_sitemaps:
            raise Exception("No relevant sitemaps found")

        print(f"✅ تم تحديد {len(target_sitemaps)} مصدر للبيانات.")

        # 3. استخراج الروابط الفعلية من الخرائط الفرعية
        for sitemap_url in target_sitemaps:
            try:
                print(f"📥 جاري قراءة: {sitemap_url.split('/')[-1]}")
                sub_response = session.get(sitemap_url, headers=headers, timeout=20)
                sub_soup = BeautifulSoup(sub_response.content, 'xml')
                
                for loc in sub_soup.find_all('loc'):
                    url = loc.text
                    all_urls.append(url)
            except Exception as e:
                print(f"⚠️ خطأ في قراءة خريطة فرعية: {e}")

        print(f"🎉 تم استخراج {len(all_urls)} رابط.")

    except Exception as e:
        print(f"❌ فشل جلب الـ Sitemap: {e}")
        print("🔄 استخدام القائمة الاحتياطية...")
        all_urls = [
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

    return all_urls

# ========= بناء قاعدة البيانات =========
def build_database():
    print("🚀 بدء عملية بناء قاعدة المعرفة...")
    start_time = datetime.datetime.now()

    all_documents = []

    # ===== 🌐 تحميل الموقع =====
    urls = get_website_urls_from_sitemap(SITEMAP_URL)
    
    if urls:
        print(f"📥 جاري تحميل المحتوى من {len(urls)} رابط...")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }

        try:
            # تحديد المناطق المستهدفة في HTML لتقليل الضوضاء
            bs_kwargs = {"parse_only": SoupStrainer(["main", "article", "div", "body"])}
            
            web_loader = WebBaseLoader(
                urls,
                continue_on_failure=True, 
                requests_per_second=1, # لطيف مع السيرفر
                requests_kwargs={"headers": headers}, 
                bs_kwargs=bs_kwargs
            )
            web_documents = web_loader.load()
            
            # تنظيف وتصنيف الوثائق
            for doc in web_documents:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["source"] = "website"
                
                url = doc.metadata.get('source', '')
                # تصنيف المصدر (هل هو خبر أم صفحة ثابتة؟)
                if "/news/" in url or "/202" in url:
                    doc.metadata["type"] = "news"
                else:
                    doc.metadata["type"] = "page"
                
            all_documents.extend(web_documents)
            print(f"✅ تم معالجة {len(web_documents)} صفحة ويب.")
        except Exception as e:
            print(f"❌ خطأ أثناء تحميل الموقع: {e}")

    # ===== 📄 تحميل PDF =====
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_data_path = os.path.join(current_dir, DATA_PATH)

    if os.path.exists(absolute_data_path):
        print("📥 جاري تحميل ملفات PDF...")
        try:
            pdf_loader = DirectoryLoader(absolute_data_path, loader_cls=PyPDFLoader, silent_errors=True)
            pdf_docs = pdf_loader.load()

            for doc in pdf_docs:
                doc.metadata["source"] = "pdf"
                doc.page_content = clean_text(doc.page_content)

            all_documents.extend(pdf_docs)
            print(f"✅ تم تحميل {len(pdf_docs)} صفحة من ملفات PDF")
        except Exception as e:
            print(f"❌ خطأ أثناء قراءة ملفات PDF: {e}")

    if not all_documents:
        print("❌ لا توجد مستندات لمعالجتها!")
        return

    # ===== ✂️ تقسيم (Chunking) =====
    print("✂️ تقسيم النصوص...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # زيادة الحجم لسياق أفضل
        chunk_overlap=200
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

    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        with open("last_update.txt", "w", encoding="utf-8") as f:
            f.write(end_time.strftime("%Y-%m-%d %H:%M:%S"))
            
        print(f"🎉 تم تحديث قاعدة البيانات بنجاح! (المدة: {duration:.2f} ثانية)")
    except Exception as e:
        print(f"❌ خطأ أثناء حفظ قاعدة البيانات: {e}")

if __name__ == "__main__":
    build_database()
