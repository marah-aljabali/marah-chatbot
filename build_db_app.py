import datetime
import hashlib
import os
import shutil
import xml.etree.ElementTree as ET

import requests
from bs4 import SoupStrainer
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========= إعدادات =========
load_dotenv()

DATA_PATH = "data/pdfs"
DB_PATH = "university_db_app"
SITEMAP_URL = "https://www.iugaza.edu.ps/wp-sitemap.xml"
UNIVERSITY_BASE_URL = "https://www.iugaza.edu.ps"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
REQUEST_HEADERS = {"User-Agent": "MarahChatbotKnowledgeBuilder/1.0"}
SKIP_URL_KEYWORDS = ["tag", "author", "feed", "comment", "embed", "/page/"]
SKIP_URL_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".zip", ".rar", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".mp4", ".mp3"
)


def normalize_text(text):
    cleaned = " ".join((text or "").split())
    return cleaned.strip()


def get_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def enrich_document_metadata(doc, source_type):
    doc.metadata["source_type"] = source_type
    doc.metadata["source_label"] = os.path.basename(doc.metadata.get("source", "")) or source_type
    doc.page_content = normalize_text(doc.page_content)
    return doc


def filter_urls(urls):
    filtered = [
        url for url in urls
        if url.startswith(UNIVERSITY_BASE_URL)
        and not any(keyword in url for keyword in SKIP_URL_KEYWORDS)
        and not url.lower().endswith(SKIP_URL_SUFFIXES)
    ]
    print(f"🔍 بعد الفلترة: {len(filtered)} رابط")
    return filtered


def get_all_urls_from_sitemap(sitemap_url):
    print("🗺️ قراءة sitemap...")
    all_urls = set()
    visited_sitemaps = set()

    def parse_sitemap(url):
        if url in visited_sitemaps:
            return
        visited_sitemaps.add(url)

        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
            response.raise_for_status()
            root = ET.fromstring(response.content)

            namespace = ""
            if root.tag.startswith("{"):
                namespace = root.tag.split("}", 1)[0] + "}"

            sitemap_tags = root.findall(f".//{namespace}sitemap")
            for sitemap_tag in sitemap_tags:
                loc_tag = sitemap_tag.find(f"{namespace}loc")
                if loc_tag is not None and loc_tag.text:
                    parse_sitemap(loc_tag.text.strip())

            url_tags = root.findall(f".//{namespace}url")
            for url_tag in url_tags:
                loc_tag = url_tag.find(f"{namespace}loc")
                if loc_tag is not None and loc_tag.text:
                    all_urls.add(loc_tag.text.strip())
        except Exception as e:
            print(f"❌ خطأ في {url}: {e}")

    parse_sitemap(sitemap_url)
    print(f"✅ تم جمع {len(all_urls)} رابط")
    return list(all_urls)


def get_website_urls_from_sitemap(sitemap_url):
    print("🗺️ جاري جلب خريطة الموقع...")
    try:
        urls = get_all_urls_from_sitemap(sitemap_url)
        valid_urls = filter_urls(urls)
        print(f"✅ تم العثور على {len(valid_urls)} رابط صالح في خريطة الموقع.")
        return valid_urls
    except Exception as e:
        print(f"❌ خطأ في جلب خريطة الموقع: {e}")
        print("🔄 الرجوع إلى قائمة روابط يدوية...")
        return [
            f"{UNIVERSITY_BASE_URL}/",
            f"{UNIVERSITY_BASE_URL}/aboutiug/",
            f"{UNIVERSITY_BASE_URL}/facalties/",
            f"{UNIVERSITY_BASE_URL}/division/",
            f"{UNIVERSITY_BASE_URL}/e3lan/",
            f"{UNIVERSITY_BASE_URL}/eservices/",
            f"{UNIVERSITY_BASE_URL}/newstd/",
            f"{UNIVERSITY_BASE_URL}/أخبار-الجامعة/",
        ]


def build_database():
    print("🚀 بدء بناء قاعدة المعرفة...")
    all_documents = []

    urls = get_website_urls_from_sitemap(SITEMAP_URL)
    if urls:
        print("📥 جاري تحميل المحتوى من صفحات الويب...")
        web_loader = WebBaseLoader(
            urls,
            continue_on_failure=True,
            requests_per_second=1,
            bs_kwargs={"parse_only": SoupStrainer(["main", "article", "body"])},
        )
        web_documents = web_loader.load()
        web_documents = [
            enrich_document_metadata(doc, "website")
            for doc in web_documents
            if normalize_text(doc.page_content)
        ]
        print(f"✅ تم تحميل {len(web_documents)} وثيقة من الموقع الإلكتروني.")
        all_documents.extend(web_documents)

    if os.path.exists(DATA_PATH):
        print("📥 تحميل ملفات PDF...")
        pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()
        pdf_docs = [
            enrich_document_metadata(doc, "pdf")
            for doc in pdf_docs
            if normalize_text(doc.page_content)
        ]
        print(f"✅ تم تحميل {len(pdf_docs)} ملف PDF")
        all_documents.extend(pdf_docs)

    if not all_documents:
        print("❌ لا يوجد بيانات!")
        return

    print("✂️ تقسيم النصوص...")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "###", "##", ".", "؟", "!", "؛", "،", " ", ""],
        chunk_size=900,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(all_documents)
    unique_chunks = []
    seen_hashes = set()
    for chunk in chunks:
        chunk.page_content = normalize_text(chunk.page_content)
        if len(chunk.page_content) < 80:
            continue
        chunk_hash = get_hash(chunk.page_content)
        if chunk_hash in seen_hashes:
            continue
        seen_hashes.add(chunk_hash)
        chunk.metadata["hash"] = chunk_hash
        unique_chunks.append(chunk)
    chunks = unique_chunks
    print(f"✅ عدد الأجزاء: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    print("💾 بناء قاعدة البيانات...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
        client_settings=Settings(anonymized_telemetry=False),
    )

    with open("last_update.txt", "w", encoding="utf-8") as file:
        file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("🎉 تم بناء قاعدة البيانات بنجاح!")


if __name__ == "__main__":
    build_database()
