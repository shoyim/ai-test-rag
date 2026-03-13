"""
pipeline.py
═══════════════════════════════════════════════════════════════
PDF → Text Extraction → Chunking → VectorDB → LLM → Structured Output → Save

Bu modul API serveri va CLI tomonidan import qilib ishlatiladi.
Barcha vositalar 100% bepul va local:
  - PyMuPDF       : PDF o'qish
  - sentence-transformers : Embedding (vektorlashtirish)
  - ChromaDB      : Vektor ma'lumotlar bazasi
  - Ollama        : Local LLM (Llama3, Mistral, Gemma2...)
═══════════════════════════════════════════════════════════════
"""

import os
import re
import json
import time
import hashlib
import logging
import requests
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

# ── Logging sozlash ───────────────────────────────────────────
# Har bir modul uchun alohida logger — debug paytida qaysi modul
# xato chiqarayotganini aniq ko'rish uchun
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  KONFIGURATSIYA
#  Barcha sozlamalar bir joyda — environment variable'lar orqali
#  ham o'zgartirib bo'ladi (Docker/server uchun qulay)
# ══════════════════════════════════════════════════════════════

class Config:
    """
    Markaziy konfiguratsiya sinfi.
    Qiymatlarni .env fayl yoki environment variable orqali bekor qilish mumkin:
        OLLAMA_MODEL=mistral python api.py
    """

    # ── Papkalar ─────────────────────────────────────────────
    UPLOAD_DIR   : str = os.getenv("UPLOAD_DIR",  "uploads")    # Yuklangan PDF lar
    OUTPUT_DIR   : str = os.getenv("OUTPUT_DIR",  "output")     # Natijalar (JSON/TXT)
    DB_DIR       : str = os.getenv("DB_DIR",      "chroma_db")  # ChromaDB fayllari

    # ── Chunking ─────────────────────────────────────────────
    # 800 belgi ≈ 150-200 so'z — LLM kontekst oynasiga yaxshi sig'adi
    CHUNK_SIZE   : int = int(os.getenv("CHUNK_SIZE",   "800"))
    # Overlap: har bo'lak oldingi bo'lakning 150 belgisini ham oladi
    # Bu jumlaning o'rtasida bo'linib qolmasligi uchun kerak
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP","150"))

    # ── Ollama (local LLM) ───────────────────────────────────
    OLLAMA_BASE  : str = os.getenv("OLLAMA_BASE", "http://localhost:11434")
    OLLAMA_MODEL : str = os.getenv("OLLAMA_MODEL","llama3")
    # Katta modellar sekin javob berishi mumkin — timeout oshirish kerak
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT","180"))

    # ── Qidiruv ──────────────────────────────────────────────
    # top_k: vektordan nechta bo'lak olinsin — ko'p olsak kontekst
    # to'liqroq, ammo LLM ga yuboriladigan token soni ko'payadi
    TOP_K        : int = int(os.getenv("TOP_K", "5"))

    # ── Standart chiqish miqdorlari ─────────────────────────
    DEFAULT_QA_COUNT : int = 10
    DEFAULT_MCQ_COUNT: int = 5


# ══════════════════════════════════════════════════════════════
#  DATA MODELLARI (Dataclass)
#  Dataclass — oddiy dict o'rniga yozilgan tuzilgan ma'lumot.
#  Afzalligi: type hint, asdict() bilan JSONga aylantirish oson,
#  IDE da avtomatik to'ldirish ishlaydi.
# ══════════════════════════════════════════════════════════════

@dataclass
class TextChunk:
    """
    PDF dan ajratilgan bir bo'lak matn.
    chunk_id: MD5 hash — bir xil bo'lak ikkinchi marta qo'shilmasligi uchun
    """
    chunk_id   : str        # Unikal identifikator (MD5 hash)
    page_num   : int        # PDF sahifa raqami (1 dan boshlanadi)
    text       : str        # Bo'lak matni
    char_count : int        # Belgilar soni
    source_pdf : str        # Manba PDF fayl nomi


@dataclass
class QAPair:
    """
    Bitta savol-javob juftligi.
    API javobida ham, faylda ham shu struktura ishlatiladi.
    """
    question    : str        # Savol matni
    answer      : str        # To'liq javob
    difficulty  : str        # "easy" / "medium" / "hard"
    source_pages: List[int]  # Javob topilgan sahifalar


@dataclass
class MCQQuestion:
    """
    Ko'p javobli test savoli (Multiple Choice Question).
    correct: "A", "B", "C" yoki "D" — faqat bitta to'g'ri javob
    """
    question    : str
    option_a    : str
    option_b    : str
    option_c    : str
    option_d    : str
    correct     : str        # To'g'ri javob harfi
    explanation : str        # Nima uchun bu to'g'ri — o'quv maqsadida
    source_pages: List[int]


@dataclass
class AnswerResult:
    """Oddiy savol-javob natijasi (generate emas, search+answer)"""
    question    : str
    answer      : str
    confidence  : str        # "high" / "medium" / "low"
    key_points  : List[str]  # Asosiy fikrlar ro'yxati
    source_pages: List[int]


@dataclass
class BookInfo:
    """
    Yuklangan kitob haqida metadata.
    API /books/{book_id} endpoint'dan qaytariladi.
    """
    book_id     : str
    filename    : str
    total_pages : int
    total_chunks: int
    upload_time : str
    status      : str        # "ready" | "processing" | "error"


@dataclass
class PipelineResult:
    """To'liq pipeline ishlashi natijasi"""
    pdf_file        : str
    total_pages     : int
    total_chunks    : int
    questions       : List[QAPair]
    test_questions  : List[MCQQuestion]
    processing_time : float
    saved_files     : Dict[str, str] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
#  1-QADAM: PDF TEXT EXTRACTION (Matn Ajratish)
#  PyMuPDF — ochiq kodli, bepul, C++ asosida yozilgan,
#  Python'dan fitz nomi bilan import qilinadi.
#  Scanned PDF uchun OCR kerak (tesseract) — hozircha oddiy PDF
# ══════════════════════════════════════════════════════════════

class PDFExtractor:
    """
    PDF fayldan sahifa-sahifa matn ajratuvchi sinf.

    Ishlatilish:
        extractor = PDFExtractor("kitob.pdf")
        pages = extractor.extract()
        # pages = [{"page": 1, "text": "...", "char_count": 450}, ...]
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = str(pdf_path)  # Path ob'ektini ham qabul qiladi
        self._verify_file()

    def _verify_file(self):
        """Fayl mavjudligi va PDF ekanligini tekshiradi"""
        p = Path(self.pdf_path)
        if not p.exists():
            raise FileNotFoundError(f"PDF topilmadi: '{self.pdf_path}'")
        if p.suffix.lower() != ".pdf":
            raise ValueError(f"Fayl PDF emas: '{self.pdf_path}'")

    def extract(self) -> List[dict]:
        """
        PDF dan barcha sahifalarni o'qib qaytaradi.

        Har bir element:
            {
                "page"      : int,   # Sahifa raqami (1 dan)
                "text"      : str,   # Tozalangan matn
                "char_count": int    # Belgilar soni
            }

        Returns:
            List[dict]: Sahifalar ro'yxati (bo'sh sahifalar o'tkazib yuboriladi)

        Raises:
            ImportError : PyMuPDF o'rnatilmagan bo'lsa
            RuntimeError: PDF o'qishda xato bo'lsa
        """
        # PyMuPDF (fitz) — pip install pymupdf
        try:
            import fitz
        except ImportError:
            raise ImportError(
                "PyMuPDF o'rnatilmagan. O'rnatish: pip install pymupdf"
            )

        logger.info(f"PDF o'qilmoqda: {self.pdf_path}")
        pages_data = []

        try:
            with fitz.open(self.pdf_path) as doc:
                total_pages = len(doc)
                logger.info(f"Jami {total_pages} sahifa")

                for idx in range(total_pages):
                    page = doc[idx]

                    # "text" rejimi: sahifadagi matnni tartib bilan oladi
                    # Muqobil: "blocks" — to'rtburchak bloklarga ajratilgan
                    raw = page.get_text("text")
                    clean = self._clean(raw)

                    # Bo'sh sahifalarni (faqat rasm yoki blank) o'tkazib yuborish
                    if len(clean) < 20:
                        continue

                    pages_data.append({
                        "page"      : idx + 1,   # 0-indexed → 1-indexed
                        "text"      : clean,
                        "char_count": len(clean)
                    })

        except Exception as e:
            raise RuntimeError(f"PDF o'qishda xato: {e}") from e

        total_chars = sum(p["char_count"] for p in pages_data)
        logger.info(f"{len(pages_data)} sahifa, {total_chars:,} belgi ajratildi")
        return pages_data

    def _clean(self, text: str) -> str:
        """
        Xom matnni tozalaydi:
          - 3+ bo'sh qatorni 2 ga kamaytiradi (paragraf strukturasini saqlash)
          - Har satr boshidagi/oxiridagi bo'shliqlarni olib tashlaydi
          - Faqat raqamdan iborat satrlarni (sahifa raqamlari) olib tashlaydi
          - Defis bilan qator oxirida bo'lingan so'zlarni birlashtiradi
        """
        if not text:
            return ""

        # Defis bilan uzilgan so'zlarni birlashtirish (giphenation)
        # Masalan: "infor-\nmation" → "information"
        text = re.sub(r'-\n(\w)', r'\1', text)

        # 3 va undan ortiq bo'sh qatorni 2 ga kamaytirish
        text = re.sub(r'\n{3,}', '\n\n', text)

        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            # Sahifa raqamlarini o'chirish (faqat 1-4 raqamdan iborat satr)
            if re.fullmatch(r'\d{1,4}', stripped):
                continue
            lines.append(stripped)

        return "\n".join(lines).strip()


# ══════════════════════════════════════════════════════════════
#  2-QADAM: TEXT CHUNKING (Matnni Bo'laklash)
#  Nega bo'laklash kerak?
#    - LLM larda kontekst oynasi cheklangan (8K, 16K token)
#    - Katta kitobni to'liq LLM ga yuborib bo'lmaydi
#    - Kichik bo'laklarni vektorlashtirish aniqroq natija beradi
#    - Qidiruv paytida faqat kerakli bo'laklar olinadi
# ══════════════════════════════════════════════════════════════

class TextChunker:
    """
    Matn sahifalarini CHUNK_SIZE belgili bo'laklarga ajratadi.

    Strategiya (ierarxik):
      1. Avval paragraflar bo'yicha ajratiladi (\\n\\n)
      2. Paragraf CHUNK_SIZE dan katta bo'lsa → jumlalar bo'yicha (. ! ?)
      3. Jumla ham katta bo'lsa → qat'iy kesish
      4. Har bo'lak CHUNK_OVERLAP belgi oldingi bo'lakning oxirini oladi

    Overlap misol (chunk=10, overlap=3):
        Bo'lak 1: "abcdefghij"
        Bo'lak 2: "hij klmnop"   ← "hij" takror (overlap)
        Bo'lak 3: "nop qrstuvw"  ← "nop" takror
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        # chunk_size: belgilar soni, taxminan 150-200 so'z
        self.chunk_size = chunk_size
        # overlap: qo'shni bo'laklar orasidagi umumiy belgilar
        self.overlap    = overlap

    def chunk_pages(self, pages: List[dict], source_pdf: str) -> List[TextChunk]:
        """
        Barcha sahifalardan TextChunk ob'ektlar ro'yxati yaratadi.

        Args:
            pages     : PDFExtractor.extract() natijasi
            source_pdf: Manba PDF fayl nomi (metadata uchun)

        Returns:
            List[TextChunk]: Barcha bo'laklar
        """
        all_chunks: List[TextChunk] = []

        for page in pages:
            # Har sahifani alohida bo'laklash
            page_chunks = self._split_text(
                text=page["text"],
                page_num=page["page"],
                source_pdf=source_pdf
            )
            all_chunks.extend(page_chunks)

        logger.info(f"Jami {len(all_chunks)} bo'lak yaratildi")
        return all_chunks

    def _split_text(self, text: str, page_num: int, source_pdf: str) -> List[TextChunk]:
        """
        Bitta sahifa matnini rekursiv bo'laklaydi.
        Har bo'lak TextChunk sifatida qaytariladi.
        """
        chunks: List[TextChunk] = []

        # Paragraflarni ajratish
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # buffer: hozirgi to'planayotgan matn
        # prev_tail: oldingi bo'lakning oxirgi OVERLAP belgisi (overlap uchun)
        buffer    = ""
        prev_tail = ""

        def flush(buf: str):
            """buffer ni bo'lak sifatida saqlaydi"""
            combined = (prev_tail + " " + buf).strip()
            if combined:
                chunks.append(self._make_chunk(combined, page_num, source_pdf))

        for para in paragraphs:
            candidate = (prev_tail + " " + buffer + " " + para).strip()

            if len(candidate) <= self.chunk_size:
                # Hali bo'lak to'lmagan — paragrafni buferga qo'shamiz
                buffer = (buffer + " " + para).strip()
            else:
                # Bo'lak to'ldi — saqlash va yangi boshlash
                if buffer:
                    flush(buffer)
                    # Keyingi bo'lak uchun overlap: oldingi bufferning oxiri
                    prev_tail = buffer[-self.overlap:]
                    buffer    = para
                else:
                    # Bitta paragraf o'zi katta — jumlalarga bo'lish
                    buffer = self._split_by_sentences(
                        para, page_num, source_pdf, chunks, prev_tail
                    )
                    prev_tail = buffer[-self.overlap:] if buffer else ""

        # Qolgan buferni saqlash
        if buffer.strip():
            flush(buffer)

        return chunks

    def _split_by_sentences(
        self, text: str, page_num: int, source_pdf: str,
        chunks: List[TextChunk], prev_tail: str
    ) -> str:
        """
        Katta paragrafni jumlalarga bo'lib, bo'laklarga ajratadi.
        Bo'lmay qolgan qisqini qaytaradi (keyingi paragraf bilan birlashishi uchun).
        """
        # Nuqta, undov, so'roq dan keyin bo'sh joy — jumla chegarasi
        sentences = re.split(r'(?<=[.!?])\s+', text)

        buffer    = ""
        prev_t    = prev_tail

        def flush_s(buf: str):
            combined = (prev_t + " " + buf).strip()
            if combined:
                chunks.append(self._make_chunk(combined, page_num, source_pdf))

        for sent in sentences:
            candidate = (prev_t + " " + buffer + " " + sent).strip()

            if len(candidate) <= self.chunk_size:
                buffer = (buffer + " " + sent).strip()
            else:
                if buffer:
                    flush_s(buffer)
                    prev_t = buffer[-self.overlap:]
                    buffer = sent
                else:
                    # Bitta jumla ham juda katta → qat'iy kesish
                    for i in range(0, len(sent), self.chunk_size - self.overlap):
                        piece = sent[i: i + self.chunk_size].strip()
                        if piece:
                            chunks.append(self._make_chunk(piece, page_num, source_pdf))
                    prev_t = sent[-self.overlap:] if sent else ""
                    buffer = ""

        return buffer  # Keyingi paragraf bilan birlashtirish uchun

    def _make_chunk(self, text: str, page_num: int, source_pdf: str) -> TextChunk:
        """
        TextChunk ob'ekti yaratadi.
        chunk_id: source_pdf + page + text boshidan MD5 hash
        → bir xil matn ikkinchi marta qo'shilmaydi (idempotent)
        """
        raw_id  = f"{source_pdf}:{page_num}:{text[:80]}"
        cid     = hashlib.md5(raw_id.encode("utf-8")).hexdigest()[:16]
        return TextChunk(
            chunk_id   = cid,
            page_num   = page_num,
            text       = text.strip(),
            char_count = len(text),
            source_pdf = source_pdf
        )


# ══════════════════════════════════════════════════════════════
#  3-QADAM: VEKTOR MA'LUMOTLAR BAZASI
#  Embedding: sentence-transformers (multilingual, bepul)
#    - "paraphrase-multilingual-MiniLM-L12-v2" modeli
#    - O'zbek/Rus/Ingliz matnlarini ham vektorlashtira oladi
#    - ~120MB, HuggingFace dan bir marta yuklanadi
#  ChromaDB: disk-based vektor DB (bepul, local)
#    - SQLite + HNSW indeks
#    - Cosine similarity bilan qidiradi
# ══════════════════════════════════════════════════════════════

class VectorDB:
    """
    ChromaDB asosidagi vektor ma'lumotlar bazasi.

    Har bir bo'lak vektorlashtiriladi (embedding) va bazaga saqlanadi.
    Qidiruv paytida query ham vektorlashtiriladi va eng yaqin
    bo'laklar cosine similarity bilan topiladi.

    Ishlatilish:
        db = VectorDB("chroma_db", collection_name="my_book")
        db.add_chunks(chunks)
        results = db.search("asosiy g'oya nima?", top_k=5)
    """

    # Ko'p tilli embedding modeli — o'zbek matni uchun ham ishlaydi
    EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, db_path: str, collection_name: str = "pdf_chunks"):
        """
        Args:
            db_path        : ChromaDB fayllari saqlanadigan papka
            collection_name: Kolleksiya nomi (har kitob uchun alohida bo'lishi mumkin)
        """
        self.db_path    = db_path
        self.coll_name  = collection_name
        self.collection = None
        self._init_db()

    def _init_db(self):
        """ChromaDB va embedding modelini bir marta yuklaydi"""
        try:
            import chromadb
            from chromadb.utils import embedding_functions as ef
        except ImportError:
            raise ImportError(
                "ChromaDB o'rnatilmagan. O'rnatish: pip install chromadb sentence-transformers"
            )

        logger.info(f"ChromaDB yuklanmoqda: {self.db_path}")

        # sentence-transformers embedding funksiyasi
        # Birinchi marta chaqirilganda model yuklanadi (~30 sekund)
        self._embed_fn = ef.SentenceTransformerEmbeddingFunction(
            model_name=self.EMBED_MODEL
        )

        # PersistentClient: diskka yozadi, dastur to'xtatilsa ham saqlanadi
        client = chromadb.PersistentClient(path=self.db_path)

        # Kolleksiyani olish yoki yaratish
        # metadata: qidiruv algoritmi — cosine (0=bir xil, 1=butunlay boshqa)
        self.collection = client.get_or_create_collection(
            name=self.coll_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Kolleksiya '{self.coll_name}': {self.collection.count()} bo'lak")

    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """
        Bo'laklarni vektorlashtirib bazaga qo'shadi.
        Allaqachon mavjud bo'laklarni (chunk_id bo'yicha) qayta qo'shmaydi.

        Args:
            chunks: TextChunk ro'yxati

        Returns:
            int: Qo'shilgan yangi bo'laklar soni
        """
        if not chunks:
            return 0

        # Bazadagi mavjud ID larni olish
        existing = set(self.collection.get(include=[])["ids"])

        # Yangi bo'laklarni filtrlash
        new_chunks = [c for c in chunks if c.chunk_id not in existing]

        if not new_chunks:
            logger.info("Yangi bo'lak yo'q — baza yangilangan")
            return 0

        logger.info(f"{len(new_chunks)} ta yangi bo'lak qo'shilmoqda...")

        # ChromaDB ga 100 tadan qo'shish (xotira tejash uchun batch)
        BATCH = 100
        for i in range(0, len(new_chunks), BATCH):
            batch = new_chunks[i: i + BATCH]

            # ChromaDB 3 ta parallel ro'yxat kutadi:
            # ids       : unikal identifikatorlar
            # documents : matn (embedding ushandan olinadi)
            # metadatas : qo'shimcha ma'lumotlar (qidiruv filtrida ishlatiladi)
            self.collection.add(
                ids       = [c.chunk_id  for c in batch],
                documents = [c.text      for c in batch],
                metadatas = [
                    {"page": c.page_num, "source": c.source_pdf}
                    for c in batch
                ]
            )
            logger.debug(f"Batch {i // BATCH + 1} qo'shildi")

        logger.info(f"Baza yangilandi. Jami: {self.collection.count()} bo'lak")
        return len(new_chunks)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Matniy so'rov bo'yicha eng mos bo'laklarni qidiradi.

        Jarayon:
          1. query → embedding vektoriga aylantiradi
          2. HNSW (Hierarchical Navigable Small World) algoritmi bilan
             cosine similarity bo'yicha eng yaqin top_k bo'lakni topadi
          3. Natijalarni tartiblab qaytaradi

        Args:
            query : Qidiruv matni (savolning o'zi)
            top_k : Nechta natija kerak

        Returns:
            List[dict]: [
                {
                    "text" : str,   # Bo'lak matni
                    "page" : int,   # Sahifa raqami
                    "score": float  # O'xshashlik (0-1, kattasi yaxshi)
                },
                ...
            ]
        """
        # Bazada top_k dan kam bo'lak bo'lsa — mavjud soni bilan qidirish
        k       = min(top_k, self.collection.count())
        if k == 0:
            return []

        results = self.collection.query(
            query_texts = [query],
            n_results   = k,
            include     = ["documents", "metadatas", "distances"]
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # ChromaDB cosine distance qaytaradi (0=bir xil, 2=butunlay farq)
            # Similarity ga aylantirish: 1 - (dist / 2)
            # Lekin ko'pchilik hollarda: similarity ≈ 1 - dist (normalized)
            score = round(max(0.0, 1.0 - dist), 4)
            output.append({
                "text" : doc,
                "page" : meta.get("page", 0),
                "score": score
            })

        # Score bo'yicha kamayish tartibida saralash
        output.sort(key=lambda x: x["score"], reverse=True)
        return output

    def count(self) -> int:
        """Bazadagi jami bo'laklar soni"""
        return self.collection.count()

    def delete_collection(self):
        """Kolleksiyani to'liq o'chiradi (kitobni qayta yuklash uchun)"""
        import chromadb
        client = chromadb.PersistentClient(path=self.db_path)
        try:
            client.delete_collection(self.coll_name)
            logger.info(f"Kolleksiya '{self.coll_name}' o'chirildi")
        except Exception:
            pass  # Mavjud bo'lmasa — xato emas


# ══════════════════════════════════════════════════════════════
#  4-QADAM: OLLAMA LLM (Local, Bepul)
#  Ollama — local LLM server. Bir marta o'rnatiladi,
#  internet kerak emas, barcha so'rovlar kompyuterda ishlaydi.
#
#  O'rnatish:
#    Linux/Mac: curl -fsSL https://ollama.com/install.sh | sh
#    Windows  : https://ollama.com/download
#    Model    : ollama pull llama3  (yoki mistral, gemma2, phi3)
#
#  API: HTTP REST, port 11434
#    POST /api/generate  → matn generatsiya
#    GET  /api/tags      → o'rnatilgan modellar
# ══════════════════════════════════════════════════════════════

class OllamaLLM:
    """
    Ollama HTTP API orqali local LLM bilan ishlaydi.

    Bu sinf har qanday Ollama modelini qo'llab-quvvatlaydi:
    llama3, mistral, gemma2, phi3, qwen2, codellama va h.k.

    Ishlatilish:
        llm = OllamaLLM(model="llama3")
        response = llm.generate("Python nima?")
    """

    def __init__(
        self,
        model   : str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout : int = 180
    ):
        self.model      = model
        self.api_url    = f"{base_url}/api/generate"
        self.tags_url   = f"{base_url}/api/tags"
        self.timeout    = timeout
        self._verify_connection()

    def _verify_connection(self):
        """
        Ollama server ishlayotganini va model yuklanganini tekshiradi.
        Server yo'q bo'lsa — tushunarli xato xabari chiqaradi.
        """
        try:
            resp = requests.get(self.tags_url, timeout=5)
            resp.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Ollama server topilmadi: {self.api_url}\n"
                f"Ishga tushirish: ollama serve\n"
                f"O'rnatish: https://ollama.com/download"
            )
        except requests.Timeout:
            raise ConnectionError("Ollama server javob bermadi (5 sekund)")

        # O'rnatilgan modellarni tekshirish
        models = [m["name"] for m in resp.json().get("models", [])]
        model_found = any(self.model in m for m in models)

        if not model_found:
            logger.warning(
                f"'{self.model}' modeli topilmadi!\n"
                f"O'rnatish: ollama pull {self.model}\n"
                f"Mavjud modellar: {models}"
            )
        else:
            logger.info(f"Ollama tayyor | Model: {self.model}")

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """
        LLM ga prompt yuborib, matn generatsiya qiladi.

        Args:
            prompt     : So'rov matni (system + user birlashtirilgan)
            temperature: Kreativlik darajasi
                         0.0 = deterministik, bir xil javob
                         0.7 = kreativ, turli javoblar
                         1.0 = juda kreativ, ba'zan noto'g'ri

        Returns:
            str: Model javobi (tozalangan)

        Raises:
            TimeoutError: Model OLLAMA_TIMEOUT ichida javob bermasa
            RuntimeError: HTTP xato
        """
        payload = {
            "model" : self.model,
            "prompt": prompt,
            "stream": False,          # To'liq javobni bir yo'la olish
            "options": {
                "temperature"   : temperature,
                "num_predict"   : 3000,   # Maksimal generatsiya token soni
                "top_p"         : 0.9,    # Nucleus sampling
                "repeat_penalty": 1.1,    # Takrorlanishni kamaytirish
                "stop"          : ["```\n\n", "Human:", "User:"]  # Stop tokenlar
            }
        }

        try:
            resp = requests.post(
                self.api_url,
                json   = payload,
                timeout= self.timeout
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()

        except requests.Timeout:
            raise TimeoutError(
                f"Model {self.timeout}s da javob bermadi.\n"
                f"Config.OLLAMA_TIMEOUT ni oshiring yoki kichikroq model ishlating."
            )
        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP xato: {e}")
        except KeyError:
            raise RuntimeError(f"Ollama noto'g'ri javob qaytardi: {resp.text[:200]}")


# ══════════════════════════════════════════════════════════════
#  5-QADAM: STRUCTURED OUTPUT GENERATOR
#  Bu sinf LLM dan tuzilgan (JSON) javob oladi.
#
#  Muammo: LLM erkin matn generatsiya qiladi, biz esa JSON kerak.
#  Yechim:
#    1. Promptda qat'iy JSON format talab qilinadi
#    2. LLM javobi JSON parse qilinadi
#    3. Xato bo'lsa — qayta urinish yoki fallback
#
#  RAG (Retrieval Augmented Generation) jarayoni:
#    Savol → VectorDB dan top_k bo'lak → Kontekst + Savol → LLM → Javob
# ══════════════════════════════════════════════════════════════

class StructuredOutputGenerator:
    """
    RAG asosida savol-javob va testlar yaratuvchi sinf.

    RAG = Retrieval Augmented Generation:
      - Retrieval: vektor DB dan kontekst topish
      - Augmented: kontekst + savol birlashtirib LLM ga berish
      - Generation: LLM javob generatsiya qiladi

    Bu yondashuv oddiy LLM dan yaxshiroq:
      ✓ Faqat kitobdagi ma'lumotlarga asoslanadi (gallyutsinatsiya kam)
      ✓ Manba sahifalarini ko'rsatadi
      ✓ Katta kitoblarni to'liq qayta ishlay oladi
    """

    def __init__(self, llm: OllamaLLM, vector_db: VectorDB):
        self.llm       = llm
        self.vector_db = vector_db

    # ─────────────────────────────────────────────────────────
    # ODDIY SAVOL-JAVOB (RAG Answer)
    # ─────────────────────────────────────────────────────────

    def answer(self, question: str) -> AnswerResult:
        """
        Savolga kitob asosida javob beradi.

        Jarayon:
          1. question → vector search → top_k bo'lak
          2. Bo'laklar kontekst sifatida LLM ga beriladi
          3. LLM faqat shu kontekstdan javob yozadi
          4. JSON parse qilinib AnswerResult qaytariladi

        Args:
            question: Foydalanuvchi savoli

        Returns:
            AnswerResult: Javob, key_points, sahifalar
        """
        # Vektor qidirish
        hits    = self.vector_db.search(question, top_k=Config.TOP_K)
        context = self._build_context(hits)
        pages   = sorted(set(h["page"] for h in hits))

        # Prompt: LLM ga faqat kontekstdan javob berishni buyuradi
        prompt = f"""You are an expert assistant answering questions based ONLY on the provided book excerpts.
Do NOT use any outside knowledge. Answer in the same language as the question.

BOOK EXCERPTS:
{context}

QUESTION: {question}

Respond with ONLY valid JSON, no extra text:
{{
  "answer"    : "comprehensive answer based on the excerpts",
  "confidence": "high | medium | low",
  "key_points": ["point 1", "point 2", "point 3"]
}}"""

        raw    = self.llm.generate(prompt, temperature=0.2)
        parsed = self._parse_json(raw)

        # Agar JSON parse bo'lmasa — xom javobni ishlatish
        if isinstance(parsed, dict):
            answer_text = parsed.get("answer", raw)
            confidence  = parsed.get("confidence", "medium")
            key_points  = parsed.get("key_points", [])
        else:
            answer_text = raw
            confidence  = "low"
            key_points  = []

        return AnswerResult(
            question    = question,
            answer      = answer_text,
            confidence  = confidence,
            key_points  = key_points,
            source_pages= pages
        )

    # ─────────────────────────────────────────────────────────
    # SAVOLLAR GENERATSIYA (Open-ended Q&A)
    # ─────────────────────────────────────────────────────────

    def generate_qa_pairs(
        self,
        count     : int = 10,
        difficulty: str = "hard"
    ) -> List[QAPair]:
        """
        Kitob mazmunidan ochiq savol-javoblar generatsiya qiladi.

        Keng qidiruv: bir nechta turli query bilan vector DB dan
        turli mavzulardagi bo'laklar olinadi — bu natijaning
        xilma-xilligini ta'minlaydi.

        Args:
            count     : Nechta savol-javob kerak
            difficulty: "easy" | "medium" | "hard"

        Returns:
            List[QAPair]: Savol-javob juftliklari
        """
        # Keng kontekst yig'ish uchun turli xil so'rovlar
        seed_queries = [
            "main ideas and core concepts",
            "important definitions and terminology",
            "key facts and specific details",
            "theories, frameworks and methodologies",
            "examples, case studies and applications",
            "conclusions, results and findings",
            "relationships between concepts",
            "historical context and background"
        ]

        # Har so'rovdan 3 ta natija olib, takrorlarni olib tashlash
        seen, hits = set(), []
        for q in seed_queries:
            for h in self.vector_db.search(q, top_k=3):
                if h["text"] not in seen:
                    hits.append(h)
                    seen.add(h["text"])

        # Eng score'li 12 ta bo'lakni ishlatish
        hits    = sorted(hits, key=lambda x: x["score"], reverse=True)[:12]
        context = self._build_context(hits)
        pages   = sorted(set(h["page"] for h in hits))

        # Qiyinlik darajasi tavsifi (LLM ga ko'rsatma)
        diff_map = {
            "easy"  : "factual recall — definitions, basic facts",
            "medium": "comprehension — explain concepts, compare ideas",
            "hard"  : "analysis and synthesis — critical thinking, implications, connections"
        }
        diff_desc = diff_map.get(difficulty, diff_map["hard"])

        prompt = f"""You are an expert educator. Create exactly {count} {difficulty.upper()} difficulty questions
based ONLY on the book content below.

Difficulty criterion: {diff_desc}

BOOK CONTENT:
{context}

RULES:
- Questions must be directly answerable from the content
- Answers must be detailed and comprehensive (3-5 sentences minimum)
- Use the content's language or Uzbek
- NO questions about page numbers or formatting

Return ONLY a valid JSON array, no markdown, no explanation:
[
  {{
    "question"  : "question text",
    "answer"    : "detailed answer (3-5 sentences)",
    "difficulty": "{difficulty}"
  }}
]"""

        logger.info(f"LLM: {count} ta {difficulty} savol generatsiya qilinmoqda...")
        t0  = time.time()
        raw = self.llm.generate(prompt, temperature=0.45)
        logger.info(f"LLM javob vaqti: {time.time()-t0:.1f}s")

        parsed = self._parse_json(raw)

        qa_list: List[QAPair] = []
        if isinstance(parsed, list):
            for item in parsed[:count]:
                if not isinstance(item, dict):
                    continue
                q = item.get("question", "").strip()
                a = item.get("answer", "").strip()
                if q and a:
                    qa_list.append(QAPair(
                        question    = q,
                        answer      = a,
                        difficulty  = item.get("difficulty", difficulty),
                        source_pages= pages
                    ))

        logger.info(f"{len(qa_list)} ta savol-javob yaratildi")
        return qa_list

    # ─────────────────────────────────────────────────────────
    # TEST GENERATSIYA (MCQ — Multiple Choice Questions)
    # ─────────────────────────────────────────────────────────

    def generate_mcq(self, count: int = 5) -> List[MCQQuestion]:
        """
        Ko'p javobli test (MCQ) savollar yaratadi.

        MCQ formatining afzalligi:
          - Tez tekshirish mumkin (avtomatik)
          - O'quvchi bilimini ob'ektiv baholaydi
          - Noto'g'ri javoblar ham o'rgatuvchi bo'lishi kerak

        Args:
            count: Test savollar soni

        Returns:
            List[MCQQuestion]: MCQ savollar
        """
        # Faktlar va aniq ma'lumotlar bo'lgan bo'laklarni qidirish
        queries = [
            "specific facts, numbers and key concepts",
            "definitions and terminology",
            "processes and procedures"
        ]

        seen, hits = set(), []
        for q in queries:
            for h in self.vector_db.search(q, top_k=4):
                if h["text"] not in seen:
                    hits.append(h)
                    seen.add(h["text"])

        hits    = hits[:10]
        context = self._build_context(hits)
        pages   = sorted(set(h["page"] for h in hits))

        prompt = f"""You are an expert educator creating multiple choice questions (MCQ).
Based ONLY on the content below, create exactly {count} MCQ questions.

CONTENT:
{context}

STRICT RULES:
- Exactly ONE correct answer per question
- "correct" field must be exactly "A", "B", "C", or "D"
- Wrong options must be plausible but clearly incorrect based on content
- Explanation must reference the content

Return ONLY a valid JSON array:
[
  {{
    "question"   : "question text here",
    "option_a"   : "first choice",
    "option_b"   : "second choice",
    "option_c"   : "third choice",
    "option_d"   : "fourth choice",
    "correct"    : "A",
    "explanation": "why A is correct based on the content"
  }}
]"""

        logger.info(f"LLM: {count} ta MCQ generatsiya qilinmoqda...")
        raw    = self.llm.generate(prompt, temperature=0.35)
        parsed = self._parse_json(raw)

        mcq_list: List[MCQQuestion] = []
        if isinstance(parsed, list):
            for item in parsed[:count]:
                if not isinstance(item, dict):
                    continue
                correct = item.get("correct", "A").strip().upper()
                # To'g'ri javob faqat A/B/C/D bo'lishi shart
                if correct not in ("A", "B", "C", "D"):
                    correct = "A"
                q = item.get("question", "").strip()
                if not q:
                    continue
                mcq_list.append(MCQQuestion(
                    question    = q,
                    option_a    = item.get("option_a", ""),
                    option_b    = item.get("option_b", ""),
                    option_c    = item.get("option_c", ""),
                    option_d    = item.get("option_d", ""),
                    correct     = correct,
                    explanation = item.get("explanation", ""),
                    source_pages= pages
                ))

        logger.info(f"{len(mcq_list)} ta MCQ yaratildi")
        return mcq_list

    # ─────────────────────────────────────────────────────────
    # YORDAMCHI METODLAR
    # ─────────────────────────────────────────────────────────

    def _build_context(self, hits: List[dict]) -> str:
        """
        Qidiruv natijalarini LLM prompt uchun formatlaydi.
        Har bo'lak sahifa raqami bilan ko'rsatiladi.
        """
        parts = []
        for hit in hits:
            parts.append(f"[Sahifa {hit['page']} | Score: {hit['score']}]\n{hit['text']}")
        return "\n\n---\n\n".join(parts)

    def _parse_json(self, text: str):
        """
        LLM javobidan JSON ob'ekt yoki massiv ajratib oladi.

        LLM ba'zan JSON ni markdown blokiga o'raydi:
            ```json
            [...]
            ```
        Bu funksiya ularni olib tashlaydi va JSON parse qiladi.

        Muvaffaqiyatsiz bo'lsa — xom matnni qaytaradi.
        """
        if not text:
            return None

        # Markdown code block larini tozalash
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        # JSON array yoki object chegaralarini topish
        # Muvozanatlangan qavslarni izlash algoritmi
        for open_c, close_c in [('[', ']'), ('{', '}')]:
            start = text.find(open_c)
            if start == -1:
                continue

            depth   = 0
            end_pos = -1
            in_str  = False   # String ichidamizmi
            escape  = False   # Oldingi belgi backslash edimi

            for i, ch in enumerate(text[start:], start):
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_str:
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == open_c:
                    depth += 1
                elif ch == close_c:
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break

            if end_pos > start:
                try:
                    return json.loads(text[start:end_pos])
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse xato: {e}")

        # JSON topilmadi — xom matnni qaytarish (fallback)
        logger.warning("JSON tuzilishi topilmadi, xom matn qaytarildi")
        return text


# ══════════════════════════════════════════════════════════════
#  6-QADAM: NATIJALARNI SAQLASH
#  Barcha natijalar bir nechta formatda saqlanadi:
#    full.json  — barcha ma'lumotlar (machine-readable)
#    qa.txt     — savol-javoblar (odam o'qiydigan)
#    test.txt   — MCQ testlar (odam o'qiydigan)
#    quiz.json  — test app uchun strukturalangan JSON
# ══════════════════════════════════════════════════════════════

class ResultSaver:
    """
    Pipeline natijalarini turli formatlarda saqlaydi.

    Ishlatilish:
        saver = ResultSaver("output")
        paths = saver.save_all(result)
        print(paths["json"])  # output/kitob_20241215_143022_full.json
    """

    def __init__(self, output_dir: str):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def save_all(self, result: PipelineResult) -> Dict[str, str]:
        """
        Barcha formatda saqlaydi va fayl yo'llarini qaytaradi.

        Args:
            result: PipelineResult ob'ekti

        Returns:
            Dict: {"json": "...", "qa_txt": "...", "test_txt": "...", "quiz_json": "..."}
        """
        # Fayl nomida timestamp: bir xil kitobni qayta ishlaganda ustiga yozmaslik
        ts   = time.strftime("%Y%m%d_%H%M%S")
        stem = Path(result.pdf_file).stem   # "kitob.pdf" → "kitob"
        pre  = f"{stem}_{ts}"

        paths = {}

        # 1. To'liq JSON (barcha ma'lumot)
        p = self.out / f"{pre}_full.json"
        self._write_json(result, p)
        paths["json"] = str(p)

        # 2. Savol-javob TXT (odam o'qiydigan)
        p = self.out / f"{pre}_qa.txt"
        self._write_qa_txt(result, p)
        paths["qa_txt"] = str(p)

        # 3. Test TXT
        p = self.out / f"{pre}_test.txt"
        self._write_test_txt(result, p)
        paths["test_txt"] = str(p)

        # 4. Quiz JSON (frontend app uchun)
        p = self.out / f"{pre}_quiz.json"
        self._write_quiz_json(result, p)
        paths["quiz_json"] = str(p)

        logger.info(f"Natijalar saqlandi: {self.out}")
        return paths

    def save_chunks_csv(self, chunks: List[TextChunk], pdf_name: str) -> str:
        """
        Bo'laklarni CSV faylga saqlaydi (debug va tahlil uchun).
        CSV ni Excel yoki pandas bilan ochish mumkin.
        """
        path = self.out / f"{Path(pdf_name).stem}_chunks.csv"
        with open(path, "w", encoding="utf-8") as f:
            # Sarlavha
            f.write("chunk_id,page_num,char_count,text_preview\n")
            for c in chunks:
                # Matn ko'rinishi (100 belgi)
                preview = c.text[:100].replace('"', "'").replace("\n", " ")
                f.write(f'"{c.chunk_id}",{c.page_num},{c.char_count},"{preview}"\n')
        return str(path)

    # ── Private write metodlari ───────────────────────────────

    def _write_json(self, result: PipelineResult, path: Path):
        """To'liq natijani JSON formatda saqlaydi"""
        data = {
            "metadata": {
                "pdf_file"       : result.pdf_file,
                "total_pages"    : result.total_pages,
                "total_chunks"   : result.total_chunks,
                "processing_time": f"{result.processing_time:.2f}s",
                "generated_at"   : time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "questions"    : [asdict(q) for q in result.questions],
            "test_questions": [asdict(q) for q in result.test_questions]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _write_qa_txt(self, result: PipelineResult, path: Path):
        """Savol-javoblarni o'qilishi qulay TXT formatda yozadi"""
        W = 62  # Chiziq kengligi
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * W + "\n")
            f.write(f"  KITOB : {result.pdf_file}\n")
            f.write(f"  SAVOL-JAVOB ({len(result.questions)} ta)\n")
            f.write("=" * W + "\n\n")

            for i, qa in enumerate(result.questions, 1):
                f.write(f"{'─' * W}\n")
                f.write(f"  {i:2}. [{qa.difficulty.upper()}]\n")
                f.write(f"{'─' * W}\n")
                f.write(f"❓ SAVOL:\n{qa.question}\n\n")
                f.write(f"✅ JAVOB:\n{qa.answer}\n\n")
                sahifalar = ", ".join(str(p) for p in qa.source_pages)
                f.write(f"📍 Sahifalar: {sahifalar}\n\n")

    def _write_test_txt(self, result: PipelineResult, path: Path):
        """MCQ testlarni TXT formatda yozadi"""
        W = 62
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * W + "\n")
            f.write(f"  KITOB : {result.pdf_file}\n")
            f.write(f"  KO'P JAVOBLI TEST ({len(result.test_questions)} ta)\n")
            f.write("=" * W + "\n\n")

            for i, q in enumerate(result.test_questions, 1):
                f.write(f"{'─' * W}\n")
                f.write(f"  {i:2}. {q.question}\n")
                f.write(f"{'─' * W}\n")
                f.write(f"  A) {q.option_a}\n")
                f.write(f"  B) {q.option_b}\n")
                f.write(f"  C) {q.option_c}\n")
                f.write(f"  D) {q.option_d}\n\n")
                f.write(f"  ✅ To'g'ri javob : {q.correct}\n")
                f.write(f"  💡 Izoh          : {q.explanation}\n\n")

    def _write_quiz_json(self, result: PipelineResult, path: Path):
        """
        Quiz ilovasi uchun strukturalangan JSON.
        Bu format frontend (React, Vue) yoki mobile app da
        to'g'ridan-to'g'ri ishlatilishi mumkin.
        """
        quiz = {
            "title"     : f"{result.pdf_file} — Quiz",
            "created_at": time.strftime("%Y-%m-%d"),
            "questions" : [
                {
                    "id"         : i + 1,
                    "question"   : q.question,
                    "options"    : {
                        "A": q.option_a,
                        "B": q.option_b,
                        "C": q.option_c,
                        "D": q.option_d
                    },
                    "answer"     : q.correct,
                    "explanation": q.explanation
                }
                for i, q in enumerate(result.test_questions)
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(quiz, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════
#  ASOSIY PIPELINE ORKESTRATOR
#  Barcha qadamlarni tartibda ishga tushiradi.
#  API server va CLI har ikkisi bu sinfni ishlatadi.
# ══════════════════════════════════════════════════════════════

class PDFPipeline:
    """
    To'liq PDF → Natija pipeline orkestrator sinfi.

    Bu sinf barcha komponentlarni bog'laydi:
      PDFExtractor → TextChunker → VectorDB → OllamaLLM
      → StructuredOutputGenerator → ResultSaver

    Ishlatilish (CLI):
        p = PDFPipeline()
        result = p.run("kitob.pdf", question_count=10, difficulty="hard")

    Ishlatilish (API):
        p = PDFPipeline()
        book_id = p.ingest("kitob.pdf")       # PDF ni yuklash
        qa = p.ask(book_id, "savol?")         # Savolga javob
        qs = p.generate_qa(book_id, 10)       # Savollar generatsiya
    """

    def __init__(self, cfg: Config = None):
        self.cfg     = cfg or Config()
        self.chunker = TextChunker(self.cfg.CHUNK_SIZE, self.cfg.CHUNK_OVERLAP)
        self.saver   = ResultSaver(self.cfg.OUTPUT_DIR)

        # Papkalarni yaratish
        Path(self.cfg.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.DB_DIR).mkdir(parents=True, exist_ok=True)

    def _get_db(self, book_id: str) -> VectorDB:
        """Har kitob uchun alohida ChromaDB kolleksiyasi"""
        return VectorDB(
            db_path        = self.cfg.DB_DIR,
            collection_name= f"book_{book_id}"  # Kitob ID bo'yicha alohida
        )

    def _get_llm(self) -> OllamaLLM:
        """LLM ob'ektini yaratadi"""
        return OllamaLLM(
            model   = self.cfg.OLLAMA_MODEL,
            base_url= self.cfg.OLLAMA_BASE,
            timeout = self.cfg.OLLAMA_TIMEOUT
        )

    def ingest(self, pdf_path: str, force_reload: bool = False) -> str:
        """
        PDF ni o'qib, bo'laklaydi va vektor bazasiga saqlaydi.
        Keyingi so'rovlar uchun book_id qaytaradi.

        Bu metod bir marta chaqiriladi (kitob yangilanmasa).
        force_reload=True bo'lsa bazani qayta yaratadi.

        Args:
            pdf_path    : PDF fayl yo'li
            force_reload: True bo'lsa bazani tozalab qayta yaratadi

        Returns:
            str: book_id (PDF nomi asosida MD5 hash)
        """
        # book_id: PDF fayl nomidan deterministik hash
        # Bir xil kitob uchun har doim bir xil ID
        book_id = hashlib.md5(Path(pdf_path).name.encode()).hexdigest()[:12]

        logger.info(f"Ingesting: {pdf_path} → book_id={book_id}")

        # PDF dan matn ajratish
        extractor = PDFExtractor(pdf_path)
        pages     = extractor.extract()

        # Matnni bo'laklash
        chunks = self.chunker.chunk_pages(pages, Path(pdf_path).name)

        # Vektor bazasiga qo'shish
        db = self._get_db(book_id)
        if force_reload:
            db.delete_collection()
            db = self._get_db(book_id)  # Yangi kolleksiya

        added = db.add_chunks(chunks)
        logger.info(f"book_id={book_id}: {added} yangi bo'lak qo'shildi")

        # Bo'laklarni CSV ga saqlash (ixtiyoriy, debug uchun)
        self.saver.save_chunks_csv(chunks, Path(pdf_path).name)

        return book_id

    def ask(self, book_id: str, question: str) -> AnswerResult:
        """
        Yuklangan kitobdan savolga javob beradi.

        Args:
            book_id : ingest() dan qaytgan ID
            question: Foydalanuvchi savoli

        Returns:
            AnswerResult: Javob, ishonch darajasi, sahifalar
        """
        db  = self._get_db(book_id)
        llm = self._get_llm()
        gen = StructuredOutputGenerator(llm, db)
        return gen.answer(question)

    def generate_qa(
        self,
        book_id   : str,
        count     : int = 10,
        difficulty: str = "hard"
    ) -> List[QAPair]:
        """
        Kitobdan savol-javoblar generatsiya qiladi.

        Args:
            book_id   : ingest() dan qaytgan ID
            count     : Nechta savol
            difficulty: "easy" | "medium" | "hard"

        Returns:
            List[QAPair]
        """
        db  = self._get_db(book_id)
        llm = self._get_llm()
        gen = StructuredOutputGenerator(llm, db)
        return gen.generate_qa_pairs(count, difficulty)

    def generate_mcq(self, book_id: str, count: int = 5) -> List[MCQQuestion]:
        """
        Kitobdan MCQ test savollar generatsiya qiladi.

        Args:
            book_id: ingest() dan qaytgan ID
            count  : Nechta test savoli

        Returns:
            List[MCQQuestion]
        """
        db  = self._get_db(book_id)
        llm = self._get_llm()
        gen = StructuredOutputGenerator(llm, db)
        return gen.generate_mcq(count)

    def get_book_info(self, book_id: str, filename: str) -> BookInfo:
        """Kitob haqida metadata qaytaradi"""
        db = self._get_db(book_id)
        return BookInfo(
            book_id     = book_id,
            filename    = filename,
            total_pages = 0,       # Kengaytirilishi mumkin
            total_chunks= db.count(),
            upload_time = time.strftime("%Y-%m-%d %H:%M:%S"),
            status      = "ready" if db.count() > 0 else "empty"
        )

    def run_full(
        self,
        pdf_path      : str,
        question_count: int = None,
        test_count    : int = None,
        difficulty    : str = "hard",
        force_reload  : bool = False
    ) -> PipelineResult:
        """
        CLI uchun to'liq pipeline: yuklash + generatsiya + saqlash.

        Args:
            pdf_path      : PDF fayl yo'li
            question_count: Savol-javoblar soni
            test_count    : MCQ testlar soni
            difficulty    : Qiyinlik darajasi
            force_reload  : Bazani qayta yaratish

        Returns:
            PipelineResult: Barcha natijalar + saqlangan fayllar
        """
        t0      = time.time()
        q_count = question_count or self.cfg.DEFAULT_QA_COUNT
        t_count = test_count     or self.cfg.DEFAULT_MCQ_COUNT

        # 1-2. Extract + Chunk + Ingest
        book_id = self.ingest(pdf_path, force_reload)

        # 3. Generatsiya
        llm = self._get_llm()
        db  = self._get_db(book_id)
        gen = StructuredOutputGenerator(llm, db)

        qa_list  = gen.generate_qa_pairs(q_count, difficulty)
        mcq_list = gen.generate_mcq(t_count)

        # 4. Natijalarni saqlash
        result = PipelineResult(
            pdf_file        = Path(pdf_path).name,
            total_pages     = 0,
            total_chunks    = db.count(),
            questions       = qa_list,
            test_questions  = mcq_list,
            processing_time = time.time() - t0
        )
        result.saved_files = self.saver.save_all(result)

        return result