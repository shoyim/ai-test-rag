"""
api.py
═══════════════════════════════════════════════════════════════
FastAPI asosidagi REST API server.

Endpointlar:
  POST /books/upload          — PDF yuklash va qayta ishlash
  GET  /books                 — Barcha yuklangan kitoblar
  GET  /books/{book_id}       — Kitob ma'lumotlari
  DELETE /books/{book_id}     — Kitobni o'chirish

  POST /books/{book_id}/ask          — Savolga javob
  POST /books/{book_id}/generate/qa  — Savol-javoblar generatsiya
  POST /books/{book_id}/generate/mcq — MCQ test generatsiya

  GET  /health                — Server holati
  GET  /docs                  — Swagger UI (avtomatik)

Ishga tushirish:
  pip install fastapi uvicorn python-multipart
  uvicorn api:app --reload --port 8000

So'rov misoli (curl):
  curl -X POST http://localhost:8000/books/upload \
       -F "file=@kitob.pdf"
═══════════════════════════════════════════════════════════════
"""

import os
import json
import time
import shutil
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import asdict

# ── FastAPI ───────────────────────────────────────────────────
# pip install fastapi uvicorn python-multipart
from fastapi import (
    FastAPI, File, UploadFile, HTTPException,
    BackgroundTasks, Query, Depends
)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Bizning pipeline ──────────────────────────────────────────
from pipeline import (
    Config, PDFPipeline,
    QAPair, MCQQuestion, AnswerResult, BookInfo,
    PipelineResult
)


# ════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt= "%H:%M:%S"
)
logger = logging.getLogger("api")


# ════════════════════════════════════════════════════════════
#  GLOBAL HOLAT
#  books_registry: {book_id: {filename, upload_time, status}}
#  Ishlab chiqarish uchun bu Redis yoki PostgreSQL ga ko'chirilishi kerak.
#  Hozir — oddiy dict (server to'xtaganda o'chadi).
# ════════════════════════════════════════════════════════════
books_registry: Dict[str, dict] = {}

# JSON faylda saqlash (server qayta ishlaganda ham qoladi)
REGISTRY_FILE = "books_registry.json"

def _load_registry():
    """Oldingi sessiyadan ro'yxatni yuklaydi"""
    global books_registry
    if Path(REGISTRY_FILE).exists():
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                books_registry = json.load(f)
            logger.info(f"Registry yuklandi: {len(books_registry)} kitob")
        except Exception as e:
            logger.warning(f"Registry yuklashda xato: {e}")

def _save_registry():
    """Ro'yxatni JSON faylga saqlaydi"""
    try:
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(books_registry, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Registry saqlashda xato: {e}")


# ════════════════════════════════════════════════════════════
#  PYDANTIC MODELLARI (Request/Response sxemalari)
#  FastAPI Swagger UI uchun avtomatik hujjatlashtiradi.
#  Pydantic validatsiya qiladi: noto'g'ri tip → 422 xato.
# ════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    """POST /books/{book_id}/ask uchun so'rov tanasi"""
    question: str = Field(
        ...,
        min_length = 3,
        max_length = 500,
        description= "Kitobga berilayotgan savol",
        example    = "Ushbu kitobning asosiy g'oyasi nima?"
    )


class GenerateQARequest(BaseModel):
    """POST /books/{book_id}/generate/qa uchun so'rov tanasi"""
    count     : int = Field(
        default    = 10,
        ge         = 1,    # greater or equal 1
        le         = 50,   # less or equal 50
        description= "Nechta savol-javob generatsiya qilish"
    )
    difficulty: str = Field(
        default    = "hard",
        description= "Qiyinlik darajasi: easy | medium | hard",
        example    = "hard"
    )

    # Pydantic v2 validator
    def model_post_init(self, __context):
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError("difficulty 'easy', 'medium' yoki 'hard' bo'lishi kerak")


class GenerateMCQRequest(BaseModel):
    """POST /books/{book_id}/generate/mcq uchun so'rov tanasi"""
    count: int = Field(
        default    = 5,
        ge         = 1,
        le         = 30,
        description= "Nechta MCQ test savoli generatsiya qilish"
    )


# ── Response modellari ────────────────────────────────────────

class BookUploadResponse(BaseModel):
    """PDF yuklash natijasi"""
    book_id     : str
    filename    : str
    total_chunks: int
    message     : str


class BookListItem(BaseModel):
    """Kitoblar ro'yxatidagi bitta element"""
    book_id    : str
    filename   : str
    upload_time: str
    status     : str
    chunk_count: int


class AskResponse(BaseModel):
    """Savol-javob natijasi"""
    question    : str
    answer      : str
    confidence  : str
    key_points  : List[str]
    source_pages: List[int]


class QAPairResponse(BaseModel):
    """Bitta savol-javob juftligi"""
    question    : str
    answer      : str
    difficulty  : str
    source_pages: List[int]


class MCQResponse(BaseModel):
    """Bitta MCQ savoli"""
    question    : str
    option_a    : str
    option_b    : str
    option_c    : str
    option_d    : str
    correct     : str
    explanation : str
    source_pages: List[int]


class GenerateQAResponse(BaseModel):
    """Savol-javoblar generatsiya natijasi"""
    book_id   : str
    count     : int
    difficulty: str
    questions : List[QAPairResponse]


class GenerateMCQResponse(BaseModel):
    """MCQ testlar generatsiya natijasi"""
    book_id  : str
    count    : int
    questions: List[MCQResponse]


class HealthResponse(BaseModel):
    """Server holati"""
    status     : str
    ollama     : str
    books_count: int
    version    : str


# ════════════════════════════════════════════════════════════
#  FASTAPI ILOVASI
# ════════════════════════════════════════════════════════════

app = FastAPI(
    title      = "PDF QA API",
    description= """
## PDF Kitob Tahlil va Savol-Javob API

Bu API PDF kitoblarni yuklaydi, matnini tahlil qiladi va
Ollama (local LLM) yordamida savol-javob va testlar yaratadi.

### Asosiy jarayon:
1. **PDF yuklash**: `POST /books/upload`
2. **Savolga javob**: `POST /books/{book_id}/ask`
3. **Savol generatsiya**: `POST /books/{book_id}/generate/qa`
4. **Test generatsiya**: `POST /books/{book_id}/generate/mcq`

### Barcha vositalar bepul:
- PyMuPDF (PDF o'qish)
- sentence-transformers (embedding)
- ChromaDB (vektor baza)
- Ollama + Llama3 (LLM)
    """,
    version    = "1.0.0",
    docs_url   = "/docs",     # Swagger UI
    redoc_url  = "/redoc"     # ReDoc
)

# CORS — browser dan so'rov qilishga ruxsat
# Ishlab chiqarishda allow_origins ni aniq domenga o'zgartiring
app.add_middleware(
    CORSMiddleware,
    allow_origins    = ["*"],         # Barcha originlarga ruxsat (dev uchun)
    allow_credentials= True,
    allow_methods    = ["*"],
    allow_headers    = ["*"],
)

# ── Global pipeline ob'ekti ───────────────────────────────────
# Bir marta yaratiladi, barcha endpointlar ishlatadi
cfg      = Config()
pipeline = PDFPipeline(cfg)


# ── Startup/Shutdown event ────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Server ishga tushganda chaqiriladi"""
    _load_registry()
    logger.info("═" * 55)
    logger.info("  PDF QA API Server ishga tushdi")
    logger.info(f"  Ollama model : {cfg.OLLAMA_MODEL}")
    logger.info(f"  Upload papka : {cfg.UPLOAD_DIR}")
    logger.info(f"  Output papka : {cfg.OUTPUT_DIR}")
    logger.info(f"  ChromaDB     : {cfg.DB_DIR}")
    logger.info("  Swagger UI   : http://localhost:8000/docs")
    logger.info("═" * 55)


@app.on_event("shutdown")
async def shutdown_event():
    """Server to'xtaganda chaqiriladi"""
    _save_registry()
    logger.info("API server to'xtatildi")


# ════════════════════════════════════════════════════════════
#  YORDAMCHI FUNKSIYALAR
# ════════════════════════════════════════════════════════════

def _get_book_or_404(book_id: str) -> dict:
    """
    Kitobni ro'yxatdan qidiradi, topilmasa 404 xato chiqaradi.
    Bu funksiya barcha endpoint'larda ishlatiladi.
    """
    if book_id not in books_registry:
        raise HTTPException(
            status_code = 404,
            detail      = f"Kitob topilmadi: '{book_id}'. "
                          f"Avval /books/upload orqali yuklang."
        )
    return books_registry[book_id]


def _success(data: Any, message: str = "OK") -> dict:
    """Standart muvaffaqiyatli javob formati"""
    return {
        "success"  : True,
        "message"  : message,
        "data"     : data,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


# ════════════════════════════════════════════════════════════
#  ENDPOINTLAR
# ════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────
#  HEALTH CHECK
# ──────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model= HealthResponse,
    summary       = "Server holati",
    tags          = ["System"]
)
async def health_check():
    """
    Server va Ollama holati tekshiradi.

    Returns:
        status : "ok" yoki "degraded"
        ollama : "connected" yoki "disconnected"
    """
    import requests as req

    # Ollama server ishlayotganini tekshirish
    ollama_status = "disconnected"
    try:
        r = req.get(f"{cfg.OLLAMA_BASE}/api/tags", timeout=3)
        if r.status_code == 200:
            ollama_status = "connected"
    except Exception:
        pass

    overall = "ok" if ollama_status == "connected" else "degraded"

    return HealthResponse(
        status     = overall,
        ollama     = ollama_status,
        books_count= len(books_registry),
        version    = "1.0.0"
    )


# ──────────────────────────────────────────────────────────
#  KITOB YUKLASH
# ──────────────────────────────────────────────────────────

@app.post(
    "/books/upload",
    response_model = BookUploadResponse,
    status_code    = 201,
    summary        = "PDF kitob yuklash",
    tags           = ["Books"],
    responses      = {
        201: {"description": "Kitob muvaffaqiyatli yuklandi"},
        400: {"description": "Noto'g'ri fayl formati"},
        500: {"description": "Server xatosi"}
    }
)
async def upload_book(
    file        : UploadFile = File(..., description="PDF fayl"),
    force_reload: bool       = Query(False, description="Bazani qayta yaratish")
):
    """
    PDF kitobni yuklaydi va vektor bazasiga saqlaydi.

    **Jarayon:**
    1. PDF fayl serverga saqlanadi
    2. PyMuPDF bilan matn ajratiladi
    3. Matn bo'laklarga ajratiladi
    4. sentence-transformers bilan vektorlashtiriladi
    5. ChromaDB ga saqlanadi
    6. `book_id` qaytariladi (keyingi so'rovlarda ishlatiladi)

    **force_reload=true** bo'lsa: eski bazani o'chirib qayta yaratadi
    (kitob yangilanganda foydali)
    """
    # ── 1. Fayl validatsiyasi ─────────────────────────────
    if not file.filename:
        raise HTTPException(400, "Fayl nomi bo'sh")

    filename = Path(file.filename).name  # Path traversal xavfsizligi
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            400,
            f"Faqat PDF formatda fayl qabul qilinadi. "
            f"Kelgan fayl: '{filename}'"
        )

    # ── 2. Faylni saqlash ────────────────────────────────
    upload_path = Path(cfg.UPLOAD_DIR) / filename
    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"Fayl saqlandi: {upload_path} ({len(content):,} byte)")
    except OSError as e:
        raise HTTPException(500, f"Fayl saqlashda xato: {e}")

    # ── 3. Pipeline: extract + chunk + embed + store ─────
    try:
        book_id = pipeline.ingest(str(upload_path), force_reload=force_reload)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ImportError as e:
        raise HTTPException(500, f"Kutubxona o'rnatilmagan: {e}")
    except Exception as e:
        logger.error(f"Ingest xato: {e}", exc_info=True)
        raise HTTPException(500, f"PDF qayta ishlashda xato: {e}")

    # ── 4. Registry ni yangilash ─────────────────────────
    chunk_count = pipeline._get_db(book_id).count()
    books_registry[book_id] = {
        "filename"   : filename,
        "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status"     : "ready",
        "chunk_count": chunk_count,
        "file_path"  : str(upload_path)
    }
    _save_registry()

    logger.info(f"Kitob yuklandi: {filename} → book_id={book_id} ({chunk_count} bo'lak)")

    return BookUploadResponse(
        book_id     = book_id,
        filename    = filename,
        total_chunks= chunk_count,
        message     = f"'{filename}' muvaffaqiyatli yuklandi. "
                      f"Jami {chunk_count} bo'lak yaratildi."
    )


# ──────────────────────────────────────────────────────────
#  KITOBLAR RO'YXATI VA MA'LUMOTLARI
# ──────────────────────────────────────────────────────────

@app.get(
    "/books",
    summary= "Yuklangan kitoblar ro'yxati",
    tags   = ["Books"]
)
async def list_books():
    """
    Serverga yuklangan barcha kitoblarni qaytaradi.
    Har kitob uchun: book_id, fayl nomi, holat, bo'laklar soni.
    """
    books = []
    for bid, info in books_registry.items():
        books.append(BookListItem(
            book_id    = bid,
            filename   = info["filename"],
            upload_time= info["upload_time"],
            status     = info["status"],
            chunk_count= info.get("chunk_count", 0)
        ))

    return _success(
        data   = [b.model_dump() for b in books],
        message= f"Jami {len(books)} ta kitob"
    )


@app.get(
    "/books/{book_id}",
    summary= "Kitob ma'lumotlari",
    tags   = ["Books"]
)
async def get_book(book_id: str):
    """
    Bitta kitob haqida to'liq ma'lumot qaytaradi.

    Path params:
        book_id: /books/upload dan qaytgan ID
    """
    info = _get_book_or_404(book_id)

    return _success(data={
        "book_id"    : book_id,
        "filename"   : info["filename"],
        "upload_time": info["upload_time"],
        "status"     : info["status"],
        "chunk_count": info.get("chunk_count", 0)
    })


@app.delete(
    "/books/{book_id}",
    summary= "Kitobni o'chirish",
    tags   = ["Books"]
)
async def delete_book(book_id: str):
    """
    Kitobni ro'yxatdan, vektor bazasidan va serverdan o'chiradi.

    **Diqqat**: O'chirilgan kitobga so'rov yuborish mumkin bo'lmaydi.
    """
    info = _get_book_or_404(book_id)

    # Vektor kolleksiyasini o'chirish
    try:
        db = pipeline._get_db(book_id)
        db.delete_collection()
    except Exception as e:
        logger.warning(f"Kolleksiya o'chirishda xato: {e}")

    # PDF faylni o'chirish (ixtiyoriy)
    file_path = info.get("file_path")
    if file_path and Path(file_path).exists():
        try:
            Path(file_path).unlink()
        except OSError as e:
            logger.warning(f"Fayl o'chirishda xato: {e}")

    # Registry dan o'chirish
    del books_registry[book_id]
    _save_registry()

    logger.info(f"Kitob o'chirildi: {book_id} ({info['filename']})")
    return _success(data={"book_id": book_id}, message="Kitob o'chirildi")


# ──────────────────────────────────────────────────────────
#  SAVOL-JAVOB (RAG)
# ──────────────────────────────────────────────────────────

@app.post(
    "/books/{book_id}/ask",
    response_model= AskResponse,
    summary       = "Kitobdan savolga javob olish",
    tags          = ["Q&A"]
)
async def ask_question(book_id: str, req: AskRequest):
    """
    Yuklangan kitob asosida savolga javob beradi.

    **RAG jarayoni:**
    1. Savol vektorlashtiriladi
    2. ChromaDB dan eng mos `top_k` bo'lak topiladi
    3. Bo'laklar + savol Ollama LLM ga beriladi
    4. LLM faqat kitob materialidan javob yozadi

    **Javob tarkibi:**
    - `answer`: To'liq javob matni
    - `confidence`: "high" | "medium" | "low"
    - `key_points`: Asosiy fikrlar ro'yxati
    - `source_pages`: Javob topilgan sahifalar

    **Misol so'rov:**
    ```json
    {"question": "Ushbu kitobning asosiy mavzusi nima?"}
    ```
    """
    _get_book_or_404(book_id)

    try:
        result: AnswerResult = pipeline.ask(book_id, req.question)
    except ConnectionError as e:
        raise HTTPException(503, f"Ollama server ulana olmadi: {e}")
    except TimeoutError as e:
        raise HTTPException(504, f"LLM javob bermadi: {e}")
    except Exception as e:
        logger.error(f"ask xato: {e}", exc_info=True)
        raise HTTPException(500, f"Javob generatsiyada xato: {e}")

    return AskResponse(
        question    = result.question,
        answer      = result.answer,
        confidence  = result.confidence,
        key_points  = result.key_points,
        source_pages= result.source_pages
    )


# ──────────────────────────────────────────────────────────
#  SAVOL GENERATSIYA
# ──────────────────────────────────────────────────────────

@app.post(
    "/books/{book_id}/generate/qa",
    response_model= GenerateQAResponse,
    summary       = "Savol-javoblar generatsiya qilish",
    tags          = ["Generate"]
)
async def generate_qa(book_id: str, req: GenerateQARequest):
    """
    Kitob materialidan ochiq uчlu savol-javoblar generatsiya qiladi.

    **Qiyinlik darajalari:**
    - `easy`  : Faktik, yodlab qolishga asoslanган
    - `medium`: Tushunish va tushuntirish
    - `hard`  : Tahlil, sintez, tanqidiy fikrlash

    **Misol so'rov:**
    ```json
    {"count": 10, "difficulty": "hard"}
    ```

    **Eslatma:** Ko'p savol so'ralganda LLM ko'proq vaqt ketishi mumkin.
    """
    _get_book_or_404(book_id)

    if req.difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(
            422,
            "difficulty qiymati: 'easy', 'medium' yoki 'hard' bo'lishi kerak"
        )

    try:
        qa_list = pipeline.generate_qa(book_id, req.count, req.difficulty)
    except ConnectionError as e:
        raise HTTPException(503, f"Ollama ulana olmadi: {e}")
    except TimeoutError as e:
        raise HTTPException(504, f"LLM javob bermadi: {e}")
    except Exception as e:
        logger.error(f"generate_qa xato: {e}", exc_info=True)
        raise HTTPException(500, f"Generatsiya xatosi: {e}")

    return GenerateQAResponse(
        book_id   = book_id,
        count     = len(qa_list),
        difficulty= req.difficulty,
        questions = [
            QAPairResponse(**asdict(q)) for q in qa_list
        ]
    )


# ──────────────────────────────────────────────────────────
#  MCQ TEST GENERATSIYA
# ──────────────────────────────────────────────────────────

@app.post(
    "/books/{book_id}/generate/mcq",
    response_model= GenerateMCQResponse,
    summary       = "Ko'p javobli test (MCQ) generatsiya",
    tags          = ["Generate"]
)
async def generate_mcq(book_id: str, req: GenerateMCQRequest):
    """
    Kitob materialidan to'rt variantli MCQ test savollar yaratadi.

    **Har savol tarkibi:**
    - Savol matni
    - A, B, C, D variantlari
    - To'g'ri javob (`correct`: "A"/"B"/"C"/"D")
    - Izoh (nima uchun bu javob to'g'ri)

    **Misol so'rov:**
    ```json
    {"count": 5}
    ```
    """
    _get_book_or_404(book_id)

    try:
        mcq_list = pipeline.generate_mcq(book_id, req.count)
    except ConnectionError as e:
        raise HTTPException(503, f"Ollama ulana olmadi: {e}")
    except TimeoutError as e:
        raise HTTPException(504, f"LLM javob bermadi: {e}")
    except Exception as e:
        logger.error(f"generate_mcq xato: {e}", exc_info=True)
        raise HTTPException(500, f"Generatsiya xatosi: {e}")

    return GenerateMCQResponse(
        book_id  = book_id,
        count    = len(mcq_list),
        questions= [MCQResponse(**asdict(q)) for q in mcq_list]
    )


# ──────────────────────────────────────────────────────────
#  TO'LIQ PIPELINE (BATCH)
# ──────────────────────────────────────────────────────────

@app.post(
    "/books/{book_id}/process",
    summary= "To'liq generatsiya (QA + MCQ bir vaqtda)",
    tags   = ["Generate"]
)
async def process_book(
    book_id       : str,
    question_count: int = Query(10,    ge=1, le=50,  description="Savol-javoblar soni"),
    test_count    : int = Query(5,     ge=1, le=20,  description="MCQ testlar soni"),
    difficulty    : str = Query("hard",              description="easy | medium | hard"),
    save_files    : bool= Query(True,                description="Natijalarni faylga saqlash")
):
    """
    Bir so'rov bilan ham QA, ham MCQ generatsiya qiladi.
    Natijalarni faylga ham saqlaydi (save_files=true bo'lsa).

    Katta kitoblar uchun uzoq vaqt ketishi mumkin.
    Background task sifatida ishlash tavsiya etiladi (kengaytirilishi mumkin).
    """
    info = _get_book_or_404(book_id)

    if difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(422, "difficulty: 'easy'|'medium'|'hard'")

    try:
        # QA generatsiya
        qa_list  = pipeline.generate_qa(book_id, question_count, difficulty)
        # MCQ generatsiya
        mcq_list = pipeline.generate_mcq(book_id, test_count)
    except ConnectionError as e:
        raise HTTPException(503, str(e))
    except TimeoutError as e:
        raise HTTPException(504, str(e))
    except Exception as e:
        logger.error(f"process_book xato: {e}", exc_info=True)
        raise HTTPException(500, str(e))

    # Faylga saqlash (ixtiyoriy)
    saved_files = {}
    if save_files:
        from pipeline import PipelineResult, ResultSaver
        result = PipelineResult(
            pdf_file       = info["filename"],
            total_pages    = 0,
            total_chunks   = info.get("chunk_count", 0),
            questions      = qa_list,
            test_questions = mcq_list,
            processing_time= 0.0
        )
        saver       = ResultSaver(cfg.OUTPUT_DIR)
        saved_files = saver.save_all(result)

    return _success(
        data={
            "book_id"    : book_id,
            "qa_count"   : len(qa_list),
            "mcq_count"  : len(mcq_list),
            "difficulty" : difficulty,
            "questions"  : [asdict(q) for q in qa_list],
            "mcq"        : [asdict(q) for q in mcq_list],
            "saved_files": saved_files
        },
        message= f"{len(qa_list)} ta QA va {len(mcq_list)} ta MCQ yaratildi"
    )


# ──────────────────────────────────────────────────────────
#  NATIJA FAYLLARINI YUKLASH
# ──────────────────────────────────────────────────────────

@app.get(
    "/files/{filename}",
    summary= "Natija faylini yuklash",
    tags   = ["Files"]
)
async def download_file(filename: str):
    """
    output/ papkasidagi fayllarni yuklash uchun.
    process endpointidan qaytgan `saved_files` yo'llarini ishlatish mumkin.

    Misol: GET /files/kitob_20241215_qa.txt
    """
    # Path traversal xavfsizligi: faqat output papkasidan
    safe_name = Path(filename).name   # "../../../etc/passwd" → "passwd"
    file_path = Path(cfg.OUTPUT_DIR) / safe_name

    if not file_path.exists():
        raise HTTPException(404, f"Fayl topilmadi: '{safe_name}'")

    # Fayl turini aniqlash
    suffix = file_path.suffix.lower()
    media_type_map = {
        ".json": "application/json",
        ".txt" : "text/plain; charset=utf-8",
        ".csv" : "text/csv; charset=utf-8",
        ".pdf" : "application/pdf"
    }
    media_type = media_type_map.get(suffix, "application/octet-stream")

    return FileResponse(
        path      = str(file_path),
        media_type= media_type,
        filename  = safe_name
    )


@app.get(
    "/files",
    summary= "Output papkasidagi fayllar ro'yxati",
    tags   = ["Files"]
)
async def list_output_files():
    """
    output/ papkasidagi barcha natija fayllarini ko'rsatadi.
    """
    out_dir = Path(cfg.OUTPUT_DIR)
    if not out_dir.exists():
        return _success(data=[], message="Output papkasi bo'sh")

    files = []
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            files.append({
                "name"       : f.name,
                "size_bytes" : f.stat().st_size,
                "download_url": f"/files/{f.name}"
            })

    return _success(data=files, message=f"{len(files)} ta fayl")


# ══════════════════════════════════════════════════════════════
#  ISHGA TUSHIRISH
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    To'g'ridan-to'g'ri ishga tushirish:
        python api.py

    Yoki uvicorn bilan:
        uvicorn api:app --reload --port 8000

    Production uchun:
        uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
    """
    import uvicorn

    uvicorn.run(
        "api:app",
        host    = "0.0.0.0",  # Barcha interfeyslarda tinglash
        port    = 8000,
        reload  = True,        # Kod o'zgarganda avtomatik qayta yuklash (dev)
        log_level= "info"
    )