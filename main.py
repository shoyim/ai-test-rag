"""
╔══════════════════════════════════════════════════════════════╗
║        PDF → CHUNKING → LLM → STRUCTURED OUTPUT → SAVE      ║
║                    100% BEPUL VOSITALAR                      ║
╠══════════════════════════════════════════════════════════════╣
║  PDF o'qish  : PyMuPDF (fitz) — tekin                       ║
║  Chunking    : Qo'lda yozilgan — tekin                       ║
║  Embedding   : sentence-transformers — tekin                  ║
║  Vektor DB   : ChromaDB — tekin                              ║
║  LLM         : Ollama (Llama 3) — tekin, local               ║
║  Output      : JSON / TXT — tekin                            ║
╚══════════════════════════════════════════════════════════════╝

O'rnatish:
    pip install pymupdf chromadb sentence-transformers requests

Ollama o'rnatish (Linux/Mac):
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3

Ishga tushirish:
    python pdf_pipeline.py
"""

import os
import re
import json
import time
import hashlib
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional


# ════════════════════════════════════════════════════════════
#  SOZLAMALAR
# ════════════════════════════════════════════════════════════

class Config:
    PDF_PATH        = "books/Matematika (1996-2003) yillar.pdf"       # PDF fayl nomi
    OUTPUT_DIR      = "output"          # Natijalar papkasi
    DB_DIR          = "chroma_db"       # Vektor baza papkasi

    # Chunking
    CHUNK_SIZE      = 800               # Belgilar soni (taxminan 150-200 so'z)
    CHUNK_OVERLAP   = 150               # Qo'shni bo'laklar orasidagi takror

    # Ollama (local LLM)
    OLLAMA_URL      = "http://localhost:11434/api/generate"
    OLLAMA_MODEL    = "llama3"          # yoki "mistral", "gemma2" va h.k.
    OLLAMA_TIMEOUT  = 120               # Sekund

    # Qidiruv
    TOP_K           = 4                 # Nechta eng yaqin bo'lak ishlatilsin

    # Natija turlari
    QUESTION_COUNT  = 10
    TEST_COUNT      = 5


# ════════════════════════════════════════════════════════════
#  DATACLASS — Tuzilgan chiqish formatlari
# ════════════════════════════════════════════════════════════

@dataclass
class TextChunk:
    """Matnning bir bo'lagi"""
    chunk_id   : str
    page_num   : int
    text       : str
    char_count : int
    source_pdf : str


@dataclass
class QAPair:
    """Savol-javob juftligi"""
    question   : str
    answer     : str
    difficulty : str
    source_pages: List[int]


@dataclass
class MCQQuestion:
    """Ko'p javobli test savoli"""
    question   : str
    option_a   : str
    option_b   : str
    option_c   : str
    option_d   : str
    correct    : str          # "A", "B", "C" yoki "D"
    explanation: str
    source_pages: List[int]


@dataclass
class PipelineResult:
    """Butun pipeline natijasi"""
    pdf_file        : str
    total_pages     : int
    total_chunks    : int
    questions       : List[QAPair]
    test_questions  : List[MCQQuestion]
    processing_time : float


# ════════════════════════════════════════════════════════════
#  1-QADAM: PDF MATN AJRATISH (Text Extraction)
# ════════════════════════════════════════════════════════════

class PDFExtractor:
    """
    PyMuPDF (fitz) yordamida PDF dan matn ajratadi.
    
    PyMuPDF — ochiq kodli, bepul, juda tez ishlaydi.
    Har bir sahifadan toza matn, sahifa raqami va metadata olinadi.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self._check_file()

    def _check_file(self):
        if not Path(self.pdf_path).exists():
            raise FileNotFoundError(
                f"❌ PDF topilmadi: '{self.pdf_path}'\n"
                f"   Faylni shu papkaga qo'ying: {Path('.').absolute()}"
            )

    def extract(self) -> List[dict]:
        """
        Har bir sahifadan {'page': N, 'text': '...'} ro'yxatini qaytaradi.
        
        Returns:
            List[dict]: Sahifalar ro'yxati
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF o'rnatilmagan!\n"
                "O'rnatish: pip install pymupdf"
            )

        print(f"\n{'─'*55}")
        print(f"  1-QADAM: MATN AJRATISH")
        print(f"{'─'*55}")
        print(f"  📄 Fayl: {self.pdf_path}")

        pages_data = []

        with fitz.open(self.pdf_path) as doc:
            total = len(doc)
            print(f"  📑 Jami sahifalar: {total}")

            for page_num in range(total):
                page = doc[page_num]
                
                # get_text("text") — toza matn, tartibli chiqaradi
                raw_text = page.get_text("text")
                
                # Keraksiz bo'shliqlarni tozalash
                clean_text = self._clean_text(raw_text)

                if clean_text:
                    pages_data.append({
                        "page"      : page_num + 1,   # 1 dan boshlaymiz
                        "text"      : clean_text,
                        "char_count": len(clean_text)
                    })

                # Progress ko'rsatish
                if (page_num + 1) % 10 == 0 or page_num == total - 1:
                    print(f"  ⏳ {page_num + 1}/{total} sahifa o'qildi...", end="\r")

        total_chars = sum(p["char_count"] for p in pages_data)
        print(f"\n  ✅ {len(pages_data)} sahifa, {total_chars:,} belgi ajratildi")

        return pages_data

    def _clean_text(self, text: str) -> str:
        """Matnni tozalaydi: ortiqcha bo'shliq, sahifa raqamlari va h.k."""
        if not text:
            return ""

        # Ko'p bo'sh qatorlarni bittaga aylantirish
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Satr boshidagi / oxiridagi bo'shliqlar
        lines = [line.strip() for line in text.splitlines()]

        # Faqat raqamdan iborat qatorlar (sahifa raqamlari) — olib tashlash
        lines = [l for l in lines if not re.fullmatch(r'\d{1,4}', l)]

        # Bo'sh qatorlarni saqlab birlashtirish
        text = "\n".join(lines)

        return text.strip()


# ════════════════════════════════════════════════════════════
#  2-QADAM: CHUNKING (Matnni Bo'laklash)
# ════════════════════════════════════════════════════════════

class TextChunker:
    """
    Matnni semantik bo'laklarga ajratadi.
    
    Strategiya:
      1. Paragraflardan bo'linadi (\\n\\n)
      2. Agar paragraf CHUNK_SIZE dan katta bo'lsa — jumlalardan bo'linadi
      3. Overlap: har bo'lak oldingi bo'lakning so'nggi qismini ham oladi
         (bu kontekst yo'qolmasligini ta'minlaydi)
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def chunk_pages(self, pages: List[dict], source_pdf: str) -> List[TextChunk]:
        """
        Sahifalar ro'yxatidan TextChunk ob'ektlar ro'yxatini yaratadi.
        
        Args:
            pages     : PDFExtractor.extract() natijasi
            source_pdf: PDF fayl nomi (metadata uchun)
            
        Returns:
            List[TextChunk]: Bo'laklar ro'yxati
        """
        print(f"\n{'─'*55}")
        print(f"  2-QADAM: CHUNKING")
        print(f"{'─'*55}")
        print(f"  📐 Chunk hajmi: {self.chunk_size} belgi")
        print(f"  🔁 Overlap    : {self.overlap} belgi")

        all_chunks: List[TextChunk] = []

        for page in pages:
            page_chunks = self._chunk_text(
                text       = page["text"],
                page_num   = page["page"],
                source_pdf = source_pdf
            )
            all_chunks.extend(page_chunks)

        print(f"  ✅ Jami {len(all_chunks)} ta bo'lak yaratildi")
        return all_chunks

    def _chunk_text(
        self, text: str, page_num: int, source_pdf: str
    ) -> List[TextChunk]:
        """Bitta sahifa matnini bo'laklarga ajratadi"""
        
        chunks = []

        # 1. Paragraflardan bo'linadi
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        buffer     = ""
        prev_tail  = ""  # Overlap uchun oldingi bo'lakning oxiri

        for para in paragraphs:
            candidate = (prev_tail + " " + buffer + " " + para).strip()

            if len(candidate) <= self.chunk_size:
                buffer = (buffer + " " + para).strip()
            else:
                # Buferni saqlash
                if buffer:
                    chunks.append(self._make_chunk(
                        (prev_tail + " " + buffer).strip(), page_num, source_pdf
                    ))
                    # Overlap: bufferning oxirgi qismini keyingiga o'tkazish
                    prev_tail = buffer[-self.overlap:] if len(buffer) > self.overlap else buffer
                    buffer = para
                else:
                    # Paragrafning o'zi juda katta — jumlalarga bo'lish
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sent in sentences:
                        candidate2 = (prev_tail + " " + buffer + " " + sent).strip()
                        if len(candidate2) <= self.chunk_size:
                            buffer = (buffer + " " + sent).strip()
                        else:
                            if buffer:
                                chunks.append(self._make_chunk(
                                    (prev_tail + " " + buffer).strip(), page_num, source_pdf
                                ))
                                prev_tail = buffer[-self.overlap:]
                                buffer = sent
                            else:
                                # Bitta jumla ham katta — qat'iy kesish
                                for i in range(0, len(sent), self.chunk_size - self.overlap):
                                    piece = sent[i : i + self.chunk_size]
                                    if piece.strip():
                                        chunks.append(self._make_chunk(piece, page_num, source_pdf))

        # Oxirgi buffer
        if buffer.strip():
            chunks.append(self._make_chunk(
                (prev_tail + " " + buffer).strip(), page_num, source_pdf
            ))

        return chunks

    def _make_chunk(self, text: str, page_num: int, source_pdf: str) -> TextChunk:
        """TextChunk ob'ekti yaratadi"""
        # Unikal ID: fayl nomi + sahifa + matn hash
        chunk_id = hashlib.md5(
            f"{source_pdf}:{page_num}:{text[:50]}".encode()
        ).hexdigest()[:12]

        return TextChunk(
            chunk_id   = chunk_id,
            page_num   = page_num,
            text       = text.strip(),
            char_count = len(text),
            source_pdf = source_pdf
        )


# ════════════════════════════════════════════════════════════
#  3-QADAM: VEKTOR BAZASI (ChromaDB + sentence-transformers)
# ════════════════════════════════════════════════════════════

class VectorDB:
    """
    ChromaDB vektor bazasi.
    
    Embedding: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
      - Ko'p tilli (o'zbek, rus, ingliz ham ishlaydi)
      - Bepul, local, ~120MB
      - HuggingFace dan avtomatik yuklanadi
    
    ChromaDB:
      - Diskka saqlanadi (fayl o'chirilguncha qoladi)
      - Bepul, Python da to'g'ridan-to'g'ri ishlaydi
    """

    EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.collection = None
        self._setup()

    def _setup(self):
        """ChromaDB va embedding modelini yuklaydi"""
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError:
            raise ImportError(
                "ChromaDB o'rnatilmagan!\n"
                "O'rnatish: pip install chromadb sentence-transformers"
            )

        print(f"\n{'─'*55}")
        print(f"  3-QADAM: VEKTOR BAZASI")
        print(f"{'─'*55}")
        print(f"  🗄️  ChromaDB papkasi: {self.db_path}")
        print(f"  🔢 Embedding: {self.EMBED_MODEL}")

        # Embedding funksiyasi (sentence-transformers, bepul)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.EMBED_MODEL
        )

        # ChromaDB mijozi (diskka saqlash)
        client = chromadb.PersistentClient(path=self.db_path)

        # Kolleksiya (mavjud bo'lsa yuklaydi, bo'lmasa yaratadi)
        self.collection = client.get_or_create_collection(
            name="pdf_chunks",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )

        existing = self.collection.count()
        print(f"  📊 Bazada mavjud bo'laklar: {existing}")

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """
        Bo'laklarni vektor bazasiga qo'shadi.
        Allaqachon mavjud bo'laklarni qayta qo'shmaydi (chunk_id tekshiruvi).
        
        Args:
            chunks: TextChunk ro'yxati
        """
        if not chunks:
            return

        # Yangi bo'laklarni filtrlash
        existing_ids = set(self.collection.get()["ids"])
        new_chunks   = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            print(f"  ℹ️  Barcha bo'laklar allaqachon bazada mavjud")
            return

        print(f"  ➕ {len(new_chunks)} ta yangi bo'lak qo'shilmoqda...")

        # Batch qo'shish (100 tadan)
        batch_size = 100
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]

            self.collection.add(
                ids        = [c.chunk_id for c in batch],
                documents  = [c.text for c in batch],
                metadatas  = [{"page": c.page_num, "source": c.source_pdf} for c in batch]
            )
            print(f"  ⏳ {min(i + batch_size, len(new_chunks))}/{len(new_chunks)} qo'shildi...", end="\r")

        print(f"\n  ✅ Vektor bazasi tayyor! Jami: {self.collection.count()} bo'lak")

    def search(self, query: str, top_k: int = 4) -> List[dict]:
        """
        Berilgan so'rovga eng mos bo'laklarni qidiradi.
        
        Args:
            query : Qidiruv matni
            top_k : Nechta natija kerak
            
        Returns:
            List[dict]: [{'text': ..., 'page': N, 'score': X.X}, ...]
        """
        results = self.collection.query(
            query_texts = [query],
            n_results   = min(top_k, self.collection.count())
        )

        output = []
        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metadatas, distances):
            output.append({
                "text"  : doc,
                "page"  : meta.get("page", 0),
                "score" : round(1 - dist, 3)  # cosine distance → similarity
            })

        return output


# ════════════════════════════════════════════════════════════
#  4-QADAM: OLLAMA LLM (Local, Bepul)
# ════════════════════════════════════════════════════════════

class OllamaLLM:
    """
    Ollama orqali local Llama 3 (yoki boshqa) model bilan ishlaydi.
    
    Ollama:
      - Bepul, open-source
      - Internet shart emas (model bir marta yuklanadi)
      - Linux/Mac/Windows da ishlaydi
      - llama3, mistral, gemma2, phi3 va boshqa modeller
    
    O'rnatish:
      curl -fsSL https://ollama.com/install.sh | sh
      ollama pull llama3      # ~4.7GB
      ollama pull mistral     # ~4.1GB (engil variant)
    """

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model    = model
        self.api_url  = f"{base_url}/api/generate"
        self._check_connection()

    def _check_connection(self):
        """Ollama server ishlayotganini tekshiradi"""
        try:
            resp = requests.get(
                self.api_url.replace("/api/generate", "/api/tags"),
                timeout=5
            )
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                print(f"  🤖 Ollama ulandi | Model: {self.model}")
                if not any(self.model in m for m in models):
                    print(f"  ⚠️  '{self.model}' yuklanmagan!")
                    print(f"     O'rnatish: ollama pull {self.model}")
        except requests.ConnectionError:
            raise ConnectionError(
                "❌ Ollama server topilmadi!\n"
                "   Ishga tushirish: ollama serve\n"
                "   O'rnatish: https://ollama.com/download"
            )

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """
        LLM dan matn generatsiya qiladi.
        
        Args:
            prompt     : So'rov matni
            temperature: Kreativlik (0=aniq, 1=kreativ)
            
        Returns:
            str: Model javobi
        """
        payload = {
            "model"  : self.model,
            "prompt" : prompt,
            "stream" : False,
            "options": {
                "temperature"  : temperature,
                "num_predict"  : 2048,   # Maksimal token soni
                "top_p"        : 0.9,
                "repeat_penalty": 1.1
            }
        }

        try:
            resp = requests.post(
                self.api_url,
                json    = payload,
                timeout = Config.OLLAMA_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()

        except requests.Timeout:
            raise TimeoutError(
                f"❌ Model {Config.OLLAMA_TIMEOUT}s da javob bermadi.\n"
                f"   Kattaroq model uchun Config.OLLAMA_TIMEOUT ni oshiring."
            )


# ════════════════════════════════════════════════════════════
#  5-QADAM: STRUCTURED OUTPUT GENERATOR
# ════════════════════════════════════════════════════════════

class StructuredOutputGenerator:
    """
    LLM dan tuzilgan (structured) chiqish oladi.
    
    Texnika:
      1. Retriever → eng mos bo'laklarni topadi
      2. Prompt Engineering → LLM ga JSON format talab qilinadi
      3. JSON Parser → javobni tahlil qiladi
      4. Validation → to'g'riligini tekshiradi
    """

    def __init__(self, llm: OllamaLLM, vector_db: VectorDB):
        self.llm       = llm
        self.vector_db = vector_db

    # ─────────────────────────────────────────────
    # ODDIY SAVOL-JAVOB
    # ─────────────────────────────────────────────
    def answer(self, question: str) -> dict:
        """
        Savolga kitobdan javob beradi.
        
        Args:
            question: Savol matni
            
        Returns:
            dict: {'answer': str, 'pages': List[int], 'confidence': str}
        """
        # Vektor qidiruv
        results = self.vector_db.search(question, top_k=Config.TOP_K)
        context = self._format_context(results)
        pages   = sorted(set(r["page"] for r in results))

        prompt = f"""You are an expert assistant. Answer the question ONLY based on the context below.
Respond in the same language as the question.

CONTEXT:
{context}

QUESTION: {question}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "answer": "detailed answer here",
  "confidence": "high/medium/low",
  "key_points": ["point1", "point2", "point3"]
}}"""

        raw = self.llm.generate(prompt, temperature=0.2)
        parsed = self._parse_json(raw)

        return {
            "question"  : question,
            "answer"    : parsed.get("answer", raw),
            "confidence": parsed.get("confidence", "medium"),
            "key_points": parsed.get("key_points", []),
            "pages"     : pages
        }

    # ─────────────────────────────────────────────
    # SAVOLLAR GENERATSIYA
    # ─────────────────────────────────────────────
    def generate_qa_pairs(
        self, count: int = 10, difficulty: str = "hard"
    ) -> List[QAPair]:
        """
        Kitobdan savol-javob juftliklarini generatsiya qiladi.
        
        Args:
            count     : Nechta savol
            difficulty: "easy" / "medium" / "hard"
            
        Returns:
            List[QAPair]: Savol-javob juftliklari
        """
        print(f"\n{'─'*55}")
        print(f"  4-QADAM: SAVOLLAR GENERATSIYA ({count} ta, {difficulty})")
        print(f"{'─'*55}")

        # Keng qidiruvlar bilan kontekst yig'ish
        queries = [
            "main concepts and key ideas",
            "important facts and details",
            "theories and principles",
            "examples and applications",
            "conclusions and results"
        ]

        all_results = []
        seen_texts  = set()
        for q in queries:
            for r in self.vector_db.search(q, top_k=3):
                if r["text"] not in seen_texts:
                    all_results.append(r)
                    seen_texts.add(r["text"])

        context = self._format_context(all_results[:12])
        pages   = sorted(set(r["page"] for r in all_results[:12]))

        difficulty_map = {
            "easy"  : "simple, factual, recall-based",
            "medium": "application and comprehension based",
            "hard"  : "analytical, synthesis, and critical thinking based"
        }
        diff_desc = difficulty_map.get(difficulty, "analytical")

        prompt = f"""You are an expert educator creating exam questions.
Based on the book content below, generate exactly {count} {diff_desc} questions.

BOOK CONTENT:
{context}

INSTRUCTIONS:
- Questions must be directly based on the content
- Each question needs a comprehensive answer
- Difficulty level: {difficulty.upper()}
- Use the content's original language or Uzbek

Return ONLY a JSON array (no markdown, no explanation):
[
  {{
    "question": "question text here",
    "answer": "detailed answer here",
    "difficulty": "{difficulty}"
  }}
]

Generate exactly {count} questions."""

        print(f"  🧠 LLM ({self.llm.model}) ishlamoqda...")
        start = time.time()
        raw   = self.llm.generate(prompt, temperature=0.4)
        elapsed = time.time() - start
        print(f"  ⏱️  Generatsiya vaqti: {elapsed:.1f}s")

        parsed = self._parse_json(raw)

        qa_pairs = []
        if isinstance(parsed, list):
            for item in parsed[:count]:
                if isinstance(item, dict) and "question" in item:
                    qa_pairs.append(QAPair(
                        question    = item.get("question", ""),
                        answer      = item.get("answer", ""),
                        difficulty  = item.get("difficulty", difficulty),
                        source_pages= pages
                    ))

        print(f"  ✅ {len(qa_pairs)} ta savol-javob yaratildi")
        return qa_pairs

    # ─────────────────────────────────────────────
    # TEST GENERATSIYA (MCQ)
    # ─────────────────────────────────────────────
    def generate_mcq(self, count: int = 5) -> List[MCQQuestion]:
        """
        Ko'p javobli testlar (MCQ) yaratadi.
        
        Args:
            count: Test savollar soni
            
        Returns:
            List[MCQQuestion]: Test savollar ro'yxati
        """
        print(f"\n  📋 {count} ta MCQ test yaratilmoqda...")

        results = self.vector_db.search("key facts and concepts", top_k=Config.TOP_K + 2)
        context = self._format_context(results)
        pages   = sorted(set(r["page"] for r in results))

        prompt = f"""Create {count} multiple choice questions (MCQ) from the content below.

CONTENT:
{context}

Return ONLY a JSON array:
[
  {{
    "question": "question text",
    "option_a": "first option",
    "option_b": "second option",
    "option_c": "third option",
    "option_d": "fourth option",
    "correct": "A",
    "explanation": "why this answer is correct"
  }}
]

Rules:
- Only ONE correct answer per question
- "correct" field must be exactly "A", "B", "C", or "D"
- Wrong options must be plausible but incorrect
- Generate exactly {count} questions."""

        raw    = self.llm.generate(prompt, temperature=0.3)
        parsed = self._parse_json(raw)

        mcq_list = []
        if isinstance(parsed, list):
            for item in parsed[:count]:
                if isinstance(item, dict) and "question" in item:
                    correct = item.get("correct", "A").upper()
                    if correct not in ("A", "B", "C", "D"):
                        correct = "A"
                    mcq_list.append(MCQQuestion(
                        question    = item.get("question", ""),
                        option_a    = item.get("option_a", ""),
                        option_b    = item.get("option_b", ""),
                        option_c    = item.get("option_c", ""),
                        option_d    = item.get("option_d", ""),
                        correct     = correct,
                        explanation = item.get("explanation", ""),
                        source_pages= pages
                    ))

        print(f"  ✅ {len(mcq_list)} ta MCQ yaratildi")
        return mcq_list

    # ─────────────────────────────────────────────
    # YORDAMCHI METODLAR
    # ─────────────────────────────────────────────
    def _format_context(self, results: List[dict]) -> str:
        """Qidiruv natijalarini LLM uchun formatlaydi"""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[Sahifa {r['page']}]\n{r['text']}")
        return "\n\n---\n\n".join(parts)

    def _parse_json(self, text: str):
        """LLM javobidan JSON ajratib oladi"""
        # Markdown code block larini olib tashlash
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # Array yoki object topish
        for start_ch, end_ch in [('[', ']'), ('{', '}')]:
            start = text.find(start_ch)
            if start == -1:
                continue
            # Muvozanatlangan yopish belgisini topish
            depth  = 0
            end_pos = -1
            for i, ch in enumerate(text[start:], start):
                if ch == start_ch:
                    depth += 1
                elif ch == end_ch:
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break
            if end_pos > start:
                try:
                    return json.loads(text[start:end_pos])
                except json.JSONDecodeError:
                    pass  # Keyingisini sinab ko'rish

        # Hech narsa topilmasa — xom matnni qaytarish
        return text


# ════════════════════════════════════════════════════════════
#  6-QADAM: NATIJALARNI SAQLASH
# ════════════════════════════════════════════════════════════

class ResultSaver:
    """
    Barcha natijalarni faylga saqlaydi.
    
    Formatlar:
      - JSON  : machine-readable, to'liq ma'lumot
      - TXT   : inson o'qiydigan, chiroyli format
      - chunks: Bo'laklarni CSV ko'rinishida
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_all(self, result: PipelineResult) -> dict:
        """
        Barcha natijalarni turli formatlarda saqlaydi.
        
        Args:
            result: PipelineResult ob'ekti
            
        Returns:
            dict: Saqlangan fayllar yo'llari
        """
        print(f"\n{'─'*55}")
        print(f"  5-QADAM: NATIJALARNI SAQLASH")
        print(f"{'─'*55}")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = Path(result.pdf_file).stem
        prefix    = f"{base_name}_{timestamp}"

        saved = {}

        # 1. To'liq JSON
        json_path = self.output_dir / f"{prefix}_full.json"
        self._save_json(result, json_path)
        saved["json"] = str(json_path)

        # 2. Savollar — o'qilishi qulay TXT
        qa_path = self.output_dir / f"{prefix}_questions.txt"
        self._save_questions_txt(result, qa_path)
        saved["questions_txt"] = str(qa_path)

        # 3. Test — TXT
        test_path = self.output_dir / f"{prefix}_test.txt"
        self._save_test_txt(result, test_path)
        saved["test_txt"] = str(test_path)

        # 4. Savollar — JSON (quiz app uchun)
        quiz_path = self.output_dir / f"{prefix}_quiz.json"
        self._save_quiz_json(result, quiz_path)
        saved["quiz_json"] = str(quiz_path)

        for fmt, path in saved.items():
            print(f"  💾 {fmt:15s} → {path}")

        return saved

    def save_chunks(self, chunks: List[TextChunk], pdf_name: str):
        """Bo'laklarni CSV faylga saqlaydi (debug uchun)"""
        path = self.output_dir / f"{Path(pdf_name).stem}_chunks.csv"
        with open(path, 'w', encoding='utf-8') as f:
            f.write("chunk_id,page_num,char_count,text_preview\n")
            for c in chunks:
                preview = c.text[:80].replace('"', "'").replace('\n', ' ')
                f.write(f'"{c.chunk_id}",{c.page_num},{c.char_count},"{preview}"\n')
        print(f"  💾 chunks_csv      → {path}")

    def _save_json(self, result: PipelineResult, path: Path):
        data = {
            "pdf_file"       : result.pdf_file,
            "total_pages"    : result.total_pages,
            "total_chunks"   : result.total_chunks,
            "processing_time": f"{result.processing_time:.1f}s",
            "questions"      : [asdict(q) for q in result.questions],
            "test_questions" : [asdict(q) for q in result.test_questions]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_questions_txt(self, result: PipelineResult, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"  KITOB: {result.pdf_file}\n")
            f.write(f"  SAVOL-JAVOB ({len(result.questions)} ta)\n")
            f.write("=" * 60 + "\n\n")

            for i, qa in enumerate(result.questions, 1):
                f.write(f"{'─'*60}\n")
                f.write(f"  {i}. SAVOL [{qa.difficulty.upper()}]\n")
                f.write(f"{'─'*60}\n")
                f.write(f"❓ {qa.question}\n\n")
                f.write(f"✅ JAVOB:\n{qa.answer}\n\n")
                f.write(f"📍 Sahifalar: {', '.join(str(p) for p in qa.source_pages)}\n\n")

    def _save_test_txt(self, result: PipelineResult, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"  KITOB: {result.pdf_file}\n")
            f.write(f"  KO'P JAVOBLI TEST ({len(result.test_questions)} ta)\n")
            f.write("=" * 60 + "\n\n")

            for i, q in enumerate(result.test_questions, 1):
                f.write(f"{'─'*60}\n")
                f.write(f"  {i}. {q.question}\n")
                f.write(f"{'─'*60}\n")
                f.write(f"  A) {q.option_a}\n")
                f.write(f"  B) {q.option_b}\n")
                f.write(f"  C) {q.option_c}\n")
                f.write(f"  D) {q.option_d}\n\n")
                f.write(f"  ✅ To'g'ri javob: {q.correct}\n")
                f.write(f"  💡 Izoh: {q.explanation}\n\n")

    def _save_quiz_json(self, result: PipelineResult, path: Path):
        quiz = {
            "title"   : f"{result.pdf_file} — Quiz",
            "questions": [
                {
                    "id"      : i + 1,
                    "question": q.question,
                    "choices" : {
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
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(quiz, f, ensure_ascii=False, indent=2)


# ════════════════════════════════════════════════════════════
#  ASOSIY PIPELINE
# ════════════════════════════════════════════════════════════

class PDFPipeline:
    """
    Barcha 5 qadamni birlashtiruvchi pipeline.
    
    Ishga tushirish:
        pipeline = PDFPipeline()
        pipeline.run("kitob.pdf")
    """

    def __init__(self):
        self.cfg       = Config()
        self.extractor = None
        self.chunker   = TextChunker(self.cfg.CHUNK_SIZE, self.cfg.CHUNK_OVERLAP)
        self.vector_db = None
        self.llm       = None
        self.generator = None
        self.saver     = ResultSaver(self.cfg.OUTPUT_DIR)

    def run(
        self,
        pdf_path      : str,
        question_count: int = None,
        test_count    : int = None,
        difficulty    : str = "hard",
        force_reload  : bool = False
    ) -> PipelineResult:
        """
        To'liq pipeline ni ishga tushiradi.
        
        Args:
            pdf_path      : PDF fayl yo'li
            question_count: Nechta ochiq savol (default: Config.QUESTION_COUNT)
            test_count    : Nechta test savoli (default: Config.TEST_COUNT)
            difficulty    : "easy" / "medium" / "hard"
            force_reload  : Bazani qayta yaratish
            
        Returns:
            PipelineResult: Barcha natijalar
        """
        start_time = time.time()
        q_count    = question_count or self.cfg.QUESTION_COUNT
        t_count    = test_count     or self.cfg.TEST_COUNT

        print("\n" + "═"*55)
        print("  PDF → CHUNKING → LLM → OUTPUT → SAVE")
        print("  100% Bepul Pipeline")
        print("═"*55)

        # ── 1. PDF dan matn ajratish ──────────────────────────
        self.extractor = PDFExtractor(pdf_path)
        pages = self.extractor.extract()

        # ── 2. Chunking ───────────────────────────────────────
        chunks = self.chunker.chunk_pages(pages, Path(pdf_path).name)

        # ── 3. Vektor bazasi ──────────────────────────────────
        self.vector_db = VectorDB(self.cfg.DB_DIR)

        if force_reload or self.vector_db.collection.count() == 0:
            self.vector_db.add_chunks(chunks)
        else:
            print(f"  ℹ️  Mavjud vektorlar ishlatilmoqda (force_reload=False)")

        # Chunks ni saqlab qo'yish (ixtiyoriy)
        self.saver.save_chunks(chunks, pdf_path)

        # ── 4. LLM ulash ─────────────────────────────────────
        print(f"\n{'─'*55}")
        print(f"  LLM ULANISHI")
        print(f"{'─'*55}")
        self.llm       = OllamaLLM(self.cfg.OLLAMA_MODEL, "http://localhost:11434")
        self.generator = StructuredOutputGenerator(self.llm, self.vector_db)

        # ── 5. Savollar generatsiya ───────────────────────────
        qa_pairs = self.generator.generate_qa_pairs(q_count, difficulty)

        # ── 6. MCQ test ───────────────────────────────────────
        mcq_list = self.generator.generate_mcq(t_count)

        # ── 7. Natijalar ──────────────────────────────────────
        elapsed = time.time() - start_time
        result  = PipelineResult(
            pdf_file        = Path(pdf_path).name,
            total_pages     = len(pages),
            total_chunks    = len(chunks),
            questions       = qa_pairs,
            test_questions  = mcq_list,
            processing_time = elapsed
        )

        saved_files = self.saver.save_all(result)

        # Yakuniy hisobot
        print(f"\n{'═'*55}")
        print(f"  ✅ PIPELINE YAKUNLANDI!")
        print(f"{'═'*55}")
        print(f"  📄 PDF       : {result.pdf_file}")
        print(f"  📑 Sahifalar : {result.total_pages}")
        print(f"  ✂️  Bo'laklar : {result.total_chunks}")
        print(f"  ❓ Savollar  : {len(result.questions)}")
        print(f"  📋 Testlar   : {len(result.test_questions)}")
        print(f"  ⏱️  Vaqt     : {elapsed:.1f}s")
        print(f"  📂 Papka     : {self.cfg.OUTPUT_DIR}/")
        print(f"{'═'*55}\n")

        return result

    def interactive(self, pdf_path: str):
        """
        Interaktiv rejim — foydalanuvchi savollar beradi.
        
        Buyruqlar:
          /savollar [N] [qiyinlik] — N ta savol generatsiya
          /test [N]                — N ta MCQ test
          /exit                    — Chiqish
        """
        # Avval pipeline ni ishlatsiz yuklash
        self.extractor = PDFExtractor(pdf_path)
        pages  = self.extractor.extract()
        chunks = self.chunker.chunk_pages(pages, Path(pdf_path).name)

        self.vector_db = VectorDB(self.cfg.DB_DIR)
        self.vector_db.add_chunks(chunks)

        self.llm       = OllamaLLM(self.cfg.OLLAMA_MODEL)
        self.generator = StructuredOutputGenerator(self.llm, self.vector_db)

        print("\n" + "═"*55)
        print("  🎯 INTERAKTIV REJIM")
        print("  Savolingizni yozing yoki buyruq bering:")
        print("  /savollar 10 hard  — 10 ta qiyin savol")
        print("  /test 5            — 5 ta MCQ test")
        print("  /exit              — Chiqish")
        print("═"*55 + "\n")

        while True:
            try:
                user_input = input("👤 Siz: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.lower() in ("/exit", "exit", "chiqish"):
                print("👋 Xayr!")
                break

            elif user_input.startswith("/savollar"):
                parts = user_input.split()
                n     = int(parts[1]) if len(parts) > 1 else 10
                diff  = parts[2] if len(parts) > 2 else "hard"
                qa    = self.generator.generate_qa_pairs(n, diff)
                for i, q in enumerate(qa, 1):
                    print(f"\n{i}. ❓ {q.question}")
                    print(f"   ✅ {q.answer}")

            elif user_input.startswith("/test"):
                parts = user_input.split()
                n     = int(parts[1]) if len(parts) > 1 else 5
                mcqs  = self.generator.generate_mcq(n)
                for i, q in enumerate(mcqs, 1):
                    print(f"\n{i}. {q.question}")
                    for opt, text in [("A", q.option_a), ("B", q.option_b),
                                       ("C", q.option_c), ("D", q.option_d)]:
                        mark = " ✅" if opt == q.correct else ""
                        print(f"   {opt}) {text}{mark}")

            else:
                # Oddiy savol
                result = self.generator.answer(user_input)
                print(f"\n🤖 Javob: {result['answer']}")
                if result.get("key_points"):
                    print("📌 Asosiy fikrlar:")
                    for kp in result["key_points"]:
                        print(f"   • {kp}")
                print(f"📍 Manbalar: {result['pages']}\n")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline = PDFPipeline()

    # ── To'liq avtomatik ishga tushirish ─────────────────────
    result = pipeline.run(
        pdf_path       = Config.PDF_PATH,   # "kitob.pdf" ni shu papkaga qo'ying
        question_count = 10,
        test_count     = 5,
        difficulty     = "hard",            # "easy" / "medium" / "hard"
        force_reload   = False              # True: bazani qayta yaratadi
    )

    # ── Interaktiv rejim ─────────────────────────────────────
    # pipeline.interactive(Config.PDF_PATH)