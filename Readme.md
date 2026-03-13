# PDF QA Tizimi — API + CLI

PDF kitoblarni tahlil qilib, savol-javob va testlar yaratuvchi tizim.
**100% bepul vositalar** — internet kerak emas (model yuklangandan keyin).

```
pdf_qa/
├── pipeline.py      ← Asosiy logika (Extract→Chunk→Embed→LLM→Save)
├── api.py           ← FastAPI REST server
├── main.py          ← CLI (terminal buyruqlari)
├── requirements.txt ← Kutubxonalar
└── README.md
```

---

## O'rnatish

```bash
# 1. Python kutubxonalari
pip install -r requirements.txt

# 2. Ollama (local LLM server)
curl -fsSL https://ollama.com/install.sh | sh   # Linux/Mac
# Windows: https://ollama.com/download

# 3. Llama 3 modeli yuklash (bir marta, ~4.7 GB)
ollama pull llama3

# 4. Ollama serverni ishga tushirish
ollama serve
```

---

## CLI Ishlatish

```bash
# PDF yuklash
python main.py upload kitob.pdf

# Savolga javob (book_id — upload dan qaytgan)
python main.py ask abc123def456 "Kitobning asosiy mavzusi nima?"

# 10 ta qiyin savol-javob generatsiya
python main.py qa abc123def456 --count 10 --difficulty hard

# 5 ta MCQ test
python main.py mcq abc123def456 --count 5 --output test.json

# To'liq pipeline (yuklash + generatsiya + saqlash)
python main.py run kitob.pdf --count 10 --difficulty hard

# Interaktiv suhbat
python main.py chat abc123def456

# Barcha kitoblar ro'yxati
python main.py books
```

---

## API Ishlatish

```bash
# Serverni ishga tushirish
uvicorn api:app --reload --port 8000

# Swagger UI
http://localhost:8000/docs
```

### Endpointlar

| Method | URL                              | Tavsif                    |
|--------|----------------------------------|---------------------------|
| GET    | /health                          | Server holati             |
| POST   | /books/upload                    | PDF yuklash               |
| GET    | /books                           | Barcha kitoblar           |
| GET    | /books/{book_id}                 | Kitob ma'lumotlari        |
| DELETE | /books/{book_id}                 | Kitobni o'chirish         |
| POST   | /books/{book_id}/ask             | Savolga javob             |
| POST   | /books/{book_id}/generate/qa     | QA generatsiya            |
| POST   | /books/{book_id}/generate/mcq    | MCQ generatsiya           |
| POST   | /books/{book_id}/process         | To'liq generatsiya        |
| GET    | /files/{filename}                | Fayl yuklash              |

### curl Misollar

```bash
# PDF yuklash
curl -X POST http://localhost:8000/books/upload \
     -F "file=@kitob.pdf"
# → {"book_id": "abc123def456", ...}

# Savolga javob
curl -X POST http://localhost:8000/books/abc123def456/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Asosiy mavzu nima?"}'

# 10 ta qiyin savol generatsiya
curl -X POST http://localhost:8000/books/abc123def456/generate/qa \
     -H "Content-Type: application/json" \
     -d '{"count": 10, "difficulty": "hard"}'

# 5 ta MCQ test
curl -X POST http://localhost:8000/books/abc123def456/generate/mcq \
     -H "Content-Type: application/json" \
     -d '{"count": 5}'
```

---

## Arxitektura

```
PDF fayl
  ↓
[1] PDFExtractor        PyMuPDF bilan sahifa-sahifa matn ajratish
  ↓
[2] TextChunker         800 belgili bo'laklar (150 belgi overlap)
  ↓
[3] VectorDB            sentence-transformers embedding + ChromaDB
  ↓
[4] OllamaLLM           Local Llama 3 (HTTP API, port 11434)
  ↓
[5] StructuredOutput    JSON formatida QA va MCQ
  ↓
[6] ResultSaver         JSON + TXT + Quiz JSON fayllari
```

---

## Sozlamalar (Environment Variables)

```bash
export OLLAMA_MODEL=mistral        # Model tanlash (default: llama3)
export CHUNK_SIZE=1000             # Bo'lak hajmi (default: 800)
export CHUNK_OVERLAP=200           # Overlap (default: 150)
export TOP_K=5                     # Vektor qidiruv soni (default: 5)
export UPLOAD_DIR=uploads          # PDF saqlash papkasi
export OUTPUT_DIR=output           # Natijalar papkasi
export DB_DIR=chroma_db            # ChromaDB papkasi
```