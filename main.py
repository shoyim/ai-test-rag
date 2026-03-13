"""
main.py
═══════════════════════════════════════════════════════════════
CLI (Command Line Interface) — terminal orqali ishlatish.

Ishlatilish:
  python main.py upload kitob.pdf
  python main.py ask    <book_id> "savol matni"
  python main.py qa     <book_id> --count 10 --difficulty hard
  python main.py mcq    <book_id> --count 5
  python main.py run    kitob.pdf             # To'liq pipeline
  python main.py chat   <book_id>             # Interaktiv suhbat
  python main.py books                        # Ro'yxat
═══════════════════════════════════════════════════════════════
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from dataclasses import asdict

# Bizning modular pipeline
from pipeline import Config, PDFPipeline

# Chiroyli chiqish uchun logging
logging.basicConfig(
    level  = logging.WARNING,   # CLI da faqat muhim xabarlar
    format = "%(levelname)s: %(message)s"
)


# ════════════════════════════════════════════════════════════
#  CHIQISH YORDAMCHILARI
# ════════════════════════════════════════════════════════════

def print_header(title: str):
    """Sarlavha chiqaradi"""
    w = 60
    print("\n" + "═" * w)
    print(f"  {title}")
    print("═" * w)

def print_section(title: str):
    """Bo'lim sarlavhasi"""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print("─" * 50)

def print_json(data):
    """JSON formatlash bilan chiqaradi"""
    print(json.dumps(data, ensure_ascii=False, indent=2))


# ════════════════════════════════════════════════════════════
#  BUYRUQ HANDLERLARI
# ════════════════════════════════════════════════════════════

def cmd_upload(args, pipeline: PDFPipeline):
    """
    PDF yuklash va bo'laklash.
    Muvaffaqiyatli bo'lsa book_id chiqaradi.
    """
    pdf = args.pdf
    if not Path(pdf).exists():
        print(f"❌ Fayl topilmadi: '{pdf}'")
        sys.exit(1)

    print_header(f"PDF YUKLANMOQDA: {pdf}")
    print(f"  force_reload: {args.force}")

    book_id = pipeline.ingest(pdf, force_reload=args.force)

    chunk_count = pipeline._get_db(book_id).count()
    print(f"\n✅ Muvaffaqiyatli yuklandi!")
    print(f"   book_id     : {book_id}")
    print(f"   Bo'laklar   : {chunk_count}")
    print(f"\n💡 Keyingi buyruqlar:")
    print(f"   python main.py ask  {book_id} \"savolingiz\"")
    print(f"   python main.py qa   {book_id} --count 10")
    print(f"   python main.py mcq  {book_id} --count 5")
    print(f"   python main.py chat {book_id}")


def cmd_ask(args, pipeline: PDFPipeline):
    """
    Kitobdan savolga javob olish.
    """
    print_header(f"SAVOL: {args.question}")

    result = pipeline.ask(args.book_id, args.question)

    print(f"\n❓ Savol     : {result.question}")
    print(f"📊 Ishonch   : {result.confidence.upper()}")
    print(f"\n💬 JAVOB:")
    print("─" * 50)
    print(result.answer)

    if result.key_points:
        print(f"\n📌 Asosiy fikrlar:")
        for kp in result.key_points:
            print(f"   • {kp}")

    print(f"\n📍 Sahifalar: {', '.join(str(p) for p in result.source_pages)}")

    # JSON chiqarish (--json bayrog'i bilan)
    if getattr(args, "json_out", False):
        print("\n── JSON ──")
        print_json(asdict(result))


def cmd_qa(args, pipeline: PDFPipeline):
    """
    Kitobdan savol-javob generatsiya.
    """
    print_header(f"SAVOL-JAVOB GENERATSIYA")
    print(f"  book_id   : {args.book_id}")
    print(f"  Soni      : {args.count}")
    print(f"  Qiyinlik  : {args.difficulty}")

    qa_list = pipeline.generate_qa(args.book_id, args.count, args.difficulty)

    print_section(f"{len(qa_list)} ta savol-javob")

    for i, qa in enumerate(qa_list, 1):
        print(f"\n{'─' * 55}")
        print(f"  {i:2}. [{qa.difficulty.upper()}]")
        print(f"{'─' * 55}")
        print(f"❓ {qa.question}\n")
        print(f"✅ {qa.answer}")
        print(f"📍 Sahifalar: {', '.join(str(p) for p in qa.source_pages)}")

    # JSON saqlash
    if getattr(args, "output", None):
        import json
        data = [asdict(q) for q in qa_list]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Saqlandi: {args.output}")


def cmd_mcq(args, pipeline: PDFPipeline):
    """
    Ko'p javobli test generatsiya.
    """
    print_header(f"MCQ TEST GENERATSIYA")
    print(f"  book_id : {args.book_id}")
    print(f"  Soni    : {args.count}")

    mcq_list = pipeline.generate_mcq(args.book_id, args.count)

    print_section(f"{len(mcq_list)} ta test savoli")

    for i, q in enumerate(mcq_list, 1):
        print(f"\n{'─' * 55}")
        print(f"  {i:2}. {q.question}")
        print(f"{'─' * 55}")
        for letter, text in [("A", q.option_a), ("B", q.option_b),
                               ("C", q.option_c), ("D", q.option_d)]:
            mark = " ✅" if letter == q.correct else ""
            print(f"  {letter}) {text}{mark}")
        print(f"\n  💡 Izoh: {q.explanation}")

    if getattr(args, "output", None):
        import json
        data = [asdict(q) for q in mcq_list]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Saqlandi: {args.output}")


def cmd_run(args, pipeline: PDFPipeline):
    """
    To'liq pipeline: yuklash + QA + MCQ + saqlash.
    """
    result = pipeline.run_full(
        pdf_path      = args.pdf,
        question_count= args.count,
        test_count    = args.test_count,
        difficulty    = args.difficulty,
        force_reload  = args.force
    )

    print_header("PIPELINE YAKUNLANDI ✅")
    print(f"  PDF         : {result.pdf_file}")
    print(f"  Bo'laklar   : {result.total_chunks}")
    print(f"  Savollar    : {len(result.questions)}")
    print(f"  Testlar     : {len(result.test_questions)}")
    print(f"  Vaqt        : {result.processing_time:.1f}s")

    if result.saved_files:
        print(f"\n📂 Saqlangan fayllar:")
        for fmt, path in result.saved_files.items():
            print(f"   {fmt:12s} → {path}")


def cmd_chat(args, pipeline: PDFPipeline):
    """
    Interaktiv suhbat rejimi.
    Foydalanuvchi savollar beradi, /quit bilan chiqadi.
    """
    print_header(f"INTERAKTIV SUHBAT | book_id: {args.book_id}")
    print("  Savolingizni yozing. Chiqish: /quit")
    print("  Buyruqlar:")
    print("    /qa  [N] [qiyinlik]  — N ta savol generatsiya")
    print("    /mcq [N]             — N ta MCQ test")
    print("    /quit                — Chiqish")
    print()

    while True:
        try:
            user_input = input("👤 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Xayr!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "exit"):
            print("👋 Xayr!")
            break

        elif user_input.startswith("/qa"):
            # /qa 10 hard
            parts = user_input.split()
            n     = int(parts[1]) if len(parts) > 1 else 5
            diff  = parts[2]      if len(parts) > 2 else "hard"
            qa_list = pipeline.generate_qa(args.book_id, n, diff)
            for i, q in enumerate(qa_list, 1):
                print(f"\n{i}. {q.question}")
                print(f"   → {q.answer}")

        elif user_input.startswith("/mcq"):
            parts    = user_input.split()
            n        = int(parts[1]) if len(parts) > 1 else 5
            mcq_list = pipeline.generate_mcq(args.book_id, n)
            for i, q in enumerate(mcq_list, 1):
                print(f"\n{i}. {q.question}")
                for l, t in [("A", q.option_a), ("B", q.option_b),
                              ("C", q.option_c), ("D", q.option_d)]:
                    mark = " ✅" if l == q.correct else ""
                    print(f"   {l}) {t}{mark}")

        else:
            # Oddiy savol
            result = pipeline.ask(args.book_id, user_input)
            print(f"\n🤖 {result.answer}")
            if result.key_points:
                for kp in result.key_points:
                    print(f"   • {kp}")
            print(f"   📍 {result.source_pages}")
            print()


def cmd_books(args, pipeline: PDFPipeline):
    """
    Barcha yuklangan kitoblarni ko'rsatadi (registry fayl orqali).
    """
    import json
    from pathlib import Path

    registry_file = "books_registry.json"
    if not Path(registry_file).exists():
        print("📚 Hali hech qanday kitob yuklanmagan.")
        print("   PDF yuklash: python main.py upload kitob.pdf")
        return

    with open(registry_file, "r", encoding="utf-8") as f:
        registry = json.load(f)

    if not registry:
        print("📚 Registry bo'sh.")
        return

    print_header(f"YUKLANGAN KITOBLAR ({len(registry)} ta)")
    for bid, info in registry.items():
        print(f"\n  book_id  : {bid}")
        print(f"  Fayl     : {info['filename']}")
        print(f"  Sana     : {info['upload_time']}")
        print(f"  Holat    : {info['status']}")
        print(f"  Bo'laklar: {info.get('chunk_count', '?')}")
        print(f"  ─────────────────────────────")


# ════════════════════════════════════════════════════════════
#  ARGUMENTLAR TAHLILI (argparse)
# ════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """
    CLI argument parser quradi.
    Subcommandlar: upload, ask, qa, mcq, run, chat, books
    """
    parser = argparse.ArgumentParser(
        prog       = "pdfqa",
        description= "PDF kitob tahlil va savol-javob tizimi (100%% bepul)",
        formatter_class= argparse.RawDescriptionHelpFormatter,
        epilog     = """
Misollar:
  python main.py upload kitob.pdf
  python main.py ask    abc123 "Kitobning asosiy g'oyasi nima?"
  python main.py qa     abc123 --count 10 --difficulty hard
  python main.py mcq    abc123 --count 5  --output test.json
  python main.py run    kitob.pdf --count 10 --difficulty hard
  python main.py chat   abc123
  python main.py books
        """
    )

    sub = parser.add_subparsers(dest="command", help="Buyruq")
    sub.required = True

    # ── upload ────────────────────────────────────────────
    p_upload = sub.add_parser("upload", help="PDF yuklash")
    p_upload.add_argument("pdf", help="PDF fayl yo'li")
    p_upload.add_argument(
        "--force", "-f",
        action ="store_true",
        help   ="Bazani qayta yaratish (force reload)"
    )

    # ── ask ───────────────────────────────────────────────
    p_ask = sub.add_parser("ask", help="Savolga javob")
    p_ask.add_argument("book_id", help="upload buyrug'idan qaytgan ID")
    p_ask.add_argument("question", help="Savol matni")
    p_ask.add_argument("--json",   dest="json_out", action="store_true")

    # ── qa ────────────────────────────────────────────────
    p_qa = sub.add_parser("qa", help="Savol-javoblar generatsiya")
    p_qa.add_argument("book_id")
    p_qa.add_argument("--count",      "-n", type=int, default=10)
    p_qa.add_argument("--difficulty", "-d", default="hard",
                      choices=["easy", "medium", "hard"])
    p_qa.add_argument("--output",     "-o", help="JSON faylga saqlash")

    # ── mcq ───────────────────────────────────────────────
    p_mcq = sub.add_parser("mcq", help="MCQ test generatsiya")
    p_mcq.add_argument("book_id")
    p_mcq.add_argument("--count",  "-n", type=int, default=5)
    p_mcq.add_argument("--output", "-o", help="JSON faylga saqlash")

    # ── run (to'liq pipeline) ─────────────────────────────
    p_run = sub.add_parser("run", help="To'liq pipeline (yuklash + generatsiya)")
    p_run.add_argument("pdf")
    p_run.add_argument("--count",      "-n",  type=int, default=10)
    p_run.add_argument("--test-count", "-t",  type=int, default=5,  dest="test_count")
    p_run.add_argument("--difficulty", "-d",  default="hard",
                       choices=["easy", "medium", "hard"])
    p_run.add_argument("--force", "-f", action="store_true")

    # ── chat ──────────────────────────────────────────────
    p_chat = sub.add_parser("chat", help="Interaktiv suhbat rejimi")
    p_chat.add_argument("book_id")

    # ── books ─────────────────────────────────────────────
    sub.add_parser("books", help="Yuklangan kitoblar ro'yxati")

    return parser


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Pipeline ob'ektini yaratish
    cfg      = Config()
    pipeline = PDFPipeline(cfg)

    # Buyruqni bajarish
    command_map = {
        "upload": cmd_upload,
        "ask"   : cmd_ask,
        "qa"    : cmd_qa,
        "mcq"   : cmd_mcq,
        "run"   : cmd_run,
        "chat"  : cmd_chat,
        "books" : cmd_books,
    }

    handler = command_map.get(args.command)
    if handler:
        try:
            handler(args, pipeline)
        except KeyboardInterrupt:
            print("\n⛔ To'xtatildi")
        except ConnectionError as e:
            print(f"\n❌ Ollama ulanish xatosi: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Xato: {e}")
            if "--debug" in sys.argv:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()