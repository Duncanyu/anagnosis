import sys, pathlib, traceback
from PySide6 import QtWidgets, QtCore, QtGui

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import markdown as mdlib
except Exception:
    mdlib = None

from api.services.parse import parse_any_bytes
from api.services.chunk import chunk_pages
from api.services.index import add_chunks, search
from api.services.summarize import summarize, summarize_document, summarizer_info
from api.services.embed import embedding_info
from api.core.config import save_secret, present_keys, load_config

APP_TITLE = "Reading Agent"

def render_markdown(widget, md_text):
    if mdlib:
        html = mdlib.markdown(md_text, extensions=["fenced_code", "tables", "toc"])
        css = "<style>body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.45;padding:8px}h1,h2,h3{margin-top:1em}code{background:#f6f8fa;padding:2px 4px;border-radius:4px}pre{background:#f6f8fa;padding:8px;border-radius:8px;overflow:auto}ul{margin-left:1.2em}blockquote{border-left:3px solid #ddd;padding-left:8px;color:#555}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:4px 8px}</style>"
        widget.setHtml(f"<html><head><meta charset='utf-8'>{css}</head><body>{html}</body></html>")
    else:
        try:
            widget.setMarkdown(md_text)
        except Exception:
            widget.setPlainText(md_text)

class IngestWorker(QtCore.QThread):
    progress = QtCore.Signal(str)
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def run(self):
        try:
            summaries = []
            details = []
            total_chunks = 0

            emb = embedding_info()
            self.progress.emit(f"Embedding: {emb['backend']} ({emb['model']})")
            sumi = summarizer_info()
            self.progress.emit(f"Summarizer: {sumi['backend']}")

            for p in self.paths:
                self.progress.emit(f"Loading {p.name}…")
                data = p.read_bytes()
                parsed = parse_any_bytes(p.name, data)
                for page in parsed["pages"]:
                    page["doc_name"] = p.name
                self.progress.emit(f"Chunking {p.name}…")
                chunks = chunk_pages(parsed["pages"])
                self.progress.emit(f"Indexing {p.name}…")
                add_chunks(chunks)
                total_chunks += len(chunks)
                self.progress.emit(f"Summarizing {p.name}…")
                docsum = summarize_document(chunks)
                summaries.append(f"## {p.name}\n\n{docsum['summary']}")
                details.append({"file": str(p), "num_pages": parsed["num_pages"], "ocr_pages": parsed["ocr_pages"], "num_chunks": len(chunks)})

            combined = "\n\n".join(summaries) if summaries else "No documents ingested."
            self.finished.emit({"details": details, "num_docs": len(self.paths), "num_chunks": total_chunks, "doc_summary": combined})
        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n{tb}")

class AskWorker(QtCore.QThread):
    progress = QtCore.Signal(str)
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, question, k, history):
        super().__init__()
        self.question = question
        self.k = k
        self.history = history

    def run(self):
        try:
            self.progress.emit("Searching index…")
            hits = search(self.question, k=self.k)
            if not hits:
                self.finished.emit({"answer": "**No results. Ingest a document first.**", "citations": []})
                return
            top_chunks = [h[1] for h in hits]
            self.progress.emit("Summarizing with context…")
            out = summarize(self.question, top_chunks, history=self.history)
            self.finished.emit(out)
        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n{tb}")

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        cfg = load_config()
        form = QtWidgets.QFormLayout()
        self.openai = QtWidgets.QLineEdit(cfg.get("OPENAI_API_KEY") or "")
        self.openai.setEchoMode(QtWidgets.QLineEdit.Password)
        self.hf = QtWidgets.QLineEdit(cfg.get("HF_TOKEN") or "")
        self.hf.setEchoMode(QtWidgets.QLineEdit.Password)
        self.embed_backend = QtWidgets.QComboBox()
        self.embed_backend.addItems(["hf","openai"])
        self.embed_backend.setCurrentText((cfg.get("EMBED_BACKEND") or "hf").lower())
        self.llm_backend = QtWidgets.QComboBox()
        self.llm_backend.addItems(["openai","vllm"])
        self.llm_backend.setCurrentText((cfg.get("LLM_BACKEND") or "openai").lower())
        form.addRow("OPENAI_API_KEY", self.openai)
        form.addRow("HF_TOKEN", self.hf)
        form.addRow("Embedding backend", self.embed_backend)
        form.addRow("LLM backend", self.llm_backend)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save); buttons.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(form); lay.addWidget(buttons)

    def save(self):
        if self.openai.text().strip():
            save_secret("OPENAI_API_KEY", self.openai.text().strip(), prefer="keyring")
        if self.hf.text().strip():
            save_secret("HF_TOKEN", self.hf.text().strip(), prefer="keyring")
        save_secret("EMBED_BACKEND", self.embed_backend.currentText(), prefer="file")
        save_secret("LLM_BACKEND", self.llm_backend.currentText(), prefer="file")
        self.accept()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1000, 720)
        tb = self.addToolBar("Main")
        act_settings = QtGui.QAction("Settings", self); act_settings.triggered.connect(self.open_settings); tb.addAction(act_settings)
        self.status = QtWidgets.QStatusBar(); self.setStatusBar(self.status)
        self.tabs = QtWidgets.QTabWidget(); self.setCentralWidget(self.tabs)

        ingest = QtWidgets.QWidget(); v1 = QtWidgets.QVBoxLayout(ingest)
        self.pick_btn = QtWidgets.QPushButton("Choose files…"); self.pick_btn.clicked.connect(self.choose_files)
        self.ingest_btn = QtWidgets.QPushButton("Ingest"); self.ingest_btn.clicked.connect(self.ingest_docs); self.ingest_btn.setEnabled(False)
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.pick_btn); row.addWidget(self.ingest_btn)
        self.sel_label = QtWidgets.QLabel("No files selected.")
        self.ingest_log = QtWidgets.QPlainTextEdit(); self.ingest_log.setReadOnly(True)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0,0)
        v1.addLayout(row); v1.addWidget(self.sel_label); v1.addWidget(self.ingest_log); v1.addWidget(self.progress); self.progress.hide()

        ask = QtWidgets.QWidget(); v2 = QtWidgets.QVBoxLayout(ask)
        self.q_edit = QtWidgets.QPlainTextEdit(); self.q_edit.setPlaceholderText("Ask anything about your readings…")
        self.k_spin = QtWidgets.QSpinBox(); self.k_spin.setRange(1, 20); self.k_spin.setValue(6)
        self.ask_btn = QtWidgets.QPushButton("Ask"); self.ask_btn.clicked.connect(self.run_ask)
        controls = QtWidgets.QHBoxLayout(); controls.addWidget(QtWidgets.QLabel("Top-k:")); controls.addWidget(self.k_spin); controls.addStretch(1); controls.addWidget(self.ask_btn)
        self.answer = QtWidgets.QTextEdit(); self.answer.setReadOnly(True)
        v2.addWidget(self.q_edit); v2.addLayout(controls); v2.addWidget(self.answer)

        self.tabs.addTab(ingest, "Ingest")
        self.tabs.addTab(ask, "Ask")

        self.selected_paths = []
        self.ingest_worker = None
        self.ask_worker = None
        self.history = []; self.history_limit = 8
        self.update_key_status()

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec(): self.update_key_status()

    def update_key_status(self):
        status = present_keys()
        bits = []
        bits.append(f"OPENAI: {'✓' if status.get('OPENAI_API_KEY') else '×'}")
        bits.append(f"HF: {'✓' if status.get('HF_TOKEN') else '×'}")
        self.status.showMessage(" | ".join(bits))

    def choose_files(self):
        fns, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose files", "", "PDF and Images (*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)")
        if not fns: return
        self.selected_paths = [pathlib.Path(fn) for fn in fns]
        names = ", ".join(p.name for p in self.selected_paths[:3])
        more = "" if len(self.selected_paths) <= 3 else f" …+{len(self.selected_paths)-3} more"
        self.sel_label.setText(f"Selected: {names}{more}")
        self.ingest_btn.setEnabled(True)

    def ingest_docs(self):
        if not self.selected_paths: return
        self.ingest_log.clear(); self.progress.show()
        self.ingest_btn.setEnabled(False); self.pick_btn.setEnabled(False)
        self.ingest_worker = IngestWorker(self.selected_paths)
        self.ingest_worker.progress.connect(self.ingest_log.appendPlainText)
        self.ingest_worker.finished.connect(self.ingest_done)
        self.ingest_worker.failed.connect(self.ingest_fail)
        self.ingest_worker.start()

    def ingest_done(self, info):
        self.progress.hide(); self.ingest_btn.setEnabled(True); self.pick_btn.setEnabled(True)
        for d in info.get("details", []):
            self.ingest_log.appendPlainText(f"\n{pathlib.Path(d['file']).name}\nPages: {d['num_pages']} (OCR: {d['ocr_pages']})\nChunks: {d['num_chunks']}")
        self.status.showMessage(f"Ingested {info.get('num_docs',0)} file(s) — chunks: {info.get('num_chunks',0)}")
        md = "### Auto-summary\n\n" + info.get("doc_summary","")
        render_markdown(self.answer, md)
        self.tabs.setCurrentIndex(1)

    def ingest_fail(self, msg):
        self.progress.hide(); self.ingest_btn.setEnabled(True); self.pick_btn.setEnabled(True)
        self.ingest_log.appendPlainText("\n[ERROR]\n" + msg)
        QtWidgets.QMessageBox.critical(self, "Ingest failed", msg)

    def run_ask(self):
        q = self.q_edit.toPlainText().strip()
        if not q:
            QtWidgets.QMessageBox.information(self, "Ask", "Type a question first."); return
        render_markdown(self.answer, "_Working…_"); self.ask_btn.setEnabled(False)
        self.ask_worker = AskWorker(q, self.k_spin.value(), history=self.history)
        self.ask_worker.progress.connect(lambda s: self.status.showMessage(s))
        self.ask_worker.finished.connect(self.ask_done)
        self.ask_worker.failed.connect(self.ask_fail)
        self.ask_worker.start()

    def ask_done(self, out):
        self.ask_btn.setEnabled(True)
        text = out.get("answer", "")
        cites = ", ".join(out.get("citations", []))
        quotes = out.get("quotes", [])
        qmd = ""
        if quotes:
            qmd = "\n\n### Evidence snippets\n" + "\n".join([f"> {q['quote']}\n>\n> — {q['source']}" for q in quotes])
        md = text + "\n\n**Citations:** " + cites + qmd
        render_markdown(self.answer, md)
        self.history.append({"q": self.q_edit.toPlainText().strip(), "a": text})
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]
        self.status.showMessage("Answer ready")


    def ask_fail(self, msg):
        self.ask_btn.setEnabled(True)
        render_markdown(self.answer, "**ERROR**\n\n```\n" + msg + "\n```")
        QtWidgets.QMessageBox.critical(self, "Ask failed", msg)

def main():
    app = QtWidgets.QApplication(sys.argv); app.setApplicationName(APP_TITLE)
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
