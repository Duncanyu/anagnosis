import sys, pathlib, traceback, json, os
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
from api.services.index import add_chunks, search, clear_index
from api.services.summarize import summarize, summarize_document, summarizer_info, summarize_batched
from api.services.embed import embedding_info
from api.services import memory as mem
from api.services import aliases
from api.core.config import save_secret, present_keys, load_config


APP_TITLE = "Anagnosis"
PREFS_PATH = pathlib.Path("artifacts") / "ui_prefs.json"

def as_bool(x):
    s = str(x).strip().lower()
    return s in {"1","true","yes","on","t","y"}

def read_prefs():
    try:
        if PREFS_PATH.exists():
            return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def write_prefs(d):
    try:
        PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREFS_PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass

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
    progress_pct = QtCore.Signal(int)
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def _log(self, s):
        try:
            print(s, flush=True)
        except Exception:
            pass
        self.progress.emit(s)

    def run(self):
        try:
            summaries, details, total_chunks = [], [], 0
            emb = embedding_info(); self.progress.emit(f"Embedding: {emb['backend']} ({emb['model']})")
            sumi = summarizer_info(); self.progress.emit(f"Summarizer: {sumi['backend']}")
            for p in self.paths:
                self.progress.emit(f"Loading {p.name}…")
                self.progress_pct.emit(2)
                data = p.read_bytes()

                self.progress.emit("Parsing document…")
                parsed = parse_any_bytes(p.name, data, progress_cb=self.progress.emit)
                self.progress_pct.emit(15)

                ocr_list = parsed.get("ocr_page_numbers") or []
                suspect_list = parsed.get("suspect_pages") or []
                if ocr_list:
                    head = ", ".join(str(x) for x in ocr_list[:20])
                    tail = " ..." if len(ocr_list) > 20 else ""
                    self._log(f"OCR pages: [{head}]{tail}")
                else:
                    self._log("OCR pages: []")
                if suspect_list:
                    head = ", ".join(str(x) for x in suspect_list[:20])
                    tail = " ..." if len(suspect_list) > 20 else ""
                    self._log(f"Suspect pages: [{head}]{tail}")
                else:
                    self._log("Suspect pages: []")
                    
                salv_list = parsed.get("salvaged_pages") or []

                if salv_list:
                    head = ", ".join(str(x) for x in salv_list[:20])
                    tail = " ..." if len(salv_list) > 20 else ""
                    self._log(f"Salvaged pages: [{head}]{tail}")
                else:
                    self._log("Salvaged pages: []")

                for page in parsed["pages"]:
                    page["doc_name"] = p.name

                self.progress.emit("Chunking…")
                chunks = chunk_pages(parsed["pages"])
                self.progress_pct.emit(35)

                def embed_cb(done, total):
                    base = 35
                    span = 55
                    pct = base + int(span * (done / max(1, total)))
                    pct = min(95, max(36, pct))
                    self.progress_pct.emit(pct)

                self.progress.emit("Embedding and indexing…")
                add_chunks(chunks, progress_cb=embed_cb)
                self.progress_pct.emit(96)

                total_chunks += len(chunks)

                self.progress.emit("Summarizing…")
                docsum = summarize_document(chunks)
                self.progress_pct.emit(100)

                summaries.append(f"## {p.name}\n\n{docsum['summary']}")
                details.append({
                    "file": str(p),
                    "num_pages": parsed["num_pages"],
                    "ocr_pages": parsed["ocr_pages"],
                    "ocr_page_numbers": ocr_list,
                    "suspect_pages": suspect_list,
                    "num_chunks": len(chunks)
                })

            combined = "\n\n".join(summaries) if summaries else "No documents ingested."
            self.finished.emit({"details": details, "num_docs": len(self.paths), "num_chunks": total_chunks, "doc_summary": combined})
        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n{tb}")

class AskWorker(QtCore.QThread):
    progress = QtCore.Signal(str)
    progress_pct = QtCore.Signal(int)
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
            self.progress_pct.emit(15)
            pool = int(os.environ.get("ASK_CANDIDATES","300"))
            hits = search(self.question, k=pool)
            self.progress.emit(f"Hits: {len(hits)}")
            if not hits:
                self.progress_pct.emit(100)
                self.finished.emit({"answer": "**No results. Ingest a document first.**", "citations": []})
                return
            chs = []
            for h in hits:
                try:
                    sc, ch = h[0], h[1]
                except Exception:
                    sc, ch = 0.0, h[1] if isinstance(h, (list, tuple)) and len(h) > 1 else h
                x = dict(ch)
                x["_score"] = float(sc) if sc is not None else 0.0
                chs.append(x)
            self.progress.emit("Summarizing in batches…")
            self.progress_pct.emit(60)
            tb = int(os.environ.get("ASK_TIME_BUDGET_SEC","120"))
            mb = int(os.environ.get("ASK_MAX_BATCHES","6"))
            exh = os.environ.get("ASK_EXHAUSTIVE","false").lower() in {"1","true","yes","on"}
            out = summarize_batched(self.question, chs, history=self.history, progress_cb=lambda s: self.progress.emit(s), max_batches=mb, time_budget_sec=tb, exhaustive=exh)
            self.progress_pct.emit(100)
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
        prefs = read_prefs()
        form = QtWidgets.QFormLayout()

        self.openai = QtWidgets.QLineEdit(cfg.get("OPENAI_API_KEY") or "")
        self.openai.setEchoMode(QtWidgets.QLineEdit.Password)
        self.hf = QtWidgets.QLineEdit(cfg.get("HF_TOKEN") or "")
        self.hf.setEchoMode(QtWidgets.QLineEdit.Password)

        self.openai_model = QtWidgets.QLineEdit(load_config().get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
        self.hf_name = QtWidgets.QLineEdit(load_config().get("HF_LLM_NAME") or os.environ.get("HF_LLM_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))

        self.embed_backend = QtWidgets.QComboBox()
        self.embed_backend.addItems(["hf","openai"])
        self.embed_backend.setCurrentText((cfg.get("EMBED_BACKEND") or "hf").lower())

        self.llm_backend = QtWidgets.QComboBox()
        self.llm_backend.addItems(["openai","vllm"])
        self.llm_backend.setCurrentText((cfg.get("LLM_BACKEND") or "openai").lower())

        self.mem_enable = QtWidgets.QCheckBox()
        self.mem_enable.setChecked(as_bool(prefs.get("MEMORY_ENABLED", cfg.get("MEMORY_ENABLED") or "0")))

        self.mem_tokens = QtWidgets.QSpinBox()
        self.mem_tokens.setRange(0, 100000)
        self.mem_tokens.setSingleStep(100)
        self.mem_tokens.setValue(int(prefs.get("MEMORY_TOKEN_LIMIT", cfg.get("MEMORY_TOKEN_LIMIT") or 1200)))

        self.mem_mb = QtWidgets.QSpinBox()
        self.mem_mb.setRange(1, 2048)
        self.mem_mb.setSingleStep(5)
        self.mem_mb.setValue(int(prefs.get("MEMORY_FILE_LIMIT_MB", cfg.get("MEMORY_FILE_LIMIT_MB") or 50)))

        self.openai_tpm = QtWidgets.QSpinBox()
        self.openai_tpm.setRange(0, 900000)
        self.openai_tpm.setSingleStep(10000)
        self.openai_tpm.setValue(int(prefs.get("OPENAI_TPM", os.environ.get("OPENAI_TPM", "0"))))

        self.openai_rpm = QtWidgets.QSpinBox()
        self.openai_rpm.setRange(0, 5000)
        self.openai_rpm.setSingleStep(100)
        self.openai_rpm.setValue(int(prefs.get("OPENAI_RPM", os.environ.get("OPENAI_RPM", "0"))))

        self.ask_time = QtWidgets.QSpinBox()
        self.ask_time.setRange(10, 18000)
        self.ask_time.setSingleStep(10)
        self.ask_time.setValue(int(prefs.get("ASK_TIME_BUDGET_SEC", os.environ.get("ASK_TIME_BUDGET_SEC", "120"))))

        self.ask_exhaustive = QtWidgets.QCheckBox()
        self.ask_exhaustive.setChecked(as_bool(prefs.get("ASK_EXHAUSTIVE", os.environ.get("ASK_EXHAUSTIVE","false"))))

        self.ask_candidates = QtWidgets.QSpinBox()
        self.ask_candidates.setRange(10, 5000)
        self.ask_candidates.setSingleStep(10)
        self.ask_candidates.setValue(int(prefs.get("ASK_CANDIDATES", os.environ.get("ASK_CANDIDATES","300"))))

        self.ask_reranker = QtWidgets.QComboBox()
        self.ask_reranker.addItems(["off","minilm","bge-m3","bge-base","bge-large"])
        self.ask_reranker.setCurrentText((prefs.get("ASK_RERANKER", os.environ.get("ASK_RERANKER","off"))).lower())

        self.ask_batch_chars = QtWidgets.QSpinBox()
        self.ask_batch_chars.setRange(2000, 60000)
        self.ask_batch_chars.setSingleStep(1000)
        self.ask_batch_chars.setValue(int(prefs.get("ASK_BATCH_CHAR_BUDGET", os.environ.get("ASK_BATCH_CHAR_BUDGET", "12000"))))

        self.ask_max_batches = QtWidgets.QSpinBox()
        self.ask_max_batches.setRange(1, 50)
        self.ask_max_batches.setSingleStep(1)
        self.ask_max_batches.setValue(int(prefs.get("ASK_MAX_BATCHES", os.environ.get("ASK_MAX_BATCHES", "6"))))

        for le in [self.openai, self.hf, self.openai_model, self.hf_name]:
            le.setClearButtonEnabled(True)
            le.setMinimumWidth(300)
            le.setDragEnabled(True)

        self.mem_tokens.setAccelerated(True)
        self.mem_mb.setAccelerated(True)
        self.openai_tpm.setAccelerated(True)
        self.openai_rpm.setAccelerated(True)
        self.ask_time.setAccelerated(True)
        self.ask_batch_chars.setAccelerated(True)
        self.ask_max_batches.setAccelerated(True)
        self.ask_candidates.setAccelerated(True)

        form.addRow("OPENAI_API_KEY", self.openai)
        form.addRow("HF_TOKEN", self.hf)
        form.addRow("OpenAI chat model", self.openai_model)
        form.addRow("HF LLM name", self.hf_name)
        form.addRow("Embedding backend", self.embed_backend)
        form.addRow("LLM backend", self.llm_backend)
        form.addRow("Memory enabled", self.mem_enable)
        form.addRow("Memory token limit", self.mem_tokens)
        form.addRow("Memory file limit (MB)", self.mem_mb)
        form.addRow("OPENAI_TPM", self.openai_tpm)
        form.addRow("OPENAI_RPM", self.openai_rpm)
        form.addRow("Ask time budget (sec)", self.ask_time)
        form.addRow("Ask batch char budget", self.ask_batch_chars)
        form.addRow("Ask max batches", self.ask_max_batches)
        form.addRow("Exhaustive sweep", self.ask_exhaustive)
        form.addRow("Candidate pool size", self.ask_candidates)
        form.addRow("Reranker", self.ask_reranker)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)
    def save(self):
        if self.openai.text().strip():
            save_secret("OPENAI_API_KEY", self.openai.text().strip(), prefer="keyring")
        if self.hf.text().strip():
            save_secret("HF_TOKEN", self.hf.text().strip(), prefer="keyring")
        save_secret("EMBED_BACKEND", self.embed_backend.currentText(), prefer="file")
        save_secret("LLM_BACKEND", self.llm_backend.currentText(), prefer="file")
        save_secret("OPENAI_CHAT_MODEL", self.openai_model.text().strip() or "gpt-4o-mini", prefer="file")
        save_secret("HF_LLM_NAME", self.hf_name.text().strip(), prefer="file")
        prefs = read_prefs()
        prefs["MEMORY_ENABLED"] = "true" if self.mem_enable.isChecked() else "false"
        prefs["MEMORY_TOKEN_LIMIT"] = int(self.mem_tokens.value())
        prefs["MEMORY_FILE_LIMIT_MB"] = int(self.mem_mb.value())
        prefs["OPENAI_TPM"] = int(self.openai_tpm.value())
        prefs["OPENAI_RPM"] = int(self.openai_rpm.value())
        prefs["ASK_TIME_BUDGET_SEC"] = int(self.ask_time.value())
        prefs["ASK_BATCH_CHAR_BUDGET"] = int(self.ask_batch_chars.value())
        prefs["ASK_MAX_BATCHES"] = int(self.ask_max_batches.value())
        prefs["ASK_EXHAUSTIVE"] = "true" if self.ask_exhaustive.isChecked() else "false"
        prefs["ASK_CANDIDATES"] = int(self.ask_candidates.value())
        prefs["ASK_RERANKER"] = self.ask_reranker.currentText()
        write_prefs(prefs)
        os.environ["ASK_EXHAUSTIVE"] = prefs["ASK_EXHAUSTIVE"]
        os.environ["ASK_CANDIDATES"] = str(prefs["ASK_CANDIDATES"])
        os.environ["ASK_RERANKER"] = prefs["ASK_RERANKER"]
        os.environ["MEMORY_ENABLED"] = prefs["MEMORY_ENABLED"]
        os.environ["MEMORY_TOKEN_LIMIT"] = str(prefs["MEMORY_TOKEN_LIMIT"])
        os.environ["OPENAI_CHAT_MODEL"] = self.openai_model.text().strip() or "gpt-4o-mini"
        os.environ["HF_LLM_NAME"] = self.hf_name.text().strip()
        os.environ["MEMORY_FILE_LIMIT_MB"] = str(prefs["MEMORY_FILE_LIMIT_MB"])
        os.environ["OPENAI_TPM"] = str(prefs["OPENAI_TPM"])
        os.environ["OPENAI_RPM"] = str(prefs["OPENAI_RPM"])
        os.environ["ASK_TIME_BUDGET_SEC"] = str(prefs["ASK_TIME_BUDGET_SEC"])
        os.environ["ASK_BATCH_CHAR_BUDGET"] = str(prefs["ASK_BATCH_CHAR_BUDGET"])
        os.environ["ASK_MAX_BATCHES"] = str(prefs["ASK_MAX_BATCHES"])
        self.accept()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1000, 720)

        tb = self.addToolBar("Main")
        act_settings = QtGui.QAction("Settings", self)
        act_settings.triggered.connect(self.open_settings)
        tb.addAction(act_settings)
        def do_clear():
            if QtWidgets.QMessageBox.question(self, "Clear index", "Delete all indexed chunks?") == QtWidgets.QMessageBox.Yes:
                clear_index()
                self.ingest_log.appendPlainText("\nIndex cleared.")
                self.status.showMessage("Index cleared")
        act_clear = QtGui.QAction("Clear Index", self)
        act_clear.triggered.connect(do_clear)
        tb.addAction(act_clear)

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.status_prog = QtWidgets.QProgressBar()
        self.status_prog.setRange(0, 100)
        self.status_prog.setValue(0)
        self.status_prog.setMaximumWidth(240)
        self.status_prog.hide()
        self.status.addPermanentWidget(self.status_prog)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        ingest = QtWidgets.QWidget()
        v1 = QtWidgets.QVBoxLayout(ingest)
        self.pick_btn = QtWidgets.QPushButton("Choose files…")
        self.pick_btn.clicked.connect(self.choose_files)
        self.ingest_btn = QtWidgets.QPushButton("Ingest")
        self.ingest_btn.clicked.connect(self.ingest_docs)
        self.ingest_btn.setEnabled(False)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.pick_btn)
        row.addWidget(self.ingest_btn)
        self.sel_label = QtWidgets.QLabel("No files selected.")
        self.ingest_log = QtWidgets.QPlainTextEdit()
        self.ingest_log.setReadOnly(True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0,0)
        v1.addLayout(row)
        v1.addWidget(self.sel_label)
        v1.addWidget(self.ingest_log)
        v1.addWidget(self.progress)
        self.progress.hide()

        ask = QtWidgets.QWidget()
        v2 = QtWidgets.QVBoxLayout(ask)

        self.input_bar = self._build_input_bar()

        self.k_spin = QtWidgets.QSpinBox()
        self.k_spin.setRange(1, 500)
        self.k_spin.setValue(10)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Top-k:"))
        controls.addWidget(self.k_spin)
        controls.addStretch(1)

        self.answer = QtWidgets.QTextEdit()
        self.answer.setReadOnly(True)

        v2.addWidget(self.input_bar)
        v2.addLayout(controls)
        v2.addWidget(self.answer)

        self.tabs.addTab(ingest, "Ingest")
        self.tabs.addTab(ask, "Ask")

        self.selected_paths = []
        self.ingest_worker = None
        self.ask_worker = None

        cfg = load_config()
        prefs = read_prefs()
        self.memory_enabled = as_bool(prefs.get("MEMORY_ENABLED", os.environ.get("MEMORY_ENABLED", cfg.get("MEMORY_ENABLED") or "0")))
        self.memory_token_limit = int(prefs.get("MEMORY_TOKEN_LIMIT", os.environ.get("MEMORY_TOKEN_LIMIT", cfg.get("MEMORY_TOKEN_LIMIT") or 1200)))
        self.memory_file_limit_mb = int(prefs.get("MEMORY_FILE_LIMIT_MB", os.environ.get("MEMORY_FILE_LIMIT_MB", cfg.get("MEMORY_FILE_LIMIT_MB") or 50)))
        self.history = mem.load_recent(self.memory_token_limit) if self.memory_enabled else []
        self.history_limit = 1000
        self.adjust_input_height()
        self.update_key_status()
        
    def _build_input_bar(self):
        wrap = QtWidgets.QFrame()
        wrap.setObjectName("InputWrap")
        wrap.setFrameShape(QtWidgets.QFrame.StyledPanel)
        wrap.setLineWidth(1)
        wrap.setStyleSheet("""
    #InputWrap { border: 1px solid rgba(255,255,255,0.18); background: palette(base); }
    QToolButton#AskInBar { border: 1px solid rgba(255,255,255,0.18); border-radius: 4px; padding: 0; min-width: 28px; min-height: 24px; }
    """)

        lay = QtWidgets.QHBoxLayout(wrap)
        lay.setContentsMargins(8, 6, 6, 6)
        lay.setSpacing(6)

        self.q_edit = QtWidgets.QTextEdit()
        self.q_edit.setObjectName("QEdit")
        self.q_edit.setFrameStyle(QtWidgets.QFrame.NoFrame)
        self.q_edit.setPlaceholderText("Ask anything about your readings…")
        self.q_edit.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.q_edit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.q_edit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.q_edit.installEventFilter(self)

        self.ask_btn_inbar = QtWidgets.QToolButton()
        self.ask_btn_inbar.setObjectName("AskInBar")
        self.ask_btn_inbar.setText("↑")
        self.ask_btn_inbar.setCursor(QtCore.Qt.PointingHandCursor)
        self.ask_btn_inbar.clicked.connect(self.run_ask)
        self.ask_btn_inbar.setToolTip("Ask")

        lay.addWidget(self.q_edit, 1)
        lay.addWidget(self.ask_btn_inbar, 0)

        self.q_min_lines, self.q_max_lines = 1, 5
        self._last_q_height = 0
        self.q_resize_timer = QtCore.QTimer(self)
        self.q_resize_timer.setSingleShot(True)
        self.q_resize_timer.setInterval(15)
        self.q_resize_timer.timeout.connect(self.adjust_input_height)
        self.q_edit.textChanged.connect(lambda: self.q_resize_timer.start())
        self.q_edit.document().documentLayout().documentSizeChanged.connect(lambda _: self.q_resize_timer.start())

        return wrap


    def _position_send_button(self):
        r = self.q_edit.viewport().rect()
        m = 6
        x = r.right() - m - self.ask_btn_inbar.width()
        y = r.bottom() - m - self.ask_btn_inbar.height()
        self.ask_btn_inbar.move(x, y)

    def eventFilter(self, obj, event):
        if obj is self.q_edit and event.type() == QtCore.QEvent.KeyPress:
            if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                if event.modifiers() & QtCore.Qt.ShiftModifier:
                    return False
                self.run_ask()
                return True
        return super().eventFilter(obj, event)

    def adjust_input_height(self):
        fm = self.q_edit.fontMetrics()
        line_h = fm.lineSpacing()
        min_h = self.q_min_lines * line_h + 12
        max_h = self.q_max_lines * line_h + 12
        doc_h = int(self.q_edit.document().documentLayout().documentSize().height()) + 8
        h = max(min_h, min(doc_h, max_h))
        if abs(h - getattr(self, "_last_q_height", 0)) > 1:
            self._last_q_height = h
            self.q_edit.setFixedHeight(h)
            self.input_bar.setFixedHeight(h + 12)


    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            cfg = load_config()
            prefs = read_prefs()
            self.memory_enabled = as_bool(prefs.get("MEMORY_ENABLED", os.environ.get("MEMORY_ENABLED", cfg.get("MEMORY_ENABLED") or "0")))
            self.memory_token_limit = int(prefs.get("MEMORY_TOKEN_LIMIT", os.environ.get("MEMORY_TOKEN_LIMIT", cfg.get("MEMORY_TOKEN_LIMIT") or 1200)))
            self.memory_file_limit_mb = int(prefs.get("MEMORY_FILE_LIMIT_MB", os.environ.get("MEMORY_FILE_LIMIT_MB", cfg.get("MEMORY_FILE_LIMIT_MB") or 50)))
            self.history = mem.load_recent(self.memory_token_limit) if self.memory_enabled else []
            self.update_key_status()

    def update_key_status(self):
        status = present_keys()
        bits = []
        bits.append(f"OPENAI: {'✓' if status.get('OPENAI_API_KEY') else '×'}")
        bits.append(f"HF: {'✓' if status.get('HF_TOKEN') else '×'}")
        bits.append(f"Memory: {'on' if self.memory_enabled else 'off'}")
        self.status.showMessage(" | ".join(bits))

    def choose_files(self):
        fns, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose files", "", "PDF and Images (*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)")
        if not fns:
            return
        self.selected_paths = [pathlib.Path(fn) for fn in fns]
        names = ", ".join(p.name for p in self.selected_paths[:3])
        more = "" if len(self.selected_paths) <= 3 else f" …+{len(self.selected_paths)-3} more"
        self.sel_label.setText(f"Selected: {names}{more}")
        self.ingest_btn.setEnabled(True)

    def ingest_docs(self):
        if not self.selected_paths:
            return
        self.ingest_log.clear()
        self.status_prog.show()
        self.status_prog.setValue(0)
        self.ingest_worker = IngestWorker(self.selected_paths)
        self.ingest_worker.progress.connect(self.ingest_log.appendPlainText)
        self.ingest_worker.progress_pct.connect(self.status_prog.setValue)
        self.ingest_worker.finished.connect(self.ingest_done)
        self.ingest_worker.failed.connect(self.ingest_fail)
        self.ingest_worker.start()

    def ingest_done(self, info):
        self.progress.hide()
        self.ingest_btn.setEnabled(True)
        self.pick_btn.setEnabled(True)
        self.status_prog.hide()
        for d in info.get("details", []):
            self.ingest_log.appendPlainText(f"\n{pathlib.Path(d['file']).name}\nPages: {d['num_pages']} (OCR: {d['ocr_pages']})\nChunks: {d['num_chunks']}")
            op = d.get("ocr_page_numbers") or []
            sp = d.get("suspect_pages") or []
            sv = d.get("salvaged_pages") or []
            if sv:
                head = ", ".join(str(x) for x in sv[:20])
                tail = " ..." if len(sv) > 20 else ""
                self.ingest_log.appendPlainText(f"Salvaged pages: [{head}]{tail}")
            else:
                self.ingest_log.appendPlainText("Salvaged pages: []")
            if op:
                head = ", ".join(str(x) for x in op[:20])
                tail = " ..." if len(op) > 20 else ""
                self.ingest_log.appendPlainText(f"OCR pages: [{head}]{tail}")
            else:
                self.ingest_log.appendPlainText("OCR pages: []")
            if sp:
                head = ", ".join(str(x) for x in sp[:20])
                tail = " ..." if len(sp) > 20 else ""
                self.ingest_log.appendPlainText(f"Suspect pages: [{head}]{tail}")
            else:
                self.ingest_log.appendPlainText("Suspect pages: []")
        self.status.showMessage(f"Ingested {info.get('num_docs',0)} file(s) — chunks: {info.get('num_chunks',0)}")
        md = "### Auto-summary\n\n" + info.get("doc_summary","")
        render_markdown(self.answer, md)
        self.tabs.setCurrentIndex(1)

    def ingest_fail(self, msg):
        self.progress.hide()
        self.ingest_btn.setEnabled(True)
        self.pick_btn.setEnabled(True)
        self.ingest_log.appendPlainText("\n[ERROR]\n" + msg)
        self.status_prog.hide()
        QtWidgets.QMessageBox.critical(self, "Ingest failed", msg)

    def run_ask(self):
        q = self.q_edit.toPlainText().strip()
        if not q:
            QtWidgets.QMessageBox.information(self, "Ask", "Type a question first.")
            return
        render_markdown(self.answer, "_Working…_")
        self.ask_btn_inbar.setEnabled(False)
        self.status_prog.show()
        self.status_prog.setValue(0)
        self.ask_worker = AskWorker(q, self.k_spin.value(), history=self.history)
        self.ask_worker.progress.connect(lambda s: self.status.showMessage(s))
        self.ask_worker.progress_pct.connect(self.status_prog.setValue)
        self.ask_worker.finished.connect(self.ask_done)
        self.ask_worker.failed.connect(self.ask_fail)
        self.ask_worker.start()

    def ask_done(self, out):
        self.ask_btn_inbar.setEnabled(True)
        text = out.get("answer", "")
        cites = ", ".join(out.get("citations", []))
        quotes = out.get("quotes", [])
        qmd = ""
        if quotes:
            qmd = "\n\n### Evidence snippets\n" + "\n".join([f"> {q['quote']}\n>\n> — {q['source']}" for q in quotes])
        render_markdown(self.answer, text + "\n\n**Citations:** " + cites + qmd)
        q = self.q_edit.toPlainText().strip()
        if self.memory_enabled and q and text:
            mem.append_turn(q, text)
            mem.prune_file(self.memory_file_limit_mb)
            self.history = mem.load_recent(self.memory_token_limit)
        self.status.showMessage("Answer ready")
        self.status_prog.hide()

    def ask_fail(self, msg):
        self.ask_btn_inbar.setEnabled(True)
        render_markdown(self.answer, "**ERROR**\n\n```\n" + msg + "\n```")
        QtWidgets.QMessageBox.critical(self, "Ask failed", msg)
        self.status_prog.hide()

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
