import sys, pathlib, traceback, json, os, shutil, re
from PySide6 import QtWidgets, QtCore, QtGui

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_icon(name: str, fallback: QtGui.QIcon | None = None) -> QtGui.QIcon:
    try:
        if "." in name:
            fnames = [name]
        else:
            fnames = [f"{name}.icns", f"{name}.png", f"{name}.ico", f"{name}.svg"]
        roots = [
            ROOT / "assets",
            ROOT / "app" / "assets",
            pathlib.Path.cwd() / "assets"
        ]
        for root in roots:
            for fn in fnames:
                p = root / fn
                if p.exists():
                    return QtGui.QIcon(str(p))
    except Exception:
        pass
    return fallback or QtGui.QIcon()

try:
    import markdown as mdlib
except Exception:
    mdlib = None

from api.services.parse import parse_any_bytes
from api.services.chunk import chunk_pages
from api.services.index import add_chunks, search, clear_index
from api.services.summarize import summarize, summarize_document, summarizer_info, summarize_batched, summarize_all_formulas, is_formula_query
from api.services import agent
from api.services.embed import embedding_info
from api.services import memory as mem
from api.services import aliases
from api.core.config import save_secret, present_keys, load_config


APP_TITLE = "ANAGNOSIS"
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
        pal = widget.palette()
        win = pal.color(QtGui.QPalette.Window)
        txt = pal.color(QtGui.QPalette.WindowText)
        lum = int(0.299 * win.red() + 0.587 * win.green() + 0.114 * win.blue())
        dark = lum < 140
        if dark:
            text_color = "#dde2ea"
            border = "rgba(255,255,255,0.13)"
            link = "#6ab3fa"
        else:
            text_color = "#23272d"
            border = "#ddd"
            link = "#0366d6"
        css = (
            "<style>"
            f"body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.45;padding:8px;color:{text_color};background:transparent;}}"
            "h1,h2,h3{margin-top:1em}"
            "code{background:transparent;padding:2px 4px;border-radius:4px}"
            "pre{background:transparent;padding:8px;border-radius:8px;overflow:auto}"
            "ul{margin-left:1.2em}"
            f"blockquote{{border-left:3px solid {border};padding-left:8px;color:{'#8ca0b3' if dark else '#555'}}}"
            "table{border-collapse:collapse}"
            f"th,td{{border:1px solid {border};padding:4px 8px}}"
            f"a{{color:{link};text-decoration:none}}"
            f"a:hover{{text-decoration:underline}}"
            "</style>"
        )
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

    def _check_cancel(self):
        if self.isInterruptionRequested():
            raise RuntimeError("Cancelled")

    def run(self):
        try:
            self._check_cancel()
            summaries, details, total_chunks = [], [], 0
            emb = embedding_info(); self.progress.emit(f"Embedding: {emb['backend']} ({emb['model']})")
            sumi = summarizer_info(); self.progress.emit(f"Summarizer: {sumi['backend']}")
            for p in self.paths:
                self._check_cancel()
                self.progress.emit(f"Loading {p.name}…")
                self.progress_pct.emit(2)
                data = p.read_bytes()

                self.progress.emit("Parsing document…")
                parsed = parse_any_bytes(p.name, data, progress_cb=self.progress.emit)
                self._check_cancel()
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

                self._check_cancel()
                self.progress.emit("Chunking…")
                self._check_cancel()
                chunks = chunk_pages(parsed["pages"])
                self._check_cancel()
                self.progress_pct.emit(35)

                def embed_cb(done, total):
                    if self.isInterruptionRequested():
                        raise RuntimeError("Cancelled")
                    base = 35
                    span = 55
                    pct = base + int(span * (done / max(1, total)))
                    pct = min(95, max(36, pct))
                    self.progress_pct.emit(pct)

                self.progress.emit("Embedding and indexing…")
                self._check_cancel()
                add_chunks(chunks, progress_cb=embed_cb)
                self._check_cancel()
                self.progress_pct.emit(96)

                total_chunks += len(chunks)

                self.progress.emit("Summarizing…")
                self._check_cancel()
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

    def __init__(self, question, k, history, formula_mode=False, agents_enabled=False):
        super().__init__()
        self.question = question
        self.k = k
        self.history = history
        self.formula_mode = formula_mode
        self.agents_enabled = agents_enabled
    
    def _check_cancel(self):
        if self.isInterruptionRequested():
            raise RuntimeError("Cancelled")

    def run(self):
        try:
            self._check_cancel()
            self.progress.emit("Searching index…")
            self.progress_pct.emit(10)

            base_pool = int(os.environ.get("ASK_CANDIDATES","300"))
            pool = int(os.environ.get("ASK_CANDIDATES_FORMULA","3000")) if self.formula_mode else base_pool
            self._pc = 12
            def s_cb(msg):
                self.progress.emit(msg)
                self._pc = min(55, self._pc + 3)
                self.progress_pct.emit(self._pc)

            tb_base = int(os.environ.get("ASK_TIME_BUDGET_SEC","120"))
            tb = int(os.environ.get("ASK_TIME_BUDGET_SEC_FORMULA","240")) if self.formula_mode else tb_base
            st_base = max(10, min(tb_base//2, 30))
            st = int(os.environ.get("SEARCH_TIMEOUT_SEC", str(st_base)))
            if self.formula_mode:
                st = int(os.environ.get("SEARCH_TIMEOUT_SEC_FORMULA", str(max(20, min(tb//2, 60)))))

            self._check_cancel()
            hits = search(self.question, k=pool, progress_cb=s_cb, timeout_sec=st, pool=pool)
            self._check_cancel()
            self.progress.emit(f"Hits: {len(hits)}")
            if not hits:
                self.progress_pct.emit(100)
                self.finished.emit({"answer": "**No results. Ingest a document first.**", "citations": []})
                return

            top_chunks = [h[1] for h in hits]

            if self.formula_mode:
                self._check_cancel()
                self.progress.emit("Formula mode: extracting all formulas in scope…")
                total = len(top_chunks)
                if total < pool:
                    self.progress.emit(f"Warning: only {total} / {pool} chunks returned; consider increasing index size or candidate pool.")
                def p_cb(msg):
                    if self.isInterruptionRequested():
                        raise RuntimeError("Cancelled")
                    self.progress.emit(msg)
                    m = re.search(r"(\d+)\s*/\s*(\d+)", msg)
                    if m:
                        done = int(m.group(1))
                        pct = 20 + int(70 * (done / max(1, total)))
                        pct = max(20, min(95, pct))
                        self.progress_pct.emit(pct)
                out = summarize_all_formulas(self.question, top_chunks, progress_cb=p_cb)
                if self.agents_enabled:
                    self._check_cancel()
                    try:
                        self.progress.emit("Agents: verifying…")
                        agent_budget = int(os.environ.get("AGENT_VERIFY_BUDGET", "30"))
                        _prev = out if isinstance(out, dict) else {"answer": str(out)}
                        _res = agent.verify_answer(self.question, out, hits, time_budget_sec=agent_budget)
                        changed = False
                        try:
                            changed = (_res.get("answer", "") != _prev.get("answer", ""))
                        except Exception:
                            changed = False
                        if isinstance(_res, dict):
                            meta = _res.get("agent_meta", {})
                            meta.update({"enabled": True, "changed": bool(changed)})
                            _res["agent_meta"] = meta
                        out = _res
                        self.progress.emit("Agents: verdict — " + ("modified" if changed else "validated"))
                    except Exception:
                        self.progress.emit("Agent verification skipped (error).")
                    self._check_cancel()
            else:
                self._check_cancel()
                self.progress.emit("Summarizing with context…")
                self.progress_pct.emit(60)
                mb = int(os.environ.get("ASK_MAX_BATCHES","6"))
                exh = os.environ.get("ASK_EXHAUSTIVE","false").lower() in {"1","true","yes","on"}
                chs = []
                for h in hits:
                    try:
                        sc, ch = h[0], h[1]
                    except Exception:
                        sc, ch = 0.0, h[1] if isinstance(h, (list, tuple)) and len(h) > 1 else h
                    x = dict(ch); x["_score"] = float(sc) if sc is not None else 0.0
                    chs.append(x)
                self._check_cancel()
                out = summarize_batched(self.question, chs, history=self.history, progress_cb=lambda s: self.progress.emit(s), max_batches=mb, time_budget_sec=tb, exhaustive=exh)
                if self.agents_enabled:
                    self._check_cancel()
                    try:
                        self.progress.emit("Agents: verifying…")
                        agent_budget = int(os.environ.get("AGENT_VERIFY_BUDGET", "30"))
                        _prev = out if isinstance(out, dict) else {"answer": str(out)}
                        _res = agent.verify_answer(self.question, out, hits, time_budget_sec=agent_budget)
                        changed = False
                        try:
                            changed = (_res.get("answer", "") != _prev.get("answer", ""))
                        except Exception:
                            changed = False
                        if isinstance(_res, dict):
                            meta = _res.get("agent_meta", {})
                            meta.update({"enabled": True, "changed": bool(changed)})
                            _res["agent_meta"] = meta
                        out = _res
                        self.progress.emit("Agents: verdict — " + ("modified" if changed else "validated"))
                    except Exception:
                        self.progress.emit("Agent verification skipped (error).")
                    self._check_cancel()

            try:
                if isinstance(out, dict) and "citations" in out:
                    out["citations"] = sorted(set(out.get("citations", [])))
            except Exception:
                pass

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
        self.ask_batch_chars.setAccelerated(True)
        self.ask_max_batches.setAccelerated(True)

        tabs = QtWidgets.QTabWidget()

        keys_w = QtWidgets.QWidget()
        keys_form = QtWidgets.QFormLayout(keys_w)
        keys_form.addRow("OPENAI_API_KEY", self.openai)
        keys_form.addRow("HF_TOKEN", self.hf)
        tabs.addTab(keys_w, "Keys")

        models_w = QtWidgets.QWidget()
        models_form = QtWidgets.QFormLayout(models_w)
        models_form.addRow("OpenAI chat model", self.openai_model)
        models_form.addRow("HF LLM name", self.hf_name)
        models_form.addRow("Embedding backend", self.embed_backend)
        models_form.addRow("LLM backend", self.llm_backend)
        tabs.addTab(models_w, "Models")

        mem_w = QtWidgets.QWidget()
        mem_form = QtWidgets.QFormLayout(mem_w)
        mem_form.addRow("Memory enabled", self.mem_enable)
        mem_form.addRow("Memory token limit", self.mem_tokens)
        mem_form.addRow("Memory file limit (MB)", self.mem_mb)
        tabs.addTab(mem_w, "Memory")

        limits_w = QtWidgets.QWidget()
        limits_form = QtWidgets.QFormLayout(limits_w)
        limits_form.addRow("OPENAI_TPM", self.openai_tpm)
        limits_form.addRow("OPENAI_RPM", self.openai_rpm)
        limits_form.addRow("Ask batch char budget", self.ask_batch_chars)
        limits_form.addRow("Ask max batches", self.ask_max_batches)
        tabs.addTab(limits_w, "Budgets")

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(tabs)
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
        prefs["ASK_BATCH_CHAR_BUDGET"] = int(self.ask_batch_chars.value())
        prefs["ASK_MAX_BATCHES"] = int(self.ask_max_batches.value())
        write_prefs(prefs)
        os.environ["MEMORY_ENABLED"] = prefs["MEMORY_ENABLED"]
        os.environ["MEMORY_TOKEN_LIMIT"] = str(prefs["MEMORY_TOKEN_LIMIT"])
        os.environ["OPENAI_CHAT_MODEL"] = self.openai_model.text().strip() or "gpt-4o-mini"
        os.environ["HF_LLM_NAME"] = self.hf_name.text().strip()
        os.environ["MEMORY_FILE_LIMIT_MB"] = str(prefs["MEMORY_FILE_LIMIT_MB"])
        os.environ["OPENAI_TPM"] = str(prefs["OPENAI_TPM"])
        os.environ["OPENAI_RPM"] = str(prefs["OPENAI_RPM"])
        os.environ["ASK_BATCH_CHAR_BUDGET"] = str(prefs["ASK_BATCH_CHAR_BUDGET"])
        os.environ["ASK_MAX_BATCHES"] = str(prefs["ASK_MAX_BATCHES"])
        self.accept()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1000, 720)
        app_icon = load_icon('icon.png')
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)
        cfg = load_config()
        prefs = read_prefs()
        self.memory_enabled = as_bool(prefs.get("MEMORY_ENABLED", os.environ.get("MEMORY_ENABLED", cfg.get("MEMORY_ENABLED") or "0")))
        self.memory_token_limit = int(prefs.get("MEMORY_TOKEN_LIMIT", os.environ.get("MEMORY_TOKEN_LIMIT", cfg.get("MEMORY_TOKEN_LIMIT") or 1200)))
        self.memory_file_limit_mb = int(prefs.get("MEMORY_FILE_LIMIT_MB", os.environ.get("MEMORY_FILE_LIMIT_MB", cfg.get("MEMORY_FILE_LIMIT_MB") or 50)))
        self.history = mem.load_recent(self.memory_token_limit) if self.memory_enabled else []
        self.history_limit = 1000

        tb = self.addToolBar("Main")
        brand = QtWidgets.QWidget()
        b_lay = QtWidgets.QHBoxLayout(brand); b_lay.setContentsMargins(6,0,12,0); b_lay.setSpacing(6)
        ic = load_icon('icon') or load_icon('icon.png')
        ic_lbl = QtWidgets.QLabel()
        if not ic.isNull():
            ic_lbl.setPixmap(ic.pixmap(18, 18))
        title_lbl = QtWidgets.QLabel(APP_TITLE)
        title_lbl.setStyleSheet('font-weight:600; letter-spacing:0.5px;')
        b_lay.addWidget(ic_lbl); b_lay.addWidget(title_lbl)
        tb.addWidget(brand)
        act_settings = QtGui.QAction("Settings", self)
        act_settings.triggered.connect(self.open_settings)
        tb.addAction(act_settings)

        act_shortcuts = QtGui.QAction("Keyboard Shortcuts", self)
        shortcut_str = "Meta+/" if sys.platform == "darwin" else "Ctrl+/"
        act_shortcuts.setShortcut(QtGui.QKeySequence(shortcut_str))
        act_shortcuts.triggered.connect(self.show_shortcuts)
        tb.addAction(act_shortcuts)
        def do_clear():
            if QtWidgets.QMessageBox.question(self, "Clear index", "Delete all indexed chunks?") == QtWidgets.QMessageBox.Yes:
                clear_index()
                self.ingest_log.appendPlainText("\nIndex cleared.")
                self.status.showMessage("Index cleared")
        def do_clear_cache():
            if QtWidgets.QMessageBox.question(self, "Clear cache", "Delete all __pycache__ folders and .pyc files under the project?") == QtWidgets.QMessageBox.Yes:
                n_dirs = 0
                n_files = 0
                try:
                    for root, dirs, files in os.walk(str(ROOT)):
                        for d in list(dirs):
                            if d == "__pycache__":
                                p = os.path.join(root, d)
                                shutil.rmtree(p, ignore_errors=True)
                                n_dirs += 1
                        for f in files:
                            if f.endswith(".pyc") or f.endswith(".pyo"):
                                p = os.path.join(root, f)
                                try:
                                    os.remove(p)
                                    n_files += 1
                                except Exception:
                                    pass
                    self.ingest_log.appendPlainText(f"\nCache cleared. Removed {n_dirs} __pycache__ dirs and {n_files} bytecode files.")
                    self.status.showMessage("Cache cleared")
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Clear cache", str(e))

        def do_clear_memory():
            if QtWidgets.QMessageBox.question(self, "Clear memory", "Erase saved Q/A memory on disk and reset in-app history?") != QtWidgets.QMessageBox.Yes:
                return
            try:
                self.history = []
                try:
                    if hasattr(mem, "clear"):
                        mem.clear()
                        cleared = True
                    else:
                        cleared = False
                except Exception:
                    cleared = False
                msg = "Memory cleared" if cleared else "Couldn't clear memory."
                self.ingest_log.appendPlainText("\n" + msg)
                self.status.showMessage(msg)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Clear memory", str(e))

        act_cache = QtGui.QAction("Clear Cache", self)
        act_cache.triggered.connect(do_clear_cache)
        tb.addAction(act_cache)

        act_mem = QtGui.QAction("Clear Memory", self)
        act_mem.triggered.connect(do_clear_memory)
        tb.addAction(act_mem)
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

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.hide()
        self.cancel_btn.clicked.connect(self.cancel_current)
        self.status.addPermanentWidget(self.cancel_btn)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabs.setMovable(True)
        self.tabs.setElideMode(QtCore.Qt.ElideRight)
        self._apply_tab_style()
        self.setCentralWidget(self.tabs)

        ingest = QtWidgets.QWidget()
        v1 = QtWidgets.QVBoxLayout(ingest)
        self.pick_btn = QtWidgets.QPushButton("Choose files…")
        self.pick_btn.clicked.connect(self.choose_files)
        self.pick_btn.setFocusPolicy(QtCore.Qt.NoFocus)
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

        self.formula_cb = QtWidgets.QCheckBox("Formula mode")
        prefs = read_prefs()
        self.formula_cb.setChecked(as_bool(prefs.get("ASK_FORMULA_FORCE", os.environ.get("ASK_FORMULA_FORCE", "0"))))

        self.strict_cb = QtWidgets.QCheckBox("Only this document")
        self.strict_cb.setChecked(as_bool(prefs.get("ASK_STRICT_DOC", os.environ.get("ASK_STRICT_DOC", "0"))))

        self.formula_cb.setToolTip("Extract canonical formulas only; uses the SFT classifier and stricter filtering.")
        self.strict_cb.setToolTip("Restrict retrieval to the currently referenced PDF only.")

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self.strict_cb)
        controls.addWidget(self.formula_cb)
        controls.addStretch(1)

        self.quick_box = QtWidgets.QGroupBox("Quick settings")
        self.quick_box.setFlat(True)
        self.quick_box.setStyleSheet(
            """
            QGroupBox { border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; margin-top: 18px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 2px 6px; background: transparent; }
            """
        )
        qv = as_bool(read_prefs().get("UI_QUICK_VISIBLE", "true"))
        self.quick_box.setVisible(qv)
        quick_layout = QtWidgets.QHBoxLayout(self.quick_box)
        quick_layout.setContentsMargins(8, 8, 8, 8)

        quick_layout.addWidget(QtWidgets.QLabel("Top-k:"))
        quick_layout.addWidget(self.k_spin)
        quick_layout.addSpacing(12)

        self.reranker_combo = QtWidgets.QComboBox()
        self.reranker_combo.addItems(["off","minilm","bge-m3","bge-base","bge-large"])
        self.reranker_combo.setCurrentText((prefs.get("ASK_RERANKER", os.environ.get("ASK_RERANKER","off")).lower()))
        self.reranker_combo.setToolTip("Reorder retrieved chunks using a cross-encoder reranker.")
        quick_layout.addWidget(QtWidgets.QLabel("Reranker:"))
        quick_layout.addWidget(self.reranker_combo)
        quick_layout.addSpacing(12)

        self.pool_spin = QtWidgets.QSpinBox()
        self.pool_spin.setRange(10, 5000)
        self.pool_spin.setSingleStep(10)
        self.pool_spin.setValue(int(prefs.get("ASK_CANDIDATES", os.environ.get("ASK_CANDIDATES","300"))))
        self.pool_spin.setToolTip("Number of candidate chunks retrieved before reranking/filtering.")
        quick_layout.addWidget(QtWidgets.QLabel("Pool:"))
        quick_layout.addWidget(self.pool_spin)
        quick_layout.addSpacing(12)

        self.time_spin = QtWidgets.QSpinBox()
        self.time_spin.setRange(10, 18000)
        self.time_spin.setSingleStep(10)
        self.time_spin.setValue(int(prefs.get("ASK_TIME_BUDGET_SEC", os.environ.get("ASK_TIME_BUDGET_SEC","120"))))
        self.time_spin.setToolTip("Overall time budget for answering (seconds).")
        quick_layout.addWidget(QtWidgets.QLabel("Time (s):"))
        quick_layout.addWidget(self.time_spin)
        quick_layout.addSpacing(12)

        self.exh_cb = QtWidgets.QCheckBox("Exhaustive sweep")
        self.exh_cb.setChecked(as_bool(prefs.get("ASK_EXHAUSTIVE", os.environ.get("ASK_EXHAUSTIVE","false"))))
        self.exh_cb.setToolTip("Try additional batches until the time budget is consumed.")
        quick_layout.addWidget(self.exh_cb)

        self.mem_cb = QtWidgets.QCheckBox("Memory")
        self.mem_cb.setChecked(self.memory_enabled)
        self.mem_cb.setToolTip("Persist conversation history and use it as context")
        quick_layout.addSpacing(12)
        quick_layout.addWidget(self.mem_cb)

        self.agents_cb = QtWidgets.QCheckBox("Agents")
        self.agents_cb.setChecked(as_bool(prefs.get("ASK_AGENTS", os.environ.get("ASK_AGENTS", "false"))))
        self.agents_cb.setToolTip("Enable agentic verification and reasoning (slower)")
        quick_layout.addSpacing(12)
        quick_layout.addWidget(self.agents_cb)


        quick_layout.addStretch(1)

        self.reranker_combo.currentTextChanged.connect(self._persist_quick_prefs)
        self.pool_spin.valueChanged.connect(lambda _: self._persist_quick_prefs())
        self.time_spin.valueChanged.connect(lambda _: self._persist_quick_prefs())
        self.exh_cb.toggled.connect(lambda _: self._persist_quick_prefs())
        self.mem_cb.toggled.connect(self._toggle_memory)
        self.agents_cb.toggled.connect(lambda _: self._persist_quick_prefs())

        answer_toolbar = QtWidgets.QWidget()
        answer_toolbar_lay = QtWidgets.QHBoxLayout(answer_toolbar)
        answer_toolbar_lay.setContentsMargins(0, 0, 0, 0)
        answer_toolbar_lay.setSpacing(4)
        btn_copy = QtWidgets.QToolButton()
        btn_copy.setIcon(load_icon('copy.png', self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)))
        btn_copy.setToolTip("Copy answer")
        btn_copy.clicked.connect(self.copy_answer)
        btn_save = QtWidgets.QToolButton()
        btn_save.setIcon(load_icon('save.png', self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)))
        btn_save.setToolTip("Save answer")
        btn_save.clicked.connect(self.save_answer)
        btn_open = QtWidgets.QToolButton()
        btn_open.setIcon(load_icon('open.png', self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)))
        btn_open.setToolTip("Open answer")
        btn_open.clicked.connect(self.open_answer)
        btn_clear = QtWidgets.QToolButton()
        btn_clear.setIcon(load_icon('clear.png', self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)))
        btn_clear.setToolTip("Clear answer")
        btn_clear.clicked.connect(self.clear_answer)
        self.state_chip = QtWidgets.QLabel("")
        self.state_chip.setStyleSheet("QLabel { border-radius: 8px; background: palette(midlight); padding: 2px 8px; margin-left: 6px; font-size: 10pt; }")
        answer_toolbar_lay.addWidget(btn_copy)
        answer_toolbar_lay.addWidget(btn_save)
        answer_toolbar_lay.addWidget(btn_open)
        answer_toolbar_lay.addWidget(btn_clear)
        answer_toolbar_lay.addStretch(1)
        answer_toolbar_lay.addWidget(self.state_chip)

        self.answer = QtWidgets.QTextBrowser()
        self.answer.setOpenExternalLinks(True)
        self.answer.setReadOnly(True)

        header = QtWidgets.QWidget()
        header_v = QtWidgets.QVBoxLayout(header)
        header_v.setContentsMargins(0, 0, 0, 0)
        header_v.setSpacing(6)
        header_v.addWidget(self.input_bar)
        header_v.addLayout(controls)
        header_v.addWidget(self.quick_box)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle{background: transparent;}")
        splitter.addWidget(header)
        answer_wrap = QtWidgets.QWidget()
        answer_wrap_lay = QtWidgets.QVBoxLayout(answer_wrap)
        answer_wrap_lay.setContentsMargins(0,0,0,0)
        answer_wrap_lay.setSpacing(2)
        answer_toolbar_lay.setSpacing(4)
        answer_wrap_lay.addWidget(answer_toolbar)
        answer_wrap_lay.addWidget(self.answer)
        splitter.addWidget(answer_wrap)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        v2.addWidget(splitter)

        ing_icon = load_icon('ingest.png', self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        ask_icon = load_icon('ask.png', self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion))
        self.tabs.addTab(ingest, "Ingest")
        self.tabs.addTab(ask, "Ask")
        tab_idx = int(read_prefs().get("UI_TAB_INDEX", 1))
        self.tabs.setCurrentIndex(tab_idx if 0 <= tab_idx < self.tabs.count() else 1)
        self.tabs.currentChanged.connect(self._persist_tab_index)

        self.selected_paths = []
        self.ingest_worker = None
        self.ask_worker = None

        act_qs = QtGui.QAction("Quick Settings", self)
        act_qs.setCheckable(True)
        act_qs.setChecked(qv)
        qs_shortcut = QtGui.QKeySequence("Meta+Shift+Q" if sys.platform == "darwin" else "Ctrl+Shift+Q")
        act_qs.setShortcut(qs_shortcut)
        act_qs.triggered.connect(self.toggle_quick_settings)
        tb.addAction(act_qs)
        self._act_quick_settings = act_qs

        focus_act = QtGui.QAction("Focus Ask", self)
        focus_act.setShortcut(QtGui.QKeySequence("Meta+L" if sys.platform == "darwin" else "Ctrl+L"))
        focus_act.triggered.connect(lambda: self.q_edit.setFocus())
        self.addAction(focus_act)

        self.adjust_input_height()
        self.update_key_status()

        esc_act = QtGui.QAction("Cancel", self)
        esc_act.setShortcut(QtGui.QKeySequence("Escape"))
        esc_act.triggered.connect(self.cancel_current)
        self.addAction(esc_act)
        ctrl_dot = "Meta+." if sys.platform == "darwin" else "Ctrl+."
        ctrl_dot_act = QtGui.QAction("Cancel (Ctrl+.)", self)
        ctrl_dot_act.setShortcut(QtGui.QKeySequence(ctrl_dot))
        ctrl_dot_act.triggered.connect(self.cancel_current)
        self.addAction(ctrl_dot_act)

    def toggle_quick_settings(self):
        vis = not self.quick_box.isVisible()
        self.quick_box.setVisible(vis)
        if hasattr(self, "_act_quick_settings"):
            self._act_quick_settings.setChecked(vis)
        prefs = read_prefs()
        prefs["UI_QUICK_VISIBLE"] = "true" if vis else "false"
        write_prefs(prefs)
    def _apply_tab_style(self):
        pal = self.palette()
        win = pal.color(QtGui.QPalette.Window)
        txt_win = pal.color(QtGui.QPalette.WindowText)
        sel_bg = win.name()
        sel_txt = txt_win.name()

        lum = int(0.299 * win.red() + 0.587 * win.green() + 0.114 * win.blue())
        dark_mode = lum < 140
        unsel = QtGui.QColor(win)
        hover = QtGui.QColor(win)
        if dark_mode:
            unsel = unsel.darker(135)
            hover = hover.darker(120)
            border = "rgba(255,255,255,0.14)"
            unsel_txt = "#cfd3da"
        else:
            unsel = unsel.darker(110)
            hover = hover.darker(105)
            border = "rgba(0,0,0,0.15)"
            unsel_txt = "#222222"

        unsel_bg = unsel.name()
        hover_bg = hover.name()

        self.tabs.setStyleSheet(f"""
        QTabWidget::pane {{
            border: 0px;
        }}
        QTabBar {{
            qproperty-drawBase: 0;
        }}
        QTabBar::tab {{
            background: {unsel_bg};
            color: {unsel_txt};
            border: 1px solid {border};
            border-bottom-color: transparent;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 7px 14px;
            margin-right: 6px;
            margin-top: 6px;
            min-height: 28px;
        }}
        QTabBar::tab:hover {{
            background: {hover_bg};
        }}
        QTabBar::tab:selected {{
            background: {sel_bg};
            color: {sel_txt};
            font-weight: 700;
            border-color: {border};
            margin-top: 0px;
        }}
        """)

    def changeEvent(self, e):
        if e.type() == QtCore.QEvent.PaletteChange:
            self._apply_tab_style()
        super().changeEvent(e)

    def _toggle_memory(self):
        self.memory_enabled = self.mem_cb.isChecked()
        prefs = read_prefs()
        prefs["MEMORY_ENABLED"] = "true" if self.memory_enabled else "false"
        write_prefs(prefs)
        os.environ["MEMORY_ENABLED"] = prefs["MEMORY_ENABLED"]
        if self.memory_enabled:
            self.history = mem.load_recent(self.memory_token_limit)
        else:
            self.history = []
        self.update_key_status()
        
    def _build_input_bar(self):
        wrap = QtWidgets.QFrame()
        wrap.setObjectName("InputWrap")
        wrap.setFrameShape(QtWidgets.QFrame.StyledPanel)
        wrap.setLineWidth(1)
        wrap.setStyleSheet(
            """
            #InputWrap { border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; background: palette(base); }
            QToolButton#AskInBar { border: 1px solid rgba(255,255,255,0.18); border-radius: 6px; padding: 2px 6px; min-width: 28px; min-height: 24px; }
            """
        )

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

    def _persist_quick_prefs(self):
        prefs = read_prefs()
        prefs["ASK_RERANKER"] = self.reranker_combo.currentText()
        prefs["ASK_CANDIDATES"] = int(self.pool_spin.value())
        prefs["ASK_TIME_BUDGET_SEC"] = int(self.time_spin.value())
        prefs["ASK_EXHAUSTIVE"] = "true" if self.exh_cb.isChecked() else "false"
        prefs["ASK_AGENTS"] = "true" if self.agents_cb.isChecked() else "false"
        write_prefs(prefs)
        os.environ["ASK_RERANKER"] = prefs["ASK_RERANKER"]
        os.environ["ASK_CANDIDATES"] = str(prefs["ASK_CANDIDATES"])
        os.environ["ASK_TIME_BUDGET_SEC"] = str(prefs["ASK_TIME_BUDGET_SEC"])
        os.environ["ASK_EXHAUSTIVE"] = prefs["ASK_EXHAUSTIVE"]
        os.environ["ASK_AGENTS"] = prefs["ASK_AGENTS"]
        self.update_key_status()

    def update_key_status(self):
        status = present_keys()
        prefs = read_prefs()
        rer = (prefs.get("ASK_RERANKER") or os.environ.get("ASK_RERANKER", "off")).lower()
        cand = prefs.get("ASK_CANDIDATES") or os.environ.get("ASK_CANDIDATES", "300")
        bits = []
        bits.append(f"OPENAI: {'✓' if status.get('OPENAI_API_KEY') else '×'}")
        bits.append(f"HF: {'✓' if status.get('HF_TOKEN') else '×'}")
        bits.append(f"Memory: {'on' if self.memory_enabled else 'off'}")
        bits.append(f"Reranker: {rer}")
        bits.append(f"Pool: {cand}")
        bits.append(f"Time(s): {prefs.get('ASK_TIME_BUDGET_SEC') or os.environ.get('ASK_TIME_BUDGET_SEC','120')}")
        bits.append(f"Exh: {prefs.get('ASK_EXHAUSTIVE') or os.environ.get('ASK_EXHAUSTIVE','false')}")
        agents_on = as_bool(prefs.get('ASK_AGENTS', os.environ.get('ASK_AGENTS','false')))
        bits.append(f"Agents: {'on' if agents_on else 'off'}")
        self.status.showMessage(" | ".join(bits))

    def cancel_current(self):
        if self.ingest_worker and self.ingest_worker.isRunning():
            self.status.showMessage("Cancelling ingest…")
            self.state_chip.setText("Cancelling…")
            self.ingest_worker.requestInterruption()
            return
        if self.ask_worker and self.ask_worker.isRunning():
            self.status.showMessage("Cancelling ask…")
            self.state_chip.setText("Cancelling…")
            self.ask_worker.requestInterruption()
            return
        self.status.showMessage("Nothing to cancel")
        self.state_chip.setText("Idle")

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
        self.cancel_btn.show()
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
        self.cancel_btn.hide()

    def ingest_fail(self, msg):
        self.progress.hide()
        self.ingest_btn.setEnabled(True)
        self.pick_btn.setEnabled(True)
        self.ingest_log.appendPlainText("\n[ERROR]\n" + msg)
        self.status_prog.hide()
        if str(msg).strip().startswith("Cancelled"):
            self.status.showMessage("Operation cancelled")
        else:
            QtWidgets.QMessageBox.critical(self, "Ingest failed", msg)
        self.cancel_btn.hide()

    def run_ask(self):
        q = self.q_edit.toPlainText().strip()
        if not q:
            QtWidgets.QMessageBox.information(self, "Ask", "Type a question first.")
            return
        self.q_edit.clear()
        render_markdown(self.answer, f"### Q: {q}\n\n_Working…_")
        self.state_chip.setText("Working…")
        self.ask_btn_inbar.setEnabled(False)
        self.status_prog.show()
        self.status_prog.setValue(0)
        self.cancel_btn.show()
        prefs = read_prefs()
        prefs["ASK_STRICT_DOC"] = "true" if self.strict_cb.isChecked() else "false"
        prefs["ASK_FORMULA_FORCE"] = "true" if self.formula_cb.isChecked() else "false"
        prefs["ASK_AGENTS"] = "true" if self.agents_cb.isChecked() else "false"
        write_prefs(prefs)
        os.environ["ASK_STRICT_DOC"] = "1" if self.strict_cb.isChecked() else "0"
        os.environ["ASK_FORMULA_FORCE"] = "1" if self.formula_cb.isChecked() else "0"
        os.environ["ASK_AGENTS"] = "true" if self.agents_cb.isChecked() else "false"
        self._persist_quick_prefs()
        self.ask_worker = AskWorker(
            q, self.k_spin.value(), history=self.history,
            formula_mode=self.formula_cb.isChecked(),
            agents_enabled=self.agents_cb.isChecked()
        )
        self.ask_worker.progress.connect(lambda s: self.status.showMessage(s))
        self.ask_worker.progress_pct.connect(self.status_prog.setValue)
        self.ask_worker.finished.connect(self.ask_done)
        self.ask_worker.failed.connect(self.ask_fail)
        self.ask_worker._last_q = q
        self.ask_worker.start()

    def ask_done(self, out):
        self.ask_btn_inbar.setEnabled(True)
        text = out.get("answer", "")
        cites = ", ".join(out.get("citations", []))
        quotes = out.get("quotes", [])
        qmd = ""
        if quotes:
            qmd = "\n\n### Evidence snippets\n" + "\n".join([f"> {q['quote']}\n>\n> — {q['source']}" for q in quotes])
        q = getattr(self.ask_worker, "_last_q", "") if hasattr(self, "ask_worker") else ""
        prefix = f"### Q: {q}\n\n" if q else ""
        agent_diag = ""
        if isinstance(out, dict):
            meta = out.get("agent_meta")
            rep = out.get("agent_report")
            if meta or self.agents_cb.isChecked():
                if meta and meta.get("enabled"):
                    agent_diag = "\n\n_Agents: " + ("modified._" if meta.get("changed") else "validated._")
                else:
                    agent_diag = "\n\n_Agents: ran._"
                if isinstance(rep, str) and rep.strip():
                    agent_diag += "\n\n<details><summary>Agent report</summary>\n\n" + rep + "\n\n</details>"
        render_markdown(self.answer, prefix + text + "\n\n**Citations:** " + cites + qmd + agent_diag)
        if self.memory_enabled and q and text:
            mem.append_turn(q, text)
            mem.prune_file(self.memory_file_limit_mb)
            self.history = mem.load_recent(self.memory_token_limit)
        self.status.showMessage("Answer ready")
        self.state_chip.setText("Ready")
        self.status_prog.hide()
        self.cancel_btn.hide()

    def ask_fail(self, msg):
        self.ask_btn_inbar.setEnabled(True)
        render_markdown(self.answer, "**ERROR**\n\n```\n" + msg + "\n```")
        self.state_chip.setText("Error")
        if str(msg).strip().startswith("Cancelled"):
            self.status.showMessage("Operation cancelled")
        else:
            QtWidgets.QMessageBox.critical(self, "Ask failed", msg)
        self.status_prog.hide()
        self.cancel_btn.hide()

    def show_shortcuts(self):
        qs_key = "Cmd+Shift+Q" if sys.platform == "darwin" else "Ctrl+Shift+Q"
        shortcuts = [
            ("Ask", "Enter"),
            ("Ask (from Ask box)", "Ctrl+Enter / Shift+Enter"),
            ("Focus Ask", "Cmd+L" if sys.platform == "darwin" else "Ctrl+L"),
            ("Toggle Quick Settings", qs_key),
            ("Cancel", "Esc, Cmd+. or Ctrl+."),
            ("Copy Answer", "—"),
            ("Save Answer", "—"),
            ("Open Answer", "—"),
            ("Clear Answer", "—"),
            ("Keyboard Shortcuts", "Cmd+/" if sys.platform == "darwin" else "Ctrl+/"),
        ]
        msg = "\n".join([f"{name}: {key}" for name, key in shortcuts])
        QtWidgets.QMessageBox.information(self, "Keyboard Shortcuts", msg)

    def copy_answer(self):
        text = self.answer.toPlainText()
        QtWidgets.QApplication.clipboard().setText(text)
        self.status.showMessage("Answer copied")

    def save_answer(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Answer", "", "Text Files (*.txt);;Markdown (*.md);;All Files (*)")
        if not fn:
            return
        try:
            with open(fn, "w", encoding="utf-8") as f:
                f.write(self.answer.toPlainText())
            self.status.showMessage(f"Answer saved to {fn}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Answer", str(e))

    def open_answer(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Answer", "", "Text Files (*.txt);;Markdown (*.md);;All Files (*)")
        if not fn:
            return
        try:
            with open(fn, "r", encoding="utf-8") as f:
                text = f.read()
            render_markdown(self.answer, text)
            self.status.showMessage(f"Loaded answer from {fn}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open Answer", str(e))

    def clear_answer(self):
        self.answer.clear()
        self.status.showMessage("Answer cleared")
        self.state_chip.setText("Idle")

    def _persist_tab_index(self, idx):
        prefs = read_prefs()
        prefs["UI_TAB_INDEX"] = idx
        write_prefs(prefs)

def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app_icon = load_icon('icon') or load_icon('icon.png')
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
