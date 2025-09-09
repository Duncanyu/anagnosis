import sys, pathlib, traceback
from typing import List, Dict
from PySide6 import QtWidgets, QtCore, QtGui
from rich import print as rprint

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services.parse import parse_pdf_bytes
from api.services.chunk import chunk_pages
from api.services.index import add_chunks, search
from api.services.summarize import summarize
from api.core.config import save_secret, present_keys, load_config

APP_TITLE = "Reading Agent"

class IngestWorker(QtCore.QThread):
    progress = QtCore.Signal(str)
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, pdf_path: pathlib.Path):
        super().__init__()
        self.pdf_path = pdf_path

    def run(self):
        try:
            self.progress.emit(f"Loading {self.pdf_path.name}…")
            pdf_bytes = self.pdf_path.read_bytes()

            self.progress.emit("Parsing PDF (OCR fallback as needed)…")
            parsed = parse_pdf_bytes(pdf_bytes)

            self.progress.emit("Chunking…")
            chunks = chunk_pages(parsed["pages"])

            self.progress.emit("Embedding & indexing…")
            ids = add_chunks(chunks)

            self.finished.emit({
                "file": str(self.pdf_path),
                "num_pages": parsed["num_pages"],
                "ocr_pages": parsed["ocr_pages"],
                "num_chunks": len(chunks),
                "ids_start": ids[0] if ids else None
            })
        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n{tb}")


class AskWorker(QtCore.QThread):
    progress = QtCore.Signal(str)
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, question: str, k: int):
        super().__init__()
        self.question = question
        self.k = k

    def run(self):
        try:
            self.progress.emit("Searching index…")
            hits = search(self.question, k=self.k)
            if not hits:
                self.finished.emit({"answer": "No results. Ingest a PDF first.", "citations": []})
                return
            top_chunks = [h[1] for h in hits]
            self.progress.emit("Summarizing with context…")
            out = summarize(self.question, top_chunks)
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

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
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
        self.accept()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(980, 680)

        tb = self.addToolBar("Main")
        act_settings = QtGui.QAction("Settings", self)
        act_settings.triggered.connect(self.open_settings)
        tb.addAction(act_settings)

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        ingest = QtWidgets.QWidget()
        v1 = QtWidgets.QVBoxLayout(ingest)

        self.pick_btn = QtWidgets.QPushButton("Choose PDF…")
        self.pick_btn.clicked.connect(self.choose_pdf)
        self.ingest_btn = QtWidgets.QPushButton("Ingest")
        self.ingest_btn.clicked.connect(self.ingest_pdf)
        self.ingest_btn.setEnabled(False)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.pick_btn)
        row.addWidget(self.ingest_btn)

        self.sel_label = QtWidgets.QLabel("No file selected.")
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

        self.q_edit = QtWidgets.QPlainTextEdit()
        self.q_edit.setPlaceholderText("Ask anything about your readings…")
        self.k_spin = QtWidgets.QSpinBox()
        self.k_spin.setRange(1, 20)
        self.k_spin.setValue(6)

        self.ask_btn = QtWidgets.QPushButton("Ask")
        self.ask_btn.clicked.connect(self.run_ask)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Top-k:"))
        controls.addWidget(self.k_spin)
        controls.addStretch(1)
        controls.addWidget(self.ask_btn)

        self.answer = QtWidgets.QPlainTextEdit()
        self.answer.setReadOnly(True)

        v2.addWidget(self.q_edit)
        v2.addLayout(controls)
        v2.addWidget(self.answer)

        tabs.addTab(ingest, "Ingest PDFs")
        tabs.addTab(ask, "Ask")

        self.selected_pdf: pathlib.Path | None = None
        self.ingest_worker: IngestWorker | None = None
        self.ask_worker: AskWorker | None = None

        self.update_key_status()

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            self.update_key_status()

    def update_key_status(self):
        status = present_keys()
        bits = []
        bits.append(f"OPENAI: {'✓' if status.get('OPENAI_API_KEY') else '×'}")
        bits.append(f"HF: {'✓' if status.get('HF_TOKEN') else '×'}")
        self.status.showMessage(" | ".join(bits))

    def choose_pdf(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose PDF", "", "PDF files (*.pdf)")
        if not fn: return
        self.selected_pdf = pathlib.Path(fn)
        self.sel_label.setText(f"Selected: {self.selected_pdf.name}")
        self.ingest_btn.setEnabled(True)

    def ingest_pdf(self):
        if not self.selected_pdf: return
        self.ingest_log.clear()
        self.progress.show()
        self.ingest_btn.setEnabled(False)
        self.pick_btn.setEnabled(False)

        self.ingest_worker = IngestWorker(self.selected_pdf)
        self.ingest_worker.progress.connect(self.ingest_log.appendPlainText)
        self.ingest_worker.finished.connect(self.ingest_done)
        self.ingest_worker.failed.connect(self.ingest_fail)
        self.ingest_worker.start()

    def ingest_done(self, info: Dict):
        self.progress.hide()
        self.ingest_btn.setEnabled(True)
        self.pick_btn.setEnabled(True)
        self.ingest_log.appendPlainText(f"\nDone.\n{info}")
        self.status.showMessage(f"Ingested: {pathlib.Path(info['file']).name} — chunks: {info['num_chunks']}")

    def ingest_fail(self, msg: str):
        self.progress.hide()
        self.ingest_btn.setEnabled(True)
        self.pick_btn.setEnabled(True)
        self.ingest_log.appendPlainText("\n[ERROR]\n" + msg)
        QtWidgets.QMessageBox.critical(self, "Ingest failed", msg)

    def run_ask(self):
        q = self.q_edit.toPlainText().strip()
        if not q:
            QtWidgets.QMessageBox.information(self, "Ask", "Type a question first.")
            return
        self.answer.setPlainText("Working…")
        self.ask_btn.setEnabled(False)

        self.ask_worker = AskWorker(q, self.k_spin.value())
        self.ask_worker.progress.connect(lambda s: self.status.showMessage(s))
        self.ask_worker.finished.connect(self.ask_done)
        self.ask_worker.failed.connect(self.ask_fail)
        self.ask_worker.start()

    def ask_done(self, out: Dict):
        self.ask_btn.setEnabled(True)
        text = out.get("answer", "")
        cites = ", ".join(out.get("citations", []))
        self.answer.setPlainText(f"{text}\n\nCitations: {cites}")
        self.status.showMessage("Answer ready")

    def ask_fail(self, msg: str):
        self.ask_btn.setEnabled(True)
        self.answer.setPlainText("[ERROR]\n" + msg)
        QtWidgets.QMessageBox.critical(self, "Ask failed", msg)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()