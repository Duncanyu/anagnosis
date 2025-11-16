import os, json, logging, pathlib
import azure.functions as func

_session = None
def _maybe_load_session():
    global _session
    if _session is not None:
        return _session
    try:
        import onnxruntime as ort
        model = pathlib.Path(__file__).parent.parent / "models" / "minilm.onnx"
        if not model.exists():
            logging.warning("ONNX model not found at %s; skipping inference", model)
            return None
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
        _session = ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
        logging.info("ONNX session ready: %s", _session.get_providers())
        return _session
    except Exception as e:
        logging.warning("Failed to init ONNX (%s); continuing without inference", e)
        return None

def main(msg: func.QueueMessage) -> None:
    body = msg.get_body().decode()
    logging.info("Queue message: %s", body)
    try:
        payload = json.loads(body)
    except Exception:
        payload = {"raw": body}
    _maybe_load_session()
    logging.info("Processed keys: %s", list(payload.keys()))
