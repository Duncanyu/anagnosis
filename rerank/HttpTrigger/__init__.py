import azure.functions as func

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Simple placeholder function for the rerank app.

    Replace this with actual Azure Function logic that triggers your reranker worker/process.
    """
    return func.HttpResponse("rerank function placeholder", status_code=200)
