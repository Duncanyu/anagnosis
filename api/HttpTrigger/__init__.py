try:
    import azure.functions as func
    from azure_functions_asgi import AsgiMiddleware
    # Import the FastAPI ASGI app from the repository
    from api.main import app as asgi_app

    asgi_handler = AsgiMiddleware(asgi_app)

    def main(req: func.HttpRequest, context: func.Context):
        # Delegate request handling to the ASGI middleware
        return asgi_handler.handle(req, context)
except Exception as e:
    # If imports fail (for example, during tooling or missing packages), return a helpful error
    def main(req, context):
        return func.HttpResponse(f"Azure Functions wrapper setup error: {e}", status_code=500)
