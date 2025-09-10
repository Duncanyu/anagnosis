# Reading Agent

A document processing and question-answering system that combines a FastAPI backend with a Qt desktop application. The system can ingest PDF documents and images, process them using OCR when needed, create embeddings, and answer questions about the content using RAG (Retrieval-Augmented Generation).

## Features

- **Document Processing**: Support for PDF files and images (PNG, JPG, JPEG, TIF, TIFF, BMP, WEBP)
- **OCR Integration**: Automatic OCR for scanned documents and images using Tesseract
- **Vector Search**: FAISS-based similarity search with configurable embedding backends
- **Question Answering**: RAG-based Q&A using OpenAI or local LLM backends
- **Conversation Memory**: Optional conversation history tracking with configurable limits
- **Smart Aliases**: Automatic learning and application of user-preferred terminology
- **Document Summarization**: Auto-generated summaries for ingested documents
- **Evidence Extraction**: Quote extraction with source citations for answers
- **Dual Interface**: Both REST API and Qt desktop application
- **Flexible Backends**: Support for multiple embedding and LLM providers

## Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        PDF[PDF Files]
        IMG[Images]
        API_REQ[API Requests]
        QT_APP[Qt Desktop App]
    end
    
    subgraph "Processing Layer"
        PARSE[Document Parser]
        OCR[OCR Engine<br/>Tesseract]
        CHUNK[Text Chunker]
        EMBED[Embedding Service<br/>HF/OpenAI]
        ALIAS[Alias Learning<br/>& Application]
    end
    
    subgraph "Storage Layer"
        FAISS[FAISS Index]
        META[Metadata Store]
        CHUNKS[Chunk Storage]
        MEMORY[Conversation<br/>Memory]
        ALIASES[User Aliases]
        DOC_SUM[Document<br/>Summaries]
    end
    
    subgraph "Query Layer"
        SEARCH[Vector Search]
        LLM[LLM Service<br/>OpenAI/vLLM/HF]
        SUMM[Summarization]
        QUOTES[Quote<br/>Extraction]
    end
    
    subgraph "Output Layer"
        API_RESP[API Response]
        QT_UI[Qt Interface]
    end
    
    PDF --> PARSE
    IMG --> PARSE
    API_REQ --> PARSE
    QT_APP --> PARSE
    
    PARSE --> OCR
    PARSE --> CHUNK
    OCR --> CHUNK
    
    CHUNK --> EMBED
    CHUNK --> SUMM
    EMBED --> FAISS
    EMBED --> META
    CHUNK --> CHUNKS
    SUMM --> DOC_SUM
    
    API_REQ --> SEARCH
    QT_APP --> SEARCH
    API_REQ --> ALIAS
    QT_APP --> ALIAS
    
    SEARCH --> FAISS
    SEARCH --> CHUNKS
    SEARCH --> MEMORY
    SEARCH --> LLM
    
    LLM --> SUMM
    LLM --> QUOTES
    ALIAS --> ALIASES
    ALIASES --> LLM
    
    SUMM --> API_RESP
    SUMM --> QT_UI
    SUMM --> MEMORY
    QUOTES --> API_RESP
    QUOTES --> QT_UI
```

## Project Structure

```
reading-agent/
├── api/                    # FastAPI backend
│   ├── main.py            # FastAPI application entry point
│   ├── core/
│   │   └── config.py      # Configuration management
│   ├── routes/
│   │   ├── upload.py      # Document upload endpoints
│   │   ├── query.py       # Query endpoints
│   │   └── secrets.py     # API key management
│   └── services/
│       ├── parse.py       # Document parsing (PDF/images)
│       ├── chunk.py       # Text chunking
│       ├── embed.py       # Text embedding
│       ├── index.py       # Vector indexing with FAISS
│       ├── summarize.py   # LLM-based summarization
│       ├── memory.py      # Conversation memory management
│       └── aliases.py     # User terminology aliases
├── app/
│   └── app_qt.py         # Qt desktop application
├── artifacts/            # Generated files and storage
│   ├── index.faiss       # FAISS vector index
│   ├── meta.json         # Index metadata
│   ├── chunks.jsonl      # Document chunks
│   ├── memory.jsonl      # Conversation history
│   ├── aliases.json      # User-defined aliases
│   ├── doc_summaries.jsonl # Document summaries
│   └── ui_prefs.json     # UI preferences
└── .env.example         # Environment configuration template
```

## Prerequisites

### System Dependencies

- **Python 3.8+**
- **Tesseract OCR**: Required for OCR functionality
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

### Optional Dependencies

- **Qdrant**: Vector database (if using instead of FAISS)
- **Local LLM Server**: For vLLM backend (e.g., running on `localhost:8001`)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Duncanyu/reading-agent.git
   cd reading-agent
   ```

2. **Install Python dependencies**:
   ```bash
   pip install fastapi uvicorn pyside6 pymupdf pillow pytesseract faiss-cpu numpy openai transformers torch sentence-transformers markdown
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your configuration:
   ```env
   # Backend selection
   EMBED_BACKEND=hf           # hf|openai
   LLM_BACKEND=openai         # vllm|openai
   RERANK_BACKEND=hf          # hf|cohere|none
   
   # Memory settings
   MEMORY_ENABLED=false       # Enable conversation memory
   MEMORY_TOKEN_LIMIT=1200    # Token limit for memory context
   MEMORY_FILE_LIMIT_MB=50    # Max memory file size in MB
   
   # API endpoints
   QDRANT_URL=http://localhost:6333
   LLM_OPENAI_BASE=http://localhost:8001/v1
   
   # API Keys
   OPENAI_API_KEY=your_openai_key_here
   HF_TOKEN=your_huggingface_token_here
   COHERE_API_KEY=your_cohere_key_here
   ```

## Usage

### Option 1: Qt Desktop Application

Run the desktop application:
```bash
python app/app_qt.py
```

**Features:**
- **Settings**: Configure API keys, backend preferences, and memory settings
- **Ingest Tab**: Upload and process documents (PDF/images) with auto-summarization
- **Ask Tab**: Query your documents with natural language and conversation memory

**Workflow:**
1. Open Settings and configure your API keys and enable memory if desired
2. Go to Ingest tab, choose files, and click "Ingest"
3. Review auto-generated document summaries
4. Switch to Ask tab and ask questions about your documents

**Advanced Features:**
- **Memory**: Enable in Settings to maintain conversation context across questions
- **Aliases**: Use natural language like "refer to ML as machine learning" to set terminology preferences
- **Evidence Snippets**: Answers include relevant quotes with page citations
- **Clear Index**: Use toolbar button to reset and start fresh

### Option 2: REST API

Start the FastAPI server:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**API Endpoints:**

- **Upload Document**: `POST /upload/`
  ```bash
  curl -X POST "http://localhost:8000/upload/" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@document.pdf"
  ```

- **Query Documents**: `POST /query/`
  ```bash
  curl -X POST "http://localhost:8000/query/" \
       -H "Content-Type: application/json" \
       -d '{"question": "What is the main topic?", "k": 5}'
  ```

- **API Documentation**: Visit `http://localhost:8000/docs`

## Configuration Options

### Embedding Backends

- **HuggingFace (`hf`)**: Uses `intfloat/e5-small-v2` model (default)
- **OpenAI (`openai`)**: Uses `text-embedding-3-small` model

### LLM Backends

- **OpenAI (`openai`)**: Uses OpenAI's GPT models (default)
- **vLLM (`vllm`)**: Uses local vLLM server for inference

### Reranking Backends

- **HuggingFace (`hf`)**: Local reranking model
- **Cohere (`cohere`)**: Cohere's reranking API
- **None (`none`)**: No reranking

## Data Flow

1. **Document Ingestion**:
   - Parse PDF/images using PyMuPDF and PIL
   - Apply OCR with Tesseract for scanned content
   - Split text into semantic chunks
   - Generate embeddings using selected backend
   - Store in FAISS index with metadata
   - Generate document summaries automatically

2. **Query Processing**:
   - Learn and apply user aliases from question
   - Embed user question with alias substitutions
   - Search FAISS index for similar chunks
   - Load conversation history (if memory enabled)
   - Retrieve top-k relevant passages
   - Extract evidence quotes with citations
   - Generate answer using LLM with context and history
   - Store conversation turn in memory (if enabled)
   - Return answer with citations and evidence snippets

3. **Memory Management**:
   - Track conversation history with configurable token limits
   - Automatic file size management and pruning
   - Context-aware responses using recent interactions

4. **Alias Learning**:
   - Detect terminology preferences from user input
   - Store and apply aliases consistently across sessions
   - Support natural language alias definitions

## Storage

- **FAISS Index**: `artifacts/index.faiss`
- **Metadata**: `artifacts/meta.json`
- **Chunks**: `artifacts/chunks.jsonl`
- **Conversation Memory**: `artifacts/memory.jsonl`
- **User Aliases**: `artifacts/aliases.json`
- **Document Summaries**: `artifacts/doc_summaries.jsonl`
- **UI Preferences**: `artifacts/ui_prefs.json`

## Troubleshooting

### Common Issues

1. **Tesseract not found**:
   - Ensure Tesseract is installed and in PATH
   - On Windows, you may need to set `TESSDATA_PREFIX`

2. **FAISS installation issues**:
   - Use `faiss-cpu` for CPU-only systems
   - Use `faiss-gpu` if you have CUDA support

3. **Memory issues with large documents**:
   - Adjust chunk size in `api/services/chunk.py`
   - Consider using Qdrant instead of FAISS for large datasets

4. **API key errors**:
   - Verify API keys in `.env` file
   - Check that the correct backend is selected

### Performance Tips

- Use GPU-accelerated embeddings when available
- Adjust `k` parameter in queries for speed vs. accuracy trade-off
- Consider using reranking for better result quality
- Use local LLM backends to reduce API costs

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both Qt app and API
5. Submit a pull request

## License

This project is open source. Please check the repository for license details.
