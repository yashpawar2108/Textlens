# main.py
import os
import asyncio
import aiohttp
import aiofiles
import pdfplumber
import time
import hashlib
import pickle
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, APIRouter, Depends, Header, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# Langchain components
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  # make sure you have this lib installed and GROQ_API_KEY set
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings as LangchainEmbeddings

# Use direct imports for retrievers
from langchain.retrievers.bm25 import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

# Required imports for sentence transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from dotenv import load_dotenv
load_dotenv()


# Optional imports for optimizations
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from functools import lru_cache

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hackrx")

# --- FastAPI App Setup ---
app = FastAPI(title="HackRx RAG API")
router = APIRouter()

# Serve frontend static directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Allow frontend to call backend from different domain (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=6)

# Groq concurrency control to avoid rate limiting
GROQ_CONCURRENCY_LIMIT = int(os.environ.get("GROQ_CONCURRENCY_LIMIT", 8))
GROQ_API_SEMAPHORE = asyncio.Semaphore(GROQ_CONCURRENCY_LIMIT)

# --- Cache Setup ---
CACHE_ENABLED = False
redis_client = None

if REDIS_AVAILABLE:
    try:
        redis_client = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"),
                                   port=int(os.environ.get("REDIS_PORT", 6379)),
                                   db=0, decode_responses=False)
        redis_client.ping()
        CACHE_ENABLED = True
        logger.info("Redis cache connected successfully")
    except Exception:
        CACHE_ENABLED = False
        logger.warning("Redis not available, caching disabled")

# --- File-based Cache Directory ---
CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp/document_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Bearer Token Authentication ---
EXPECTED_TOKEN = os.environ.get(
    "HACKRX_BEARER_TOKEN",
    "23rjbh3h9ucn8nwuch8ef8soci9eyh3enfksbff"
)

def verify_bearer_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    token = authorization.split(" ")[1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")

# --- Request Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class ChatRequest(BaseModel):
    message: str

# --- Sentence Transformer Embeddings Class ---
class SentenceTransformerEmbeddings(LangchainEmbeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.dimension = None

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded with {self.dimension} dimensions")
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings_result = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            return embeddings_result.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode([text], convert_to_tensor=False, show_progress_bar=False)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise

# --- Utility Functions ---
def generate_cache_key(*args) -> str:
    key_string = "|".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()

def generate_document_cache_name(url: str) -> str:
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    try:
        filename = url.split('/')[-1].split('?')[0]
        clean_name = filename.replace('.pdf', '').replace('%20', '_').replace(' ', '_')
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
        if clean_name and len(clean_name) > 3:
            return f"{clean_name}_{url_hash}"
    except:
        pass
    return f"document_{url_hash}"

# --- Document Cache Management ---
def get_document_cache_paths(url: str) -> tuple:
    cache_name = generate_document_cache_name(url)
    vectorstore_path = os.path.join(CACHE_DIR, f"vectorstore_{cache_name}")
    documents_path = os.path.join(CACHE_DIR, f"documents_{cache_name}.pkl")
    metadata_path = os.path.join(CACHE_DIR, f"metadata_{cache_name}.pkl")
    return vectorstore_path, documents_path, metadata_path

def is_document_cached(url: str) -> bool:
    vectorstore_path, _, _ = get_document_cache_paths(url)
    return os.path.exists(vectorstore_path)

def save_document_cache(url: str, vectorstore, documents: List[Document], extracted_text: str):
    try:
        vectorstore_path, documents_path, metadata_path = get_document_cache_paths(url)
        cache_name = generate_document_cache_name(url)
        vectorstore.save_local(vectorstore_path)
        with open(documents_path, 'wb') as f:
            pickle.dump(documents, f)
        metadata = {
            'url': url, 'cache_name': cache_name, 'timestamp': time.time(),
            'num_documents': len(documents), 'text_length': len(extracted_text),
            'text_preview': extracted_text[:500] + "...",
            'embedding_model': embeddings.model_name,
            'embedding_dimension': embeddings.dimension
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"📦 Document cached as: {cache_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to save document cache: {str(e)}")
        return False

def load_document_cache(url: str) -> tuple:
    try:
        vectorstore_path, documents_path, metadata_path = get_document_cache_paths(url)
        vectorstore = FAISS.load_local(
            vectorstore_path, embeddings, allow_dangerous_deserialization=True
        )
        with open(documents_path, 'rb') as f:
            documents = pickle.load(f)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        logger.info(f"✅ Loaded cached document: {metadata['cache_name']}")
        return vectorstore, documents, metadata
    except Exception as e:
        logger.error(f"Failed to load document cache: {str(e)}")
        return None, None, None

# --- Sentence Transformer Utilities ---
@lru_cache(maxsize=1)
def get_sentence_transformer():
    return embeddings.model

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    try:
        model = get_sentence_transformer()
        if model is None: return 0.5
        embeddings_result = model.encode([text1, text2])
        similarity = cosine_similarity([embeddings_result[0]], [embeddings_result[1]])[0][0]
        return float(similarity)
    except:
        return 0.5

# --- Model Router ---
class ModelRouter:
    def __init__(self):
        self.fast_model = ChatGroq(
            model=os.environ.get("GROQ_FAST_MODEL", "llama3-8b-8192"),
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.1, max_tokens=256
        )
        self.powerful_model = ChatGroq(
            model=os.environ.get("GROQ_POWER_MODEL", "llama3-70b-8192"),
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.1, max_tokens=512
        )

    def route_model(self, question: str, context_length: int = 0):
        question_words = len(question.split())
        is_complex = (
            question_words > 15 or
            any(w in question.lower() for w in ['analyze', 'compare', 'explain', 'summarize']) or
            context_length > 2000
        )
        return self.powerful_model if is_complex else self.fast_model

model_router = ModelRouter()

# --- Enhanced PDF Processing ---
async def extract_text_from_pdf_async(pdf_url: str) -> str:
    """
    Supports:
     - http(s) URL: downloads with aiohttp
     - file://localpath or local absolute path: reads locally
    """
    if CACHE_ENABLED and redis_client:
        cache_key = f"pdf_text:{generate_cache_key(pdf_url)}"
        try:
            if cached_text := redis_client.get(cache_key):
                logger.info("PDF text retrieved from Redis cache")
                return pickle.loads(cached_text)
        except Exception as e:
            logger.warning(f"Redis cache retrieval failed: {str(e)}")

    # Local file support
    if pdf_url.startswith("file://") or os.path.exists(pdf_url):
        if pdf_url.startswith("file://"):
            local_path = pdf_url.replace("file://", "")
        else:
            local_path = pdf_url
        try:
            text = _extract_text_sync(local_path)
            if CACHE_ENABLED and redis_client and text:
                try:
                    redis_client.setex(f"pdf_text:{generate_cache_key(pdf_url)}", 3600, pickle.dumps(text))
                except Exception as e:
                    logger.warning(f"Redis cache storage failed: {str(e)}")
            return text
        except Exception as e:
            logger.error(f"Local PDF read failed: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to read local PDF: {e}")

    # Remote download
    temp_filename = f"temp_{hashlib.md5(pdf_url.encode()).hexdigest()}.pdf"
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(pdf_url) as response:
                response.raise_for_status()
                async with aiofiles.open(temp_filename, "wb") as f:
                    await f.write(await response.read())

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(executor, _extract_text_sync, temp_filename)

        if CACHE_ENABLED and redis_client and text:
            try:
                redis_client.setex(f"pdf_text:{generate_cache_key(pdf_url)}", 3600, pickle.dumps(text))
            except Exception as e:
                logger.warning(f"Redis cache storage failed: {str(e)}")
        return text
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def _extract_text_sync(filename: str) -> str:
    text = ""
    try:
        with pdfplumber.open(filename) as pdf:
            for page in pdf.pages:
                if page_text := page.extract_text():
                    text += " ".join(page_text.strip().split()) + "\n\n"
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise
    return text

# --- Query Enhancement ---
async def enhance_query(question: str) -> str:
    try:
        enhancement_prompt = f"Rewrite this question to include synonyms and related terms to improve document search. Keep it concise. Original: {question}\nEnhanced:"
        llm = model_router.fast_model
        loop = asyncio.get_event_loop()
        enhanced = await loop.run_in_executor(
            executor, lambda: llm.invoke(enhancement_prompt).content
        )
        return enhanced.strip()
    except Exception:
        return question

# --- Semantic Chunking ---
def create_semantic_chunks(text: str) -> List[Document]:
    return create_advanced_chunks(text)

def create_advanced_chunks(text: str) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"chunk_id": i}) for i, chunk in enumerate(chunks)]

# --- Initialize Embeddings ---
embeddings = SentenceTransformerEmbeddings(model_name=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

# --- Document Processing Pipeline ---
async def get_or_create_document_pipeline(document_url: str) -> tuple:
    if is_document_cached(document_url):
        logger.info("🔍 Document found in cache, loading...")
        vectorstore, documents, metadata = load_document_cache(document_url)
        if vectorstore and documents:
            return vectorstore, documents, metadata
        else:
            logger.warning("⚠️ Cache corrupted, rebuilding...")

    logger.info("🚀 Processing document from scratch...")
    start_time = time.time()
    text = await extract_text_from_pdf_async(document_url)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
    documents = create_semantic_chunks(text)
    loop = asyncio.get_event_loop()
    vectorstore = await loop.run_in_executor(
        executor, lambda: FAISS.from_documents(documents, embedding=embeddings)
    )
    logger.info(f"✨ Document processed in {time.time() - start_time:.2f}s")
    save_document_cache(document_url, vectorstore, documents, text)
    metadata = {'url': document_url, 'cache_name': generate_document_cache_name(document_url)}
    return vectorstore, documents, metadata

# --- Hybrid Retrieval ---
def create_hybrid_retriever(documents: List[Document], vectorstore, k: int = 6):
    try:
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = max(2, k // 3)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k - bm25_retriever.k})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7]
        )
        return ensemble_retriever
    except Exception as e:
        logger.warning(f"Hybrid retriever failed, using semantic only: {str(e)}")
        return vectorstore.as_retriever(search_kwargs={"k": k})

# --- Answer Quality Scoring ---
def score_answer_quality(question: str, answer: str, context: str) -> float:
    return calculate_semantic_similarity(question, answer)

# Use a global uploaded doc path (single process). You can extend to per-session later.
uploaded_doc_url: Optional[str] = None

# --- Enhanced answer generation with rate limiting, retries, and quality control.
async def get_answer_async(vectorstore, question: str, documents: List[Document]) -> str:
    async with GROQ_API_SEMAPHORE:
        retries = 3
        backoff_factor = 2  # 1s, 2s, 4s

        for attempt in range(retries):
            try:
                if CACHE_ENABLED and redis_client:
                    cache_key = f"answer:{generate_cache_key(question, len(documents))}"
                    if cached_answer := redis_client.get(cache_key):
                        logger.info("💨 Answer retrieved from Redis cache")
                        return pickle.loads(cached_answer)

                enhanced_question = await enhance_query(question)
                context_length = sum(len(doc.page_content) for doc in documents)
                llm = model_router.route_model(question, context_length)

                template = """You are an expert document analyst. Use the provided context to answer questions accurately.
Instructions:
1. Synthesize information from the context - don't copy-paste directly
2. Use clear, accessible language that non-experts can understand
3. Provide complete, relevant information in a well-formed paragraph
4. Start directly with your answer - no introductory phrases
5. If answering yes/no questions, start with the answer then explain
6. Base your response only on the provided context
Context: {context}
Question: {question}
Answer:"""

                prompt = PromptTemplate(template=template, input_variables=["context", "question"])
                retriever = create_hybrid_retriever(documents, vectorstore, k=6)

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm, chain_type="stuff", retriever=retriever,
                    chain_type_kwargs={"prompt": prompt}, return_source_documents=True
                )

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    executor, lambda: qa_chain.invoke({"query": enhanced_question})
                )
                answer = result["result"]

                if CACHE_ENABLED and redis_client and len(answer.strip()) > 10:
                    redis_client.setex(cache_key, 900, pickle.dumps(answer))

                return answer

            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    if attempt < retries - 1:
                        delay = backoff_factor ** attempt
                        logger.warning(
                            f"Rate limit hit for question '{question[:30]}...'. Retrying in {delay}s... (Attempt {attempt + 1}/{retries})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Final attempt failed for question '{question[:30]}...' due to rate limiting.")
                        raise HTTPException(status_code=429, detail="API rate limit exceeded after multiple retries.")
                else:
                    logger.error(f"An unexpected error occurred in get_answer_async: {e}", exc_info=True)
                    raise

# --- Batch Processing ---
async def batch_process_questions(vectorstore, questions: List[str], documents: List[Document]) -> List[str]:
    tasks = [get_answer_async(vectorstore, q, documents) for q in questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    final_answers = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"Failed to get answer for question '{questions[i]}': {res}")
            final_answers.append(f"Error: Could not process question due to an internal error or API limit.")
        else:
            final_answers.append(res)
    return final_answers

# --- Endpoints ---
@router.post("/run", dependencies=[Depends(verify_bearer_token)])
async def process_query(request: QueryRequest):
    start_time = time.time()
    try:
        vectorstore, documents, metadata = await get_or_create_document_pipeline(request.documents)
        answers = await batch_process_questions(vectorstore, request.questions, documents)
        total_time = time.time() - start_time
        logger.info(f"🎯 Query completed in {total_time:.2f}s")
        return {"answers": answers}
    except aiohttp.ClientError as e:
        logger.error(f"PDF download failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during processing")

# Upload document endpoint
@router.post("/upload", dependencies=[Depends(verify_bearer_token)])
async def upload_document(file: UploadFile = File(...)):
    global uploaded_doc_url
    filename = file.filename
    sanitized = "".join(c for c in filename if c.isalnum() or c in (" ", ".", "_", "-")).strip()
    path = os.path.join(CACHE_DIR, f"upload_{int(time.time())}_{sanitized}")
    # ensure .pdf
    if not sanitized.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    async with aiofiles.open(path, "wb") as f:
        content = await file.read()
        await f.write(content)
    uploaded_doc_url = f"file://{path}"
    logger.info(f"Uploaded document saved to: {path}")
    return {"message": "Document uploaded successfully", "filename": sanitized}

# Chat endpoint (document-aware)
@router.post("/chat", dependencies=[Depends(verify_bearer_token)])
async def chat_endpoint(req: ChatRequest):
    global uploaded_doc_url
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")
    # If a doc is uploaded, answer from doc
    if uploaded_doc_url:
        vectorstore, documents, _ = await get_or_create_document_pipeline(uploaded_doc_url)
        answers = await batch_process_questions(vectorstore, [message], documents)
        return {"response": answers[0], "mode": "document"}
    # Else, direct LLM call
    llm = model_router.route_model(message, context_length=0)
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(executor, lambda: llm.invoke(message))
    return {"response": res.content, "mode": "llm"}

# Cache status & clear endpoints
@router.get("/cache/status", dependencies=[Depends(verify_bearer_token)])
async def get_cache_status():
    return {"status": "ok", "cache_dir": CACHE_DIR}

@router.delete("/cache/clear", dependencies=[Depends(verify_bearer_token)])
async def clear_cache():
    # simple clear: remove files in cache dir (careful in prod)
    for f in os.listdir(CACHE_DIR):
        try:
            os.remove(os.path.join(CACHE_DIR, f))
        except Exception:
            pass
    return {"message": "Cache cleared"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Serve frontend index
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>Place frontend in static/index.html</h1>")

app.include_router(router, prefix="/api/v1/hackrx")

# --- Startup/Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("RAG API starting up...")
    logger.info(f"Groq API concurrency limit set to: {GROQ_CONCURRENCY_LIMIT}")
    try:
        logger.info("Loading embedding model (background)...")
        # pre-load the embedding model in background to reduce first-request latency
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, lambda: setattr(embeddings, "_model", embeddings.model))
        logger.info(f"✅ Embedding model '{embeddings.model_name}' loaded.")
    except Exception as e:
        logger.error(f"⚠️ Error loading embedding model: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("RAG API shutting down...")
    executor.shutdown(wait=True)
    if CACHE_ENABLED and redis_client:
        try:
            redis_client.close()
        except Exception:
            pass
