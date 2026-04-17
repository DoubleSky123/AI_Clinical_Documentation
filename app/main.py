"""
FastAPI Application Entry Point
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from langchain_community.retrievers import BM25Retriever

from app.routers import notes, evaluate, voice
from app.vectorstore.build_index import load_chunks, load_vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load FAISS index + build BM25 retriever, share across all requests
    app.state.vectorstore = load_vectorstore()
    chunks = load_chunks()
    app.state.bm25_retriever = BM25Retriever.from_documents(chunks)
    yield
    # Shutdown: close Redis connection
    from app.cache import close_client
    await close_client()


app = FastAPI(
    title="Clinical Documentation AI",
    description="RAG-powered SOAP note generation for respiratory infection encounters",
    version="1.0.0",
    lifespan=lifespan,
)

# Expose /metrics endpoint for Prometheus scraping
# Automatically tracks: request count, latency histogram, error rate per route
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(notes.router)
app.include_router(evaluate.router)
app.include_router(voice.router)
