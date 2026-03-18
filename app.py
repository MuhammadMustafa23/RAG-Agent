import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from sentence_transformers import CrossEncoder

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
llm = ChatGroq(model="llama-3.3-70b-versatile")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chat_histories = {}

# CrossEncoder reranker — scores (question, chunk) pairs for relevance
print("Loading reranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Reranker ready!")

CHROMA_PATH = "chroma_db"
vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

class Question(BaseModel):
    question: str
    session_id: str = "default"

# --- Reranker function ---
def rerank(question, chunks, top_n=3):
    # Build pairs of (question, chunk_text) for scoring
    pairs = [(question, doc.page_content) for doc in chunks]
    
    # Score each pair — higher = more relevant
    scores = reranker.predict(pairs)
    
    # Sort chunks by score descending
    scored_chunks = sorted(zip(scores, chunks), reverse=True)
    
    # Return only top_n most relevant
    print(f"\nReranking: {len(chunks)} chunks → top {top_n}")
    for i, (score, doc) in enumerate(scored_chunks[:top_n]):
        print(f"  Rank {i+1} | Score: {score:.3f} | {doc.page_content[:60]}...")
    
    return [doc for _, doc in scored_chunks[:top_n]]

# --- ENDPOINT 1: Upload PDF ---
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    file_path = f"docs/{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["filename"] = file.filename

    vector_store.add_documents(chunks)

    return {
        "message": f"'{file.filename}' uploaded successfully!",
        "pages": len(pages),
        "chunks": len(chunks),
        "total_docs": len(vector_store.get()["ids"])
    }

# --- ENDPOINT 2: List documents ---
@app.get("/documents")
def list_documents():
    data = vector_store.get()

    filenames = set()
    for metadata in data["metadatas"]:
        if metadata and "filename" in metadata:
            filenames.add(metadata["filename"])

    return {
        "total_chunks": len(data["ids"]),
        "documents": list(filenames)
    }

# --- ENDPOINT 3: Delete document ---
@app.delete("/documents/{filename}")
def delete_document(filename: str):
    data = vector_store.get()

    ids_to_delete = []
    for i, metadata in enumerate(data["metadatas"]):
        if metadata and metadata.get("filename") == filename:
            ids_to_delete.append(data["ids"][i])

    if not ids_to_delete:
        return {"message": f"'{filename}' not found."}

    vector_store.delete(ids=ids_to_delete)
    return {
        "message": f"'{filename}' deleted!",
        "chunks_removed": len(ids_to_delete)
    }

# --- ENDPOINT 4: Ask with Memory + Reranking ---
@app.post("/ask", response_class=PlainTextResponse)
def ask(body: Question):
    total = len(vector_store.get()["ids"])
    if total == 0:
        return "No documents uploaded yet. Please upload a PDF first."

    if body.session_id not in chat_histories:
        chat_histories[body.session_id] = []
    history = chat_histories[body.session_id]

    # Step 1: Retrieve broadly — get 10 candidates
    candidate_chunks = vector_store.similarity_search(body.question, k=10)

    # Step 2: Rerank and keep only top 3
    best_chunks = rerank(body.question, candidate_chunks, top_n=3)

    # Step 3: Build context from best chunks only
    context = ""
    sources = []
    for doc in best_chunks:
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page", 0) + 1
        context += f"[From: {filename}, Page {page}]\n{doc.page_content}\n\n"
        sources.append(f"{filename} p.{page}")

    # Step 4: Build messages
    messages = [
        SystemMessage(content="""You are a helpful assistant that answers questions
based strictly on the provided document context.
Multiple documents may be provided — always mention which document your answer comes from.
You remember the conversation history for follow-up questions.
Format your response with:
- Clear headings
- Bullet points where needed
- Mention source document for each point
- A short summary at the end
If the answer is not found, say 'This information is not in the uploaded documents'.""")
    ]

    messages += history[-6:]

    messages.append(HumanMessage(content=f"""
Document Context:
{context}

Current Question: {body.question}
"""))

    response = llm.invoke(messages)
    answer = response.content

    history.append(HumanMessage(content=body.question))
    history.append(AIMessage(content=answer))

    sources_text = f"\n\n📌 Sources: {', '.join(set(sources))}"
    return answer + sources_text

# --- ENDPOINT 5: Clear history ---
@app.post("/clear")
def clear_history(session_id: str = "default"):
    if session_id in chat_histories:
        chat_histories[session_id] = []
    return {"message": "Chat history cleared!"}