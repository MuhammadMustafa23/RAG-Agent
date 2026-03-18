FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create docs folder
RUN mkdir -p docs

# Expose port
EXPOSE 7860

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

### Step 4 — Update `requirements.txt`
```
fastapi
uvicorn
langchain
langchain-groq
langchain-community
langchain-huggingface
langchain-chroma
langchain-text-splitters
chromadb
pypdf
streamlit
python-multipart
requests
python-dotenv
sentence-transformers
torch==2.1.0+cpu
--extra-index-url https://download.pytorch.org/whl/cpu