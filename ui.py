import streamlit as st
import requests
import uuid

st.set_page_config(page_title="RAG Agent", page_icon="📄", layout="wide")

st.title("📄 RAG Agent — Multi-Document Knowledge Base")
st.markdown("Upload multiple PDFs and have a conversation across all of them.")
st.divider()

# Session setup
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Knowledge Base")

    # Show existing documents
    try:
        docs_res = requests.get("http://127.0.0.1:8000/documents")
        if docs_res.status_code == 200:
            data = docs_res.json()
            st.markdown(f"**Total chunks:** {data['total_chunks']}")

            if data["documents"]:
                st.markdown("**Uploaded Documents:**")
                for doc in data["documents"]:
                    col1, col2 = st.columns([3, 1])
                    col1.markdown(f"📄 {doc}")
                    if col2.button("🗑️", key=f"del_{doc}"):
                        del_res = requests.delete(f"http://127.0.0.1:8000/documents/{doc}")
                        st.success(f"Deleted {doc}")
                        st.rerun()
            else:
                st.info("No documents uploaded yet.")
    except:
        st.warning("API not running. Start uvicorn first.")

    st.divider()

    # Upload new document
    st.subheader("➕ Add Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file is not None:
        if st.button("Upload & Process"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                res = requests.post(
                    "http://127.0.0.1:8000/upload",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                )
                if res.status_code == 200:
                    data = res.json()
                    st.success(data["message"])
                    st.info(f"📄 Pages: {data['pages']} | 🧩 Chunks: {data['chunks']}")
                    st.info(f"📚 Total chunks in KB: {data['total_chunks']}")
                    st.rerun()
                else:
                    st.error("Upload failed.")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        requests.post(f"http://127.0.0.1:8000/clear?session_id={st.session_state.session_id}")
        st.session_state.messages = []
        st.rerun()

    st.markdown(f"**Session:** `{st.session_state.session_id[:8]}...`")
    st.markdown("**Model:** llama-3.3-70b")
    st.markdown("**Vector DB:** ChromaDB ✨")
    st.markdown("**Memory:** Last 3 exchanges")

# --- MAIN CHAT ---
# Check if any documents exist
try:
    docs_res = requests.get("http://127.0.0.1:8000/documents")
    has_docs = docs_res.json()["total_chunks"] > 0
except:
    has_docs = False

if not has_docs:
    st.info("👈 Upload at least one PDF from the sidebar to get started.")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything across all your documents...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching across all documents..."):
                res = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"question": question, "session_id": st.session_state.session_id}
                )
                answer = res.text
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})