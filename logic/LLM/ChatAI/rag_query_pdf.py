import pdfplumber
import os
from openai import OpenAI
from config import API_KEY, BASE_URL, RAG_MODEL
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ---------- 1️⃣ Extract text from PDF ----------
def pdf_to_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ---------- 2️⃣ Split text into chunks ----------
def chunk_text(text, max_chars=1500):
    """Split text into chunks of roughly max_chars."""
    chunks = []
    while len(text) > 0:
        chunk = text[:max_chars]
        end = chunk.rfind(".")  # end at last full sentence if possible
        if end != -1:
            chunk = chunk[:end+1]
        chunks.append(chunk.strip())
        text = text[len(chunk):].strip()
    return chunks

# ---------- 3️⃣ Simple vector store using TF-IDF ----------
class SimpleRetriever:
    def __init__(self, chunks):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.embeddings = self.vectorizer.fit_transform(chunks)
        self.chunks = chunks

    def retrieve(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        scores = np.dot(self.embeddings, query_vec.T).toarray().ravel()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]

# ---------- 4️⃣ Query the RAG model ----------
def query_rag_model(question, retrieved_contexts):
    """Send question + context to the model."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    context_text = "\n\n".join(retrieved_contexts)
    prompt = f"Use the following document excerpts to answer:\n{context_text}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=RAG_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering based on retrieved document context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content

# ---------- 5️⃣ Main ----------
if __name__ == "__main__":
    pdf_path = "C:/Master thesis/files/pdf/rag.pdf"
    question = "What is the main conclusion of this document?"

    print("🔍 Extracting PDF text...")
    pdf_text = pdf_to_text(pdf_path)

    print("✂️ Splitting text into chunks...")
    chunks = chunk_text(pdf_text)

    print(f"📚 Created {len(chunks)} chunks for retrieval.")

    retriever = SimpleRetriever(chunks)
    retrieved = retriever.retrieve(question)

    print("\n📖 Retrieved context:")
    for c in retrieved:
        print("-", c[:200], "...\n")

    print("🤖 Querying RAG model...")
    answer = query_rag_model(question, retrieved)

    print("\n=== Model Answer ===")
    print(answer)
