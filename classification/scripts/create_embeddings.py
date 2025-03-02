import chromadb
from sentence_transformers import SentenceTransformer

# Load extracted text
with open("classification/data/brain_tumor_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Chunk the text
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size].strip()  # Remove unnecessary spaces
        if chunk:  # ✅ Store only non-empty chunks
            chunks.append(chunk)
    return chunks

chunks = chunk_text(text)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="classification/data/chroma_db")
collection = chroma_client.get_or_create_collection(name="brain_tumor_data")

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Store embeddings in ChromaDB (FIXED)
for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk]  # ✅ Store text in "documents" instead of "metadatas"
    )

print("✅ Fixed: Embeddings and text stored correctly in documents!")
