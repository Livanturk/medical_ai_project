import chromadb

# Load ChromaDB
chroma_client = chromadb.PersistentClient(path="classification/data/chroma_db")
collection = chroma_client.get_or_create_collection(name="brain_tumor_data")

# Fetch stored documents
data = collection.get()

print("Total stored documents:", len(data["documents"]))
print("Sample document:", data["documents"][:3])  # Print first 3 stored docs
