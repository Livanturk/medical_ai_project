import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("test") + "\n\n"
    return text

def chunk_text(text, chunk_size = 512, chunk_overlap = 50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    
    return text_splitter.split(text)

def create_faiss_index(chunks, index_path = "classification/data/faiss_index"):
    embedding_model = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embedding_model)
    vector_store.save_local(index_path)
    print("Index created and saved at", index_path)