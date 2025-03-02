from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.core.files.storage import default_storage
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

model_path = os.path.join(settings.BASE_DIR, 'classification/models/brain_tumor_model.keras')
model = tf.keras.models.load_model(model_path)

labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']

chroma_client = chromadb.PersistentClient(path="classification/data/chroma_db")
collection = chroma_client.get_or_create_collection(name="brain_tumor_data")

print("Number of stored documents:", collection.count())

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

llm = OllamaLLM(model="llama3.2:3b", base_url="http://127.0.0.1:11434")

class BrainTumorPredictionView(APIView):
    def post(self, request, *args, **kwargs):
        file = request.FILES.get('image')
        if not file:
            return Response({'error': 'No image provided.'}, status=status.HTTP_400_BAD_REQUEST)

        # Save uploaded image
        file_name = default_storage.save(file.name, file)
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        # Preprocess image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence_score = float(np.max(predictions))

        # Retrieve relevant medical text using ChromaDB
        diagnosis = labels[predicted_class]
        query = f"What are the symptoms and treatment options for {diagnosis}?"
        query_embedding = embedding_model.encode(query).tolist()

        # Perform retrieval from ChromaDB
        # Perform retrieval from ChromaDB
        # Perform retrieval from ChromaDB
        search_results = collection.query(query_texts=[query], n_results=3)  # ✅ Fix: Use `query_texts` for retrieval

        # Ensure retrieval results are valid
        if search_results and "documents" in search_results and search_results["documents"]:
            retrieved_docs = [doc for doc in search_results["documents"][0] if doc is not None]  # ✅ Remove None values

            # Join retrieved text
            retrieved_text = "\n\n".join(retrieved_docs) if retrieved_docs else "No relevant medical information found."
        else:
            retrieved_text = "No relevant medical information found."


        # Clean the retrieved text
        retrieved_text = retrieved_text.replace("\n", " ").replace("  ", " ").strip()
        retrieved_text = retrieved_text[:1000]  # Limit to 1000 characters for readability


        # Generate medical questions using LLaMA3
        formatted_prompt = f"""
        You are an AI medical assistant. Use the following medical text to generate questions a doctor should ask a patient with {diagnosis}. 

        Medical Text:
        {retrieved_text}

        Now, based on this medical information, what are the 5 most important questions a doctor should ask?
        """
        llm_response = llm.invoke(formatted_prompt)

        return Response({
            "prediction": diagnosis,
            "confidence": confidence_score,
            "retrieved_info": retrieved_text,
            "questions_for_doctor": llm_response
        })
