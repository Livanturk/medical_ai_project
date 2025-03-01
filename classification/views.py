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
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Load the pre-trained model
model_path = os.path.join(settings.BASE_DIR, 'classification/models/brain_tumor_model.keras')
model = tf.keras.models.load_model(model_path)

labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']  # Correct class mapping

# Initialize Llama3 using Ollama (Runs locally)
llm = OllamaLLM(model="llama3")

# Define prompt template for medical questions
prompt = PromptTemplate(
    input_variables=["diagnosis"],
    template="Given a patient diagnosed with {diagnosis}, generate 5 important questions a doctor should ask to assess their condition."
)

class BrainTumorPredictionView(APIView):
    def post(self, request, *args, **kwargs):
        file = request.FILES.get('image')
        if not file:
            return Response({'error': 'No image provided.'}, status=status.HTTP_400_BAD_REQUEST)

        # Save the uploaded image
        file_name = default_storage.save(file.name, file)
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        # Preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence_score = float(np.max(predictions))  # Extract confidence score

        # Generate medical questions using Llama3
        diagnosis = labels[predicted_class]
        formatted_prompt = prompt.format(diagnosis=diagnosis)
        llm_response = llm(formatted_prompt)  # Call Llama3 directly

        result = {
            "prediction": diagnosis,
            "confidence": confidence_score,
            "probabilities": {labels[i]: float(predictions[0][i]) for i in range(len(labels))},
            "questions_for_doctor": llm_response
        }

        return Response(result)
