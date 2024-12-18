from fastapi import FastAPI
from pydantic import BaseModel
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Model identifier on Hugging Face
model_id = "MESSItom/BERT-review-sentiment-analysis"  # Replace with your actual model identifier

# Load the model and tokenizer from Hugging Face
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize FastAPI app
app = FastAPI()

# Define input schema using Pydantic
class Review(BaseModel):
    text: str

# Prediction function
def predict_sentiment(text):
    class_names = ['positive', 'neutral', 'negative']
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted label (0 or 1)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Map predicted class to sentiment label

    sentiment = class_names[predicted_class]
    return sentiment


# API endpoint for prediction
@app.post("/predict/")
async def get_sentiment(review: Review):
    sentiment = predict_sentiment(review.text)
    return {"text": review.text, "sentiment": sentiment}
# 
# Welcome message at root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Sentiment Analysis API. Send your review to /predict/ to get sentiment!"}

