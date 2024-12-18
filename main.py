from transformers import BertTokenizer, BertForSequenceClassification
import torch

# the directory where your model is saved
model_dir = './results'

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

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


