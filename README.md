# Fine-Tuning BERT for Sentiment Analysis

![Project Banner](doc/banner.webp)

![GitHub repo size](https://img.shields.io/github/repo-size/messi10tom/Fine-Tuning-BERT-for-Sentiment-Analysis) ![GitHub last commit](https://img.shields.io/github/last-commit/messi10tom/Fine-Tuning-BERT-for-Sentiment-Analysis) ![GitHub issues](https://img.shields.io/github/issues/messi10tom/Fine-Tuning-BERT-for-Sentiment-Analysis) ![GitHub license](https://img.shields.io/github/license/messi10tom/Fine-Tuning-BERT-for-Sentiment-Analysis)

## Problem Statement

The task involves fine-tuning a pre-trained language model, such as BERT, to perform sentiment analysis on a custom dataset. The dataset contains student reviews about campus events or amenities, labeled by sentiment (e.g., positive, negative, neutral). The objective is to train the model to effectively classify the sentiments while maintaining high performance metrics like accuracy.

## Approach

1. **Data Preparation:**
   - Synthetically generated data using ```Groq/llama-3.3-70b-versatile```
   - Preprocessed the data to clean and tokenize the review texts.
   - Split the dataset into training, validation, and test sets.

2. **Model Fine-Tuning:**
   - Used Hugging Face's Transformers library to fine-tune the BERT model.
   - Customized the model's architecture by adding a classification head.


3. **Evaluation:**
   - Measured accuracy and confusion matrix metrics.
   - Visualized sentiment distributions using graphs and charts.

4. **Deployment (Bonus):**
   - Built a REST API using FastAPI to enable users to submit reviews and receive sentiment predictions.

## Results

### Performance Metrics

| Metric         | Value  |
|----------------|--------|
| Train loss       | 0.7093283434708914  |
|Eval loss| 0.4079654812812805|

### Confusion Matrix

![Confusion Matrix](./doc/confusion%20matrix.png)

### Sentiment Distribution

![Sentiment Distribution](./doc/distribution.png)

## Challenges

1. **Data Collection Issues:**
    - Challenge: It was difficult to obtain a comprehensive dataset of student reviews about campus events or amenities.
    - Solution: Overcame this by synthetically generating data using ```Groq/llama-3.3-70b-versatile```.

2. **Model Overfitting:**
   - Solution: Used dropout layers and reduced the learning rate.

3. **API Deployment:**
   - Solution: Researched FastAPI and integrated it effectively with the trained model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/messi10tom/Fine-Tuning-BERT-for-Sentiment-Analysis
   ```
2. Navigate to the project directory:
   ```bash
   cd Fine-Tuning-BERT-for-Sentiment-Analysis
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app
   ```bash
   uvicorn main:app --reload
   ```
   Wait for model to load

5. Go to ```http://127.0.0.1:8000/docs```

6. You’ll see a Swagger UI page with the POST /predict/  endpoint.

7. Expand the POST /predict/ section, and you will find a text box where the user can input their review. Enter a review like this:
   ```json
   Copy code
   {
   "text": "The collage is absolutely fantastic!"
   }
   ```
   Click the Execute button.
   You will get a response containing the sentiment.

## Model Description

The model is based on BERT (Bidirectional Encoder Representations from Transformers). Below is a high-level architecture:


1. Input Layer: Accepts tokenized review text.
2. Pre-trained BERT Layers: Encodes contextual information from the text.
3. Classification Head: Adds fully connected layers to classify sentiments.

## Screenshots

### Training - Validation Logs
![Training Logs](./doc/train%20log.png)


## Additional Features

1. REST API:
   - Submit a review to receive sentiment predictions.
   - Endpoint: `/predict`
2. Hugging Face Model:
      - The fine-tuned BERT model is also deployed on Hugging Face for easy access and inference.
      - You can find the model [here](https://huggingface.co/MESSItom/BERT-review-sentiment-analysis).

   ### Model Card for MESSItom/BERT-review-sentiment-analysis

   This model is fine-tuned from BERT to perform sentiment analysis on a custom dataset containing student reviews about campus events or amenities. The objective is to classify the sentiments (positive, negative, neutral) while maintaining high performance metrics like accuracy.

   #### Model Details

   - **Developed by:** Messy Tom Binoy
   - **Funded by:** No funding, self-funded
   - **Shared by:** Messy Tom Binoy
   - **Model type:** BERT
   - **Language(s) (NLP):** English
   - **License:** MIT
   - **Finetuned from model:** google-bert/bert-base-uncased

   #### Model Sources

   - **Repository:** [GitHub Repository](https://github.com/messi10tom/Fine-Tuning-BERT-for-Sentiment-Analysis/tree/main)
   - **Demo:** [GitHub Demo](https://github.com/messi10tom/Fine-Tuning-BERT-for-Sentiment-Analysis/tree/main)

   #### Uses

   - **Direct Use:** Sentiment classification of student reviews about campus events or amenities.
   - **Downstream Use:** Further fine-tuning for other sentiment analysis tasks or integration into larger applications for sentiment classification.
   - **Out-of-Scope Use:** Not suitable for tasks outside sentiment analysis, such as language translation or text generation.

   #### Bias, Risks, and Limitations

   The model may inherit biases from the pre-trained BERT model and the custom dataset used for fine-tuning. It may not perform well on reviews that are significantly different from the training data.

   #### Recommendations

   Users should be aware of the potential biases and limitations of the model. It is recommended to evaluate the model on a diverse set of reviews to understand its performance and limitations.

   #### How to Get Started with the Model

   ```python
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   import torch

   model_id = "MESSItom/BERT-review-sentiment-analysis"

   model = AutoModelForSequenceClassification.from_pretrained(model_id)
   tokenizer = AutoTokenizer.from_pretrained(model_id)

   def predict_sentiment(text):
       inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
       with torch.no_grad():
           outputs = model(**inputs)
       logits = outputs.logits
       predicted_class = torch.argmax(logits, dim=-1).item()
       class_names = ['positive', 'neutral', 'negative']
       sentiment = class_names[predicted_class]
       return sentiment
   ```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
