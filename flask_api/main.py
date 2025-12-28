import matplotlib
matplotlib.use('Agg') # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert the comment to lowercase
        comment = comment.lower()

        # Remove leading and trailing whitespace
        comment = comment.strip()
    
        # Replace newlines with spaces
        comment = re.sub(r'\n', ' ', comment)

        # Remove special characters and numbers
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        raise

# TODO: Loading and predicting model from mlflow model registry


# Load the model and vectorizer from the model registry and local storage
# def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
#     # Set MLFLow tracking URI to your server
#     mlflow_trcking_uri = os.getenv("MLFLOW_TRACKING_URI")
#     mlflow.set_tracking_uri(mlflow_trcking_uri)

#     client = MlflowClient()
#     model_uri = f"models:/{model_name}/{model_version}"
#     model = mlflow.pyfunc.load_model(model_uri)

#     with open(vectorizer_path, 'rb') as f:
#         vectorizer = pickle.load(f)

#     return model, vectorizer

# Initialize the model and vectorizer
# model, vectorizer = load_model_and_vectorizer(model_name='yt_chrome_plugin_model', model_version='1', vectorizer_path='tfidf_vectorizer.pkl')

    
# To load the model and vectorizer from local
def load_model_and_vectorizer_from_local(model_path, vectorizer_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer_from_local(model_path='lgbm_model.pkl', vectorizer_path='tfidf_vectorizer.pkl')


@app.route('/')
def home():
    return "Welcome to the YouTube Comment Analyzer Flask API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({'error': 'No comments provided'}), 400

    try:
        # Preprocess the comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform the comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments = transformed_comments.toarray() # Convert to dense array
        
        # Make predictions
        predictions = model.predict(dense_comments).tolist()

        # Convert the predictions to a list of strings for consistency
        # predictions = [str(pred) for pred in predictions]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response), 200


if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
        

# TODO: Testing of the /predict api using Postman