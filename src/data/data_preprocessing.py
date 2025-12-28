import numpy as np
import pandas as pd
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Logging Configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Define the preprocessing function
def preprocess_comment(comment: str) -> str:
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

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the text data in the dataframe."""
    try:
        # Apply the preprocessing function to the 'clean_comment' column
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug("Text normalization completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error in normalizing text: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory: {interim_data_path}")

        # Create the data/raw directory if it doesn't exist.
        os.makedirs(interim_data_path, exist_ok=True)
        logger.debug(f"Directory {interim_data_path} created successfully or already exists.")

        # Save the train and test datasets.
        train_data.to_csv(os.path.join(interim_data_path, 'train_preprocessed.csv'), index=False)
        test_data.to_csv(os.path.join(interim_data_path, 'test_preprocessed.csv'), index=False)

        logger.debug(f"Processed data saved successfully to {interim_data_path}")
    except Exception as e:
        logger.error(f"Failed to save data due to some unexpected error: {e}")
        raise

def main():
    try:
        logger.debug("Started data preprocessing.")

        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')

        logger.debug("Data loaded successfully.")

        # Preprocess the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save the processed data
        save_data(train_data=train_processed_data, test_data=test_processed_data, data_path='./data')

        logger.debug("Data preprocessing completed successfully.")

    except Exception as e:
        logger.error(f"Failed to complete the data preprocessing process: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
