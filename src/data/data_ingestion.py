import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import yaml
import logging

"""
Data Ingestion Module
=====================

This module is responsible for:
1. Loading configuration parameters.
2. Ingesting raw data from a remote URL.
3. Preprocessing the raw data (cleaning, duplicate removal).
4. Splitting the data into training and testing sets.
5. Saving the processed datasets to the local file system.
"""


# Logging Configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.

    Args:
        params_path (str): The file path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug("Parameters retrieved from %s", params_path)
        return params

    except FileNotFoundError:
        logger.error(f"File not found at {params_path}")
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        logger.error("Failed to load parameters: %s", e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        data_url (str): The URL or path to the CSV data file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug(f"Data loaded successfully from {data_url}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the CSV file at {data_url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load data due to some unexpected error: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing data, duplicates and empty strings.

    Args:
        df (pd.DataFrame): The raw dataframe.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    try:
        # Removing missing values.
        df = df.dropna()
        # Removing duplicate rows.
        df = df.drop_duplicates()
        # Removing empty strings.
        df = df[df['clean_comment'].str.strip() != '']

        logger.debug("Data preprocessing completed successfully.")
        return df
    except KeyError as e:
        logger.error(f"Missing column in the dataframe: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to preprocess data due to some unexpected error: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save the train and test datasets, creating the raw folder if it doesn't exist.

    Args:
        train_data (pd.DataFrame): The training dataset.
        test_data (pd.DataFrame): The testing dataset.
        data_path (str): The base path where the 'raw' directory will be created.
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')

        # Create the data/raw directory if it doesn't exist.
        os.makedirs(raw_data_path, exist_ok=True)

        # Save the train and test datasets.
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)

        logger.debug(f"Data saved successfully to {raw_data_path}")
    except Exception as e:
        logger.error(f"Failed to save data due to some unexpected error: {e}")
        raise

def main():
    """
    Main function to execute the data ingestion pipeline.
    
    This function:
    - Loads parameters.
    - Loads data from source.
    - Preprocesses the data.
    - Splits data into train and test sets.
    - Saves the datasets locally.
    """
    try:
        # Load parameters from params.yml in the root directory
        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = params['data_ingestion']['test_size']

        # Load data from the specified URL.
        df = load_data(data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')

        # Preprocess the data.
        final_df = preprocess_data(df)

        # Split the data into train and test sets.
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        # Save the train and test datasets and create the raw folder if it doesn't exist.
        save_data(train_data=train_data, test_data=test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))

    except Exception as e:
        logger.error(f"Failed to complete the data ingestion process: {e}")
        print(f"Error: {e}")



if __name__ == "__main__":
    main()
