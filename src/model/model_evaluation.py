import numpy as np
import pandas as pd
import pickle
import yaml
import logging
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models import infer_signature


# Logging Configuration
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True) # Fill missing values with empty string.
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the CSV file at {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load data due to some unexpected error: {e}")
        raise


def load_model(model_path: str):
    """Load the trained model from a file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model due to some unexpected error: {e}")
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the TF-IDF vectorizer from a file."""
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.debug(f"TF-IDF vectorizer loaded successfully from {vectorizer_path}")
        return vectorizer
    except Exception as e:
        logger.error(f"Failed to load TF-IDF vectorizer due to some unexpected error: {e}")
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug("Parameters retrieved from %s", params_path)
        return params

    except Exception as e:
        logger.error("Failed to load parameters: %s", e)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        
        # Calculate classification metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        # logger.debug("Classification report: %s", report)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # logger.debug("Confusion matrix: %s", cm)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # logger.debug("Accuracy: %s", accuracy)
        
        # # Log classification metrics and confusion matrix
        # mlflow.log_metric("accuracy", accuracy)
        # mlflow.log_artifact("classification_report.txt", report)
        # mlflow.log_artifact("confusion_matrix.txt", cm)
        
        logger.debug("Model evaluation completed successfully.")

        return report, cm, accuracy
    except Exception as e:
        logger.error(f"Failed to evaluate model due to some unexpected error: {e}")
        raise


def log_confusion_matrix(cm, dataset_name):
    """Log the confusion matrix as an image artifact."""
    try:
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.savefig(f'confusion_matrix_{dataset_name}.png')
        mlflow.log_artifact(f'confusion_matrix_{dataset_name}.png')
        plt.close()
        logger.debug("Confusion matrix logged successfully.")

    except Exception as e:
        logger.error(f"Failed to log confusion matrix due to some unexpected error: {e}")
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model information to a JSON file."""
    try:
        # Create a dictionary with the info to save.
        model_info = {
            'run_id': run_id,
            'model_path': model_path,
        }
        # Save the dictionary to a JSON file.
        with open(file_path, 'w') as f:
            json.dump(model_info, f)
        logger.debug(f"Model information saved successfully to {file_path}.")

    except Exception as e:
        logger.error(f"Failed to save model information due to some unexpected error: {e}")
        raise


def main():
    """Main function to e   valuate the model."""
    load_dotenv()
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("dvc-pipeline-runs")

    with mlflow.start_run() as run:
        try:
            # Load the parameters from YAML file
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params_path = os.path.join(root_dir, "params.yaml")
            params = load_params(params_path)

            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Load the model and vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Load the test data for signature reference
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_preprocessed.csv'))
            
            # Prepare the test data
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Create a DataFrame for signature inference (using first few rows as an example)
            input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())

            # Infer the signature
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))

            # Log the model with signature
            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature,
                input_example=input_example,
            )

            # Save model info
            artifact_uri = mlflow.get_artifact_uri()
            model_path = f"{artifact_uri}/lgbm_model"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            # Log the vectorizer as an artifact
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Evaluate the model
            report, cm, accuracy = evaluate_model(model, X_test_tfidf, y_test)
            
            # Log the classification report metrics for the test data
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if metric != 'support':
                            mlflow.log_metrics({
                                f"test_{label}_precision": metrics['precision'],
                                f"test_{label}_recall": metrics['recall'],
                                f"test_{label}_f1": metrics['f1-score'],
                            })

            # Log the confusion matrix
            log_confusion_matrix(cm, "Test Data")

            # Add important tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            raise

if __name__ == "__main__":
    main()