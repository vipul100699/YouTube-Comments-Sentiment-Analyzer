import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

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

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
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

def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """Apply TF-IDF vectorization to the training data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        # Perform TF-IDF transformation
        X_train_tfidf = vectorizer.fit_transform(X_train)

        logger.debug(f"TF-IDF transformation completed successfully. Training data shape: {X_train_tfidf.shape}")

        # Save the vectorizer in the root directory
        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug("TF-IDF vectorizerization applied with trigrams and data transformed successfully.")
        return X_train_tfidf, y_train
    except Exception as e:
        logger.error(f"Failed to apply TF-IDF vectorization due to some unexpected error: {e}")
        raise


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier."""
    try:
        best_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            is_unbalance=True,
            class_weight='balanced',
            reg_alpha=0.1,  # L1 regularization.
            reg_lambda=0.1, # L2 regularization.
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )

        best_model.fit(X_train, y_train)
        logger.debug("LightGBM model trained successfully.")
        return best_model
    except Exception as e:
        logger.error(f"Failed to train LightGBM model due to some unexpected error: {e}")
        raise


def save_model(model: lgb.LGBMClassifier, model_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved successfully to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model due to some unexpected error: {e}")
        raise


def get_root_directory() -> str:
    """Get the root directory of the project."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        # Get the root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # Load parameters from the root directory
        params = load_params(params_path=os.path.join(root_dir, 'params.yaml'))
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])
        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        # Load the preprocessed training data from the interim directory
        train_data = load_data(file_path=os.path.join(root_dir, 'data/interim/train_preprocessed.csv'))

        # Apply TF-IDF vectorization to the training data
        X_train_tfidf, y_train = apply_tfidf(train_data=train_data, max_features=max_features, ngram_range=ngram_range)

        # Train the LightGBM model using the Hyperparameters from params.yaml
        best_model = train_lgbm(X_train=X_train_tfidf, y_train=y_train, learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)

        # Save the trained model to the root directory
        save_model(model=best_model, model_path=os.path.join(root_dir, 'lgbm_model.pkl'))

    except Exception as e:
        logger.error(f"Failed to complete the model building process: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
