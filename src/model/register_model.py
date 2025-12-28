import mlflow
import logging
import os
from dotenv import load_dotenv
import json

load_dotenv()

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(mlflow_tracking_uri)

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


def load_model_info(file_path: str) -> dict:
    """Load the model information from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            model_info = json.load(f)
        logger.debug(f"Model information loaded successfully from {file_path}")
        return model_info
    except FileNotFoundError:
        logger.error(f"Model information file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model information: {e}")
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLFlow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.debug(f"Model {model_name} version {model_version.version} registered and transitioned to Staging stage.")

    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise


def main():
    """Main function to register the model."""
    try:
        # Load the model information from the JSON file
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = 'yt_chrome_plugin_model'
        # Register the model
        register_model(model_name, model_info)
        
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise

if __name__ == "__main__":
    main()
