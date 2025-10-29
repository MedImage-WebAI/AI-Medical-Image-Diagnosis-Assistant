# model_loader.py
from huggingface_hub import hf_hub_download
import tensorflow as tf

# Dictionary mapping disease -> Hugging Face repo (updated to your account)
MODEL_REPOS = {
    "pneumonia": "Saitama30/pneumonia-model",
    "malaria": "Saitama30/malaria-model",
    "dental_cavity": "Saitama30/dental-cavity-model",
    "brain_tumor": "Saitama30/brain-tumor-model"
}

def load_model_from_hf(disease_name, filename):
    """
    Downloads and loads a model from Hugging Face based on disease name.
    
    Args:
        disease_name (str): Key name like 'pneumonia', 'malaria', etc.
        filename (str): Model file name in Hugging Face repo, e.g., 'model.h5' or 'model.keras'
    Returns:
        TensorFlow model instance
    """
    if disease_name not in MODEL_REPOS:
        raise ValueError(f"Unknown disease name '{disease_name}'. Available: {list(MODEL_REPOS.keys())}")

    repo_id = MODEL_REPOS[disease_name]
    print(f"📦 Downloading {disease_name} model from {repo_id} ...")

    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = tf.keras.models.load_model(model_path)

    print(f"✅ Loaded {disease_name} model successfully!")
    return model
