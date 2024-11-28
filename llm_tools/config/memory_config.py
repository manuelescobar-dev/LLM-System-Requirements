import streamlit as st
import os
import json

# Data type sizes in bytes
DATA_TYPE_SIZES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
}

# Optimizer memory multipliers
OPTIMIZERS = {
    "AdamW": 8,
    "Quantized AdamW": 2,
    "SGD": 4,
}

# Available data types
DATA_TYPES = list(DATA_TYPE_SIZES.keys())

# Model parameters mapping
PARAMETERS = {
    "model_size": "model_size",
    "precision": "torch_dtype",
    "hidden_size": "hidden_size",
    "num_hidden_layers": "num_hidden_layers",
    "num_attention_heads": "num_attention_heads",
    "num_key_value_heads": "num_key_value_heads",
}


@st.cache_data
def load_predefined_models() -> dict:
    """Load model configurations from the 'predefined_models' folder."""
    models = {}
    for model_file in os.listdir("models"):
        if model_file.endswith(".json"):
            with open(os.path.join("models", model_file), "r") as f:
                models[model_file[:-5]] = json.load(f)
    return models
