import streamlit as st
from models import MODELS

# GLOBALS
DATA_TYPE_SIZES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
}
OPTIMIZERS = {
    "AdamW": 8,
    "Quantized AdamW": 2,
    "SGD": 4,
}
DATA_TYPES = list(DATA_TYPE_SIZES.keys())
PARAMETERS = {
    "param_size": "param_size",
    "precision": "torch_dtype",
    "hidden_size": "hidden_size",
    "num_hidden_layers": "num_hidden_layers",
    "num_attention_heads": "num_attention_heads",
    "num_key_value_heads": "num_key_value_heads",
}

# Streamlit UI
st.title("LLM Memory Requirements")


def set_values():
    if st.session_state.model in MODELS:
        model_info = MODELS[st.session_state.model]
        for param in PARAMETERS:
            if PARAMETERS[param] in model_info:
                if param == "precision":
                    st.session_state[param] = model_info[PARAMETERS[param]]
                else:
                    st.session_state[param] = model_info[PARAMETERS[param]]
            else:
                st.session_state[param] = None
    else:
        for param in PARAMETERS:
            st.session_state[param] = None


model = st.sidebar.selectbox(
    "Model", list(MODELS.keys()), index=None, on_change=set_values, key="model"
)


param_size = st.sidebar.number_input(
    "Number of parameters (in billions)",
    min_value=0,
    step=1,
    value=None,
    key="param_size",
)
precision = st.sidebar.selectbox("Precision", DATA_TYPES, index=None, key="precision")
batch_size = st.sidebar.number_input(
    "Batch Size", min_value=0, step=1, value=1, key="batch_size"
)
sequence_length = st.sidebar.number_input(
    "Sequence Length", min_value=0, step=1, value=2048, key="sequence_length"
)
hidden_size = st.sidebar.number_input(
    "Hidden Size", min_value=0, step=1, value=None, key="hidden_size"
)
num_hidden_layers = st.sidebar.number_input(
    "Number of Layers", min_value=0, step=1, value=None, key="num_hidden_layers"
)
num_attention_heads = st.sidebar.number_input(
    "Number of Attention Heads",
    min_value=0,
    step=1,
    value=None,
    key="num_attention_heads",
)


# Memory Calculation
def get_memory(bytes):
    # Input is in Bytes
    if bytes == 0:
        return ""
    elif bytes < 1024:
        return f"{bytes} Bytes"
    elif bytes < 1024**2:
        return f"{bytes / 1024:.2f} KB"
    elif bytes < 1024**3:
        return f"{bytes / (1024**2):.2f} MB"
    elif bytes < 1024**4:
        return f"{bytes / (1024**3):.2f} GB"
    else:
        return f"{bytes / (1024**4):.2f} TB"


def get_model_weights(param_size, precision):
    try:
        return param_size * DATA_TYPE_SIZES[precision] * (10**9)
    except:
        return 0


def get_kv_cache(batch_size, sequence_length, hidden_size, num_hidden_layers):
    try:
        return (
            2
            * batch_size
            * sequence_length
            * num_hidden_layers
            * hidden_size
            * DATA_TYPE_SIZES[precision]
        )
    except:
        return 0


def get_activation_memory(batch_size, sequence_length, hidden_size):
    precision = "float32"
    try:
        return batch_size * sequence_length * hidden_size * DATA_TYPE_SIZES[precision]
    except:
        return 0


def get_optimizer_memory(param_size, optimizer):
    try:
        return OPTIMIZERS[optimizer] * param_size * (10**9)
    except:
        return 0


def get_gradient_memory(param_size, precision):
    precision = "float32"
    try:
        return DATA_TYPE_SIZES[precision] * param_size
    except:
        return 0


# General Memory
model_weights = get_model_weights(param_size, precision)
kv_cache = get_kv_cache(batch_size, sequence_length, hidden_size, num_hidden_layers)
activation_memory = get_activation_memory(batch_size, sequence_length, hidden_size)
st.write(f"- Model Weights: {get_memory(model_weights)}")
st.write(f"- KV Cache: {get_memory(kv_cache)}")
st.write(f"- Activation Memory: {get_memory(activation_memory)}")

inference, training = st.columns(2)
inference.markdown("## Inference")
training.markdown("## Training")

# Inference Memory
total_inference_memory = model_weights + kv_cache + activation_memory
inference.write(f"**Total Inference Memory**: {get_memory(total_inference_memory)}")


# Training Memory
optimizer = training.selectbox("Optimizer", list(OPTIMIZERS.keys()), key="optimizer")
trainable_parameters = training.slider(
    "Percentage of trainable parameters", 0, 100, 100, key="trainable_params"
)
optimizer_memory = (
    get_optimizer_memory(param_size, optimizer) * trainable_parameters / 100
)
gradients_memory = (
    get_gradient_memory(param_size, precision) * trainable_parameters / 100
)
total_training_memory = (
    model_weights + kv_cache + activation_memory + optimizer_memory + gradients_memory
)
training.write(f"**Total Memory**: {get_memory(total_training_memory)}")
training.write(f"- Optimizer Memory: {get_memory(optimizer_memory)}")
training.write(f"- Gradients Memory: {get_memory(gradients_memory)}")
