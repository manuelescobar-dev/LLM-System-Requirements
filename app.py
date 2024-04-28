import streamlit as st
from models import MODELS

st.set_page_config(page_title="LLM Memory Requirements")

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
    # Update the values based on the selected model
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


# Model Selection
model = st.sidebar.selectbox(
    "Model", list(MODELS.keys()), index=None, on_change=set_values, key="model"
)

# Parameters
param_size = st.sidebar.number_input(
    "Number of parameters (in billions)",
    min_value=0,
    step=1,
    value=None,
    key="param_size",
    help="Number of parameters in the model in billions",
)
precision = st.sidebar.selectbox(
    "Precision",
    DATA_TYPES,
    index=None,
    key="precision",
    help="Data type used (int 8 and int 4 are for quantization)",
)
batch_size = st.sidebar.number_input(
    "Batch Size",
    min_value=0,
    step=1,
    value=1,
    key="batch_size",
)
sequence_length = st.sidebar.number_input(
    "Sequence Length",
    min_value=0,
    step=1,
    value=2048,
    key="sequence_length",
    help="Number of tokens in the input sequence.",
)
hidden_size = st.sidebar.number_input(
    "Hidden Size",
    min_value=0,
    step=1,
    value=None,
    key="hidden_size",
    help="Size of the hidden layer (given by the model card).",
)
num_hidden_layers = st.sidebar.number_input(
    "Number of Layers",
    min_value=0,
    step=1,
    value=None,
    key="num_hidden_layers",
    help="Number of layers in the model (given by the model card).",
)
num_attention_heads = st.sidebar.number_input(
    "Number of Attention Heads",
    min_value=0,
    step=1,
    value=None,
    key="num_attention_heads",
    help="Number of attention heads in the model (given by the model card).",
)


# Memory Calculations
def get_memory(*args):
    total = 0
    warning = False
    for arg in args:
        if arg > 0:
            total += arg
        else:
            warning = True
    # Convert bytes to human readable format
    if total == 0:
        result = ""
    elif total < 1024:
        result = f"{total} Bytes"
    elif total < 1024**2:
        result = f"{total / 1024:.2f} KB"
    elif total < 1024**3:
        result = f"{total / (1024**2):.2f} MB"
    elif total < 1024**4:
        result = f"{total / (1024**3):.2f} GB"
    else:
        result = f"{total / (1024**4):.2f} TB"
    result += " \* " if warning else ""
    return result


def get_model_weights(param_size, precision):
    # Calculate the memory required for model weights
    try:
        return param_size * DATA_TYPE_SIZES[precision] * (10**9)
    except:
        return 0


def get_kv_cache(batch_size, sequence_length, hidden_size, num_hidden_layers):
    # Calculate the memory required for key-value cache
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
    # Calculate the memory required for activations
    precision = "float32"
    try:
        return (
            batch_size
            * sequence_length
            * hidden_size
            * (34 + (5 * sequence_length * num_attention_heads) / (hidden_size))
            * DATA_TYPE_SIZES[precision]
        )
    except:
        return 0


def get_optimizer_memory(param_size, optimizer):
    # Calculate the memory required for optimizer
    try:
        return OPTIMIZERS[optimizer] * param_size * (10**9)
    except:
        return 0


def get_gradient_memory(param_size, precision):
    # Calculate the memory required for gradients
    precision = "float32"
    try:
        return DATA_TYPE_SIZES[precision] * param_size * (10**9)
    except:
        return 0


# General Memory
model_weights = get_model_weights(param_size, precision)
kv_cache = get_kv_cache(batch_size, sequence_length, hidden_size, num_hidden_layers)
activation_memory = get_activation_memory(batch_size, sequence_length, hidden_size)
st.write(f"- **Model Weights**: {get_memory(model_weights)}")
st.write(f"- **KV Cache**: {get_memory(kv_cache)}")
st.write(f"- **Activation Memory**: {get_memory(activation_memory)}")

st.markdown("## Inference")

# Inference Memory
st.write(
    f"**Total Inference Memory**: {get_memory(model_weights,kv_cache,activation_memory)}"
)


# Training Memory
st.markdown("## Training")
training1, training2 = st.columns(2)
training2 = training2.container(border=True)
optimizer = training2.selectbox("Optimizer", list(OPTIMIZERS.keys()), key="optimizer")
trainable_parameters = training2.slider(
    "Percentage of trainable parameters", 0, 100, 100, key="trainable_params"
)
optimizer_memory = (
    get_optimizer_memory(param_size, optimizer) * trainable_parameters / 100
)
gradients_memory = (
    get_gradient_memory(param_size, precision) * trainable_parameters / 100
)
training1.write(
    f"**Total Training Memory**: {get_memory(model_weights,kv_cache,activation_memory,optimizer_memory,gradients_memory)}"
)
training1.write(f"- **Optimizer Memory**: {get_memory(optimizer_memory)}")
training1.write(f"- **Gradients Memory**: {get_memory(gradients_memory)}")

st.warning("\* Missing information.")
