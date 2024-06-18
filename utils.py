import streamlit as st
from config import DATA_TYPES, PARAMETERS, DATA_TYPE_SIZES, OPTIMIZERS


# ----------------- Memory Functions ----------------- #
@st.cache_data
def get_memory(*args):
    """Convert total memory from bytes to human-readable format."""
    total = 0
    warning = False
    for arg in args:
        if arg > 0:
            total += arg
        else:
            warning = True
    # Convert bytes to human-readable format
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
    result += " * " if warning else ""
    return result


@st.cache_data
def get_model_weights(model_size, precision):
    """Calculate the memory required for model weights."""
    try:
        return model_size * DATA_TYPE_SIZES[precision] * (10**9)
    except:
        return 0


@st.cache_data
def get_kv_cache(
    precision, batch_size, sequence_length, hidden_size, num_hidden_layers
):
    """Calculate the memory required for key-value cache."""
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


@st.cache_data
def get_activation_memory(
    batch_size, sequence_length, hidden_size, num_attention_heads
):
    """Calculate the memory required for activations."""
    precision = "float32"
    try:
        return (
            batch_size
            * sequence_length
            * hidden_size
            * (34 + (5 * sequence_length * num_attention_heads) / hidden_size)
            * DATA_TYPE_SIZES[precision]
        )
    except:
        return 0


@st.cache_data
def get_optimizer_memory(model_size, optimizer):
    """Calculate the memory required for optimizer."""
    try:
        return OPTIMIZERS[optimizer] * model_size * (10**9)
    except:
        return 0


@st.cache_data
def get_gradient_memory(model_size, precision):
    """Calculate the memory required for gradients."""
    precision = "float32"
    try:
        return DATA_TYPE_SIZES[precision] * model_size * (10**9)
    except:
        return 0


@st.cache_data
def calculate_inference_memory(
    model_size,
    precision,
    batch_size,
    sequence_length,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
):
    """Calculate the total memory required for inference."""
    model_weights = get_model_weights(model_size, precision)
    kv_cache = get_kv_cache(
        precision, batch_size, sequence_length, hidden_size, num_hidden_layers
    )
    activation_memory = get_activation_memory(
        batch_size, sequence_length, hidden_size, num_attention_heads
    )
    return {
        "model_weights": get_memory(model_weights),
        "kv_cache": get_memory(kv_cache),
        "activation_memory": get_memory(activation_memory),
        "inference_memory": get_memory(model_weights, kv_cache, activation_memory),
    }


@st.cache_data
def calculate_training_memory(
    model_size,
    precision,
    batch_size,
    sequence_length,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
    optimizer,
    trainable_parameters,
):
    """Calculate the total memory required for training."""
    model_weights = get_model_weights(model_size, precision)
    kv_cache = get_kv_cache(
        precision, batch_size, sequence_length, hidden_size, num_hidden_layers
    )
    activation_memory = get_activation_memory(
        batch_size, sequence_length, hidden_size, num_attention_heads
    )
    optimizer_memory = (
        get_optimizer_memory(model_size, optimizer) * trainable_parameters / 100
    )
    gradients_memory = (
        get_gradient_memory(model_size, precision) * trainable_parameters / 100
    )

    return {
        "model_weights": get_memory(model_weights),
        "kv_cache": get_memory(kv_cache),
        "activation_memory": get_memory(activation_memory),
        "optimizer_memory": get_memory(optimizer_memory),
        "gradients_memory": get_memory(gradients_memory),
        "training_memory": get_memory(
            model_weights,
            kv_cache,
            activation_memory,
            optimizer_memory,
            gradients_memory,
        ),
    }
