**Available on:** [https://llm-system-requirements.streamlit.app](https://llm-system-requirements.streamlit.app)

# LLM System Requirements Calculator
Welcome to the LLM System Requirements Calculator, an open-source tool designed to help estimate the system requirements for running Large Language Models (LLMs).

The LLM System Requirements Calculator aims to address this challenge by providing a user-friendly interface for estimating the memory resources needed to run LLMs for both inference and training tasks.

All the initial formulas and explanations can be found it the following article but these may change as the tool evolves:

[**Memory Requirements for LLM Training and Inference**](https://medium.com/@manuelescobar-dev/memory-requirements-for-llm-training-and-inference-97e4ab08091b)

---

**Disclaimer**: The LLM System Requirements Calculator is a tool for estimating memory requirements for LLMs and the only real way of estimating the exact memory is trying things out. The tool is provided as-is, without any warranties or guarantees.

---
# Memory Requirements for LLM Training and Inference

LLMs are characterized by being very computationally demanding, having billions of parameters and being trained on terabytes of data. This has been possible due to the latest advancements in computational power of the last decade and with new optimization techniques and architectures. Despite these advancements, LLMs are still far from being generally accessible due to computational limitations and proprietary models. For example, training GPT-4 is estimated to cost around 100 million dollars.

However, thanks to open-source models like Llama 3 and others, all types of companies and persons can now use and personalize these models. These models come in various sizes, with even the smallest variants suitable for mobile applications. While fine-tuning a 70-billion-parameter model may still demand substantial computational resources, costs have significantly decreased, allowing tech-enthusiasts like myself to run some of them locally.

Therefore, this blog post is dedicated to people who want to use LLMs locally and usually don’t have powerful GPUs or don’t want to waste too much time in adding optimization techniques. We will cover the basic memory requirements of running an LLM locally, only mentioning the basic optimizations that can be made. Towards the end of this post, you'll find resources to explore further optimization of LLM memory usage. 

**If you are lazy and don't want to read the rest, use this as a general rule of thumb:**

**Inference:** Number of parameters * Precision (usually 2 or 4 Bytes)

**Training:** 4 - 6 times the inference resources

**Also, I started an open-source LLM System Requirements calculator for everyone who wants to contribute:**

**Github:** [https://github.com/manuelescobar-dev/LLM-System-Requirements](https://github.com/manuelescobar-dev/LLM-System-Requirements)

**Website:** [https://llm-system-requirements.streamlit.app](https://llm-system-requirements.streamlit.app/)

***DISCLAIMER:*** This article serves as a guide for estimating memory requirements, but the exact figures depend on various factors related to your setup and frameworks. Therefore, the only real way of estimating the exact memory is trying things out.

# Inference

Performing inference requires resources for loading the model weights and storing the KV cache and activation memory. As we will see later, it doesn’t require gradients and optimizer states as these are only required for training.

**Formula**

$\text{Total Inference Memory} = \text{Model Size} + \text{KV Cache} + \text{Activations}$


Comparison of approximate GPU RAM needed to load versus load and train a 1-billion-parameter model at 32-bit full precision [5].

## Model Weights

The first and most important memory requirement is the memory needed for loading the model. This depends on the number of parameters of the model and the precision you want. A common optimization technique is called quantization. **Quantization consists of loading the model weights with a lower precision.** Although it affects performance, it’s effect is not significant and it is preferred over choosing a smaller model with a higher precision.

**Formula [1]**

$\text{Model Size} = \text{Number of Parameters} * \text{Precision}$

**Precision**

- **4 Bytes:** FP32 / Full-precision / float32 / 32-bit
- **2 Bytes:** FP16 / float16 / bfloat16 / 16-bit
- **1 Byte:** int8 / 8-bit
- **0.5 Bytes:** int4 / 4-bit

**Some additional memory optimizations**

- Double quantization


Approximate GPU RAM needed to load a 1-billion-parameter model at 32-bit, 16-bit, and 8-bit precision [5]

## KV Cache

In transformers, the decoding phase generates a single token at each time step, dependent on previous token tensors. To avoid recomputing these tensors, they are cached in the GPU memory.

**Formula [3]**

$\text{KV Cache} = 2 * \text{Batch Size} * \text{Sequence Length} * \text{Number of Layers} * \text{Hidden Size} * \text{Precision}$

**Some additional memory optimizations:**

- PagedAttention

## Activations

Intermediate activation values must be stored during the forward pass of the model. These activations represent the outputs of each layer in the neural network as data propagates forward through the model. They **must be kept in FP32** to avoid numerical explosion and ensuring convergence.

**Formula [4]**

$\text{Activation Memory} = \text{Batch Size} * \text{Sequence Length} * \text{Hidden Size} * (34+ \frac{5*\text{Sequence Length}*\text{Number of attention heads}}{\text{Hidden Size}})$

**Some additional memory optimizations:**

- PagedAttention
- Sequence-Parallelism [4]
- Activation Recomputation [4]

# Training

Training requires more resources than inference due to the optimizer and gradient states. These are required for training the model and significantly increase the memory resources needed.

**Formula:**

$\text{Total Memory} = \text{Model Size} + \text{KV cache} + \text{Activations} + (\text{Optimizer States} + \text{Gradients})* \text{Number of Trainable Parameters}$


Additional RAM needed to train a model [5].

## Fine-Tuning

Additional memory resources are required for training due to the calculation of optimizer and gradient states. Parameter Efficient Fine-Tuning (PEFT) techniques are often employed, such as Low-rank Adaptation (LoRA) and Quantized Low-rank Adaptation (QLoRA), in order to reduce the number of trainable parameters.

**Techniques** (depends on multiple parameters that will be covered in a future article)

- **LoRA**
- **QLoRA**

## Optimizer States

Optimization algorithms require resources to store the parameters and auxiliary variables. These variables include parameters like the momentum and variance used by optimization algorithms such as Adam (2 states) or SGD (1 state). This depends on the number of optimized states and their precision.

**Formula [1]**

- **AdamW (2 states):** 8 Bytes per parameter
- **AdamW (bitsandbytes Quantized):** 2 Bytes per parameter
- **SGD (1 state):** 4 Bytes per parameter

**Some additional memory optimizations [2]**

- Alternative optimizers (NVIDIA/apex, Adafactor, Quantized Adam, …)
- Paged optimizers

## Gradients

Gradient values are computed during the backward pass of the model. They represent the rate of change of the loss function with respect to each model parameter and are crucial for updating the parameters during optimization. As activations, they must be **stored in FP32** to maintain numerical stability.

**Formula [1]**

4 Bytes per parameter

**Some additional memory optimizations [2]**

- Gradient Accumulation
- Gradient checkpointing

# Conclusion

Calculating the exact memory requirements for running an LLM can be difficult, due to the significant number of frameworks, models, and optimization techniques. Nevertheless, this guide serves as an starting point for estimating the memory resources needed to perform LLM inference and training.

**This is the first article of the LLM series, so stay up to date to the new articles that are coming**

**Hint:** How to deploy a LLM in your own personal computer.

**Thanks for reading and remember to contribute to the LLM System Requirements Calculator!**

# References

[1] [https://huggingface.co/docs/transformers/model_memory_anatomy](https://huggingface.co/docs/transformers/model_memory_anatomy)

[2] [https://huggingface.co/docs/transformers/perf_train_gpu_one](https://huggingface.co/docs/transformers/perf_train_gpu_one)

[3] [https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

[4] [https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/activation_memory_reduction.html](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/activation_memory_reduction.html)

[5] [https://www.oreilly.com/library/view/generative-ai-on/9781098159214/ch04.html](https://www.oreilly.com/library/view/generative-ai-on/9781098159214/ch04.html)
