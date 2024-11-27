<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <a href="https://github.com/manuelescobar-dev/LLM-Tools.git">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">LLM Tools</h3>
  <p>Useful tools for LLM development!</p>

  <p>
    Available on <a href="https://llm-system-requirements.streamlit.app"><img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit"></a>
  </p>
  <p><a href="https://llm-system-requirements.streamlit.app">
    https://llm-system-requirements.streamlit.app</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#memory-requirements-calculator">Memory Requirements Calculator</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#roadmap">Getting Started</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

The LLM Tools is an open-source project designed to provide essential tools for developing and running large language models (LLMs).

* **Memory Requirements Calculator:** Estimates the memory needed to run or train LLMs based on factors like model size, precision, batch size, and sequence length.

**Formerly Known As:** LLM System Requirements Calculator

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MEMORY REQUIREMENTS CALCULATOR -->
## Memory Requirements Calculator

For an in-depth guide, see: [Memory Requirements for LLM Training and Inference](https://medium.com/@manuelescobar-dev/memory-requirements-for-llm-training-and-inference-97e4ab08091b).


The Memory Requirements Calculator helps estimate the memory needed to run or train large language models (LLMs). Calculating the exact memory requirements for running an LLM can be challenging due to the significant number of frameworks, models, and optimization techniques. This tool provides essential formulas and considerations to offer accurate and practical estimates for various use cases.

**If you are lazy and don't want to use the calculator, use this as a general rule of thumb:**

- **Inference:** Number of parameters × Precision (usually 2 or 4 Bytes)
- **Training:** 4–6 times the inference resources

### Inference

Performing inference requires resources for loading the model weights, storing the KV cache, and for the activation memory.

**Formula**  

$$\text{Total Inference Memory} = \text{Model Size} + \text{KV Cache} + \text{Activations}$$

#### Model Weights

The memory required for loading the model depends on the number of parameters and precision.

**Formula [[1]](https://huggingface.co/docs/transformers/model_memory_anatomy)**  

$$\text{Model Size} = \text{Number of Parameters} \times \text{Precision}$$

**Precision Values**  
- **4 Bytes:** FP32 / Full-precision / float32 / 32-bit
- **2 Bytes:** FP16 / bfloat16 / 16-bit
- **1 Byte:** int8 / 8-bit
- **0.5 Bytes:** int4 / 4-bit

#### KV Cache

The decoding phase generates a single token at each time step, dependent on previous token tensors. To avoid recomputing these tensors, they are cached in the GPU memory.

**Formula [[3]](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)**  

$$\text{KV Cache} = 2 \times \text{Batch Size} \times \text{Sequence Length} \times \text{Number of Layers} \times \text{Hidden Size} \times \text{Precision}$$

#### Activations

Intermediate activation values must be stored during the forward pass of the model. These activations represent the outputs of each layer in the neural network as data propagates forward through the model. They **must be kept in FP32** to avoid numerical instability and ensure convergence.

**Formula [[4]](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/activation_memory_reduction.html)**  

$$\text{Activation Memory} = \text{Batch Size} \times \text{Sequence Length} \times \text{Hidden Size} \times \left(34 + \frac{5 \times \text{Sequence Length} \times \text{Number of attention heads}}{\text{Hidden Size}}\right)$$

### Training

Training and fine-tuning require more resources than inference due to the optimizer and gradient states. For fine-tuning, Parameter Efficient Fine-Tuning (PEFT) techniques, such as Low-rank Adaptation (LoRA) and Quantized Low-rank Adaptation (QLoRA), are often employed to reduce the number of trainable parameters.

**Formula**  

$$\text{Total Memory} = \text{Model Size} + \text{KV Cache} + \text{Activations} + (\text{Optimizer States} + \text{Gradients}) \times \text{Number of Trainable Parameters}$$

#### Optimizer States

Optimization algorithms require resources to store the parameters and auxiliary variables. These variables include momentum and variance used by algorithms such as Adam (2 states) or SGD (1 state). The precision and type of optimizer affect memory usage.

**Formula [[1]](https://huggingface.co/docs/transformers/model_memory_anatomy)**

- **AdamW (2 states):** 8 Bytes per parameter
- **AdamW (bitsandbytes Quantized):** 2 Bytes per parameter
- **SGD (1 state):** 4 Bytes per parameter

#### Gradients

Gradient values are computed during the backward pass of the model. They represent the rate of change of the loss function with respect to each model parameter and are crucial for updating the parameters during optimization. As with activations, they **must be stored in FP32** for numerical stability.

**Formula [[1]](https://huggingface.co/docs/transformers/model_memory_anatomy)**  

4 Bytes per parameter

### References  

[1] [https://huggingface.co/docs/transformers/model_memory_anatomy](https://huggingface.co/docs/transformers/model_memory_anatomy)

[2] [https://huggingface.co/docs/transformers/perf_train_gpu_one](https://huggingface.co/docs/transformers/perf_train_gpu_one)

[3] [https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

[4] [https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/activation_memory_reduction.html](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/activation_memory_reduction.html)

[5] [https://www.oreilly.com/library/view/generative-ai-on/9781098159214/ch04.html](https://www.oreilly.com/library/view/generative-ai-on/9781098159214/ch04.html)

### Disclaimer
The Memory Requirements Calculator is a tool for estimating memory requirements for LLMs and the only real way of estimating the exact memory is trying things out. The tool is provided as-is, without any warranties or guarantees.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Memory Requirements Calculator
- [ ] Cost Estimation Calculator

See the [open issues](https://github.com/manuelescobar-dev/LLM-Tools/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To set up and run the project locally, follow these steps:

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/manuelescobar-dev/LLM-Tools.git
   ```
2. Navigate to the project directory:
   ```sh
   cd LLM-Tools
   ```
3. Install the dependencies using Poetry:
   ```sh
   poetry install
   ```

### Running the Project

1. Start the Streamlit app:
   ```sh
   streamlit run app.py
   ```
2. Open the provided URL in your browser to view the application.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Top contributors:

<a href="https://github.com/manuelescobar-dev/LLM-Tools/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=manuelescobar-dev/LLM-Tools" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

**Github:** [@manuelescobar-dev](https://github.com/manuelescobar-dev)

**Medium:** [@manuelescobar-dev](https://medium.com/@manuelescobar-dev)

**Email:** manuelescobar.dev@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list any other cool resources for LLM Developers.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/manuelescobar-dev/LLM-Tools.svg?style=for-the-badge
[contributors-url]: https://github.com/manuelescobar-dev/LLM-Tools/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/manuelescobar-dev/LLM-Tools.svg?style=for-the-badge
[forks-url]: https://github.com/manuelescobar-dev/LLM-Tools/network/members
[stars-shield]: https://img.shields.io/github/stars/manuelescobar-dev/LLM-Tools.svg?style=for-the-badge
[stars-url]: https://github.com/manuelescobar-dev/LLM-Tools/stargazers
[issues-shield]: https://img.shields.io/github/issues/manuelescobar-dev/LLM-Tools.svg?style=for-the-badge
[issues-url]: https://github.com/manuelescobar-dev/LLM-Tools/issues
[license-shield]: https://img.shields.io/github/license/manuelescobar-dev/LLM-Tools.svg?style=for-the-badge
[license-url]: https://github.com/manuelescobar-dev/LLM-Tools/blob/main/LICENSE
[streamlit-shield]: https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white
[streamlit-url]: https://streamlit.io
[app-url]: https://llm-system-requirements.streamlit.app