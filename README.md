<h1 align="center">Tech Tweakers - Polaris LLM </h1>
<p align="center"><i>Learning project to build a Llama LLM model from scratch based on "Llama from Scratch" by Brian Kitano.</i></p>

<div align="center">
  <a href="https://github.com/Tech-Tweakers/polaris-baby/stargazers"><img src="https://img.shields.io/github/stars/Tech-Tweakers/polaris-baby" alt="Stars Badge"/></a>
<a href="https://github.com/Tech-Tweakers/polaris-baby/network/members"><img src="https://img.shields.io/github/forks/Tech-Tweakers/polaris-baby" alt="Forks Badge"/></a>
<a href="https://github.com/Tech-Tweakers/polaris-baby/pulls"><img src="https://img.shields.io/github/issues-pr/Tech-Tweakers/polaris-baby" alt="Pull Requests Badge"/></a>
<a href="https://github.com/Tech-Tweakers/polaris-baby/issues"><img src="https://img.shields.io/github/issues/Tech-Tweakers/polaris-baby" alt="Issues Badge"/></a>
<a href="https://github.com/Tech-Tweakers/polaris-baby/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/Tech-Tweakers/polaris-baby?color=2b9348"></a>
<a href="https://github.com/Tech-Tweakers/polaris-baby/blob/master/LICENSE"><img src="https://img.shields.io/github/license/Tech-Tweakers/polaris-baby?color=2b9348" alt="License Badge"/></a>
</div>

<br>
<p align="center"><i>Got problems or have some time to help? Please open an <a href="https://github.com/Tech-Tweakers/polaris-baby/issues/new">Issue</a> to tell us!</i></p>

# About

Polaris LLM is a custom language model developed using PyTorch. This project encompasses scripts for training the model on a character-level dataset and executing inference with the trained model. The development of Polaris Baby LLM serves educational purposes.

# Latest Inference

| Original Input | Latest Inference |
| -------------- | ---------------- |
| ![Original Input](docs/original-input.png) | ![Latest Inference](docs/latest-inference.png) |

## Overview

Polaris LLM demonstrates the process of building a language model from the ground up, showcasing the intricacies of model architecture and training. It provides hands-on experience with advanced concepts in natural language processing and deep learning.

The training script (train.py) uses PyTorch to train a language model named 'Llama', which incorporates advanced techniques like RoPE (Rotary Positional Embeddings) and SwiGLU activations.

The inference script (inference.py) is designed to load the trained 'Llama' model and generate predictions based on input text.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- NumPy
- Pandas 
- Matplotlib

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/Tech-Tweakers/polaris-baby.git
cd polaris-baby
```

### Training the Model

1 - **Prepare your dataset**: The dataset should be a text file where the text data is used for training the model. The input file should be placed in the root directory. In the **Makefile** there is some functions to improve the dataset, like removing special characters, strings, add tags, etc.

2 - **Run the training script**: The training script will train the model and save the trained model to the models directory. The script will also monitor performance metrics. The script can be run with the following command:

```bash
python train.py
```

### Inference

After training, use the inference.py script for generating text:

1 - Load the trained model: Ensure that the trained model .pth file is accessible to the script.

2 - Run the inference script:

```bash
# Example command
python3 inference.py --maxtokens 5000 --temperature 0.7 --top_k 1 --top_p 10 --user_input

# Help
Polaris LLM Inferencer v0.1.0
usage: inference.py [-h] [--maxtokens MAXTOKENS] [--temperature TEMPERATURE] [--top_k TOP_K] [--top_p TOP_P] [--user_input]

options:
  -h, --help            show this help message and exit
  --maxtokens MAXTOKENS
                        Maximum new tokens to generate
  --temperature TEMPERATURE
                        Sampling temperature
  --top_k TOP_K         Top K filtering
  --top_p TOP_P         Top P (nucleus) filtering
  --user_input          Enable user input for text generation
```

## Dataset and Vocabulary

The dataset file used for training **must be the same file** to be used for the Vocabulary. The Vocabulary is generated from the dataset file and is used to encode the text data into integers. If you train the model on one dataset and then try to use a different Vocabulary for inference, the model will not be able to decode the text data.

## Components

### Training Script (train.py)

- **Character-Level Tokenization**: Splits the text into characters and encodes them as integers.
- **Model Architecture**: Defines the 'Llama' model along with its subcomponents like RoPEMaskedMultiheadAttention and SwiGLU.
- **Training Loop**: Iteratively trains the model, saves checkpoints, and monitors performance metrics.

### Inference Script (inference.py) - Under Development!

- **Model Loading**: Loads the trained 'Llama' model from a **.pth** file.
- **Tokenization and Encoding**: Converts input text into the model-readable format.
- **Text Generation**: Generates output text based on the input provided.

## Customization

You can customize various aspects of the training and inference process:

- **Model Architecture**: Modify the model architecture in **model.py**.

## Contributing

Contributions to Polaris Baby LLM are welcome. Please read CONTRIBUTING.md for guidelines on how to contribute.
