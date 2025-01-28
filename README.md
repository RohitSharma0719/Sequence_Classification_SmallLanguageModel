# Sequence_Classification_SmallLanguageModel

# Text Classification with Fine-Tuned Transformer Models

This repository provides a complete implementation of a fine-tuned transformer model for multi-class text classification using Hugging Face's Transformers library. It preprocesses the dataset, trains the model, handles imbalanced classes, and saves the trained model for deployment.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Dataset](#dataset)
5. [Usage](#usage)
   - [1. Preprocessing the Dataset](#1-preprocessing-the-dataset)
   - [2. Training the Model](#2-training-the-model)
   - [3. Testing and Submitting Results](#3-testing-and-submitting-results)
6. [File Descriptions](#file-descriptions)
7. [How It Works](#how-it-works)
8. [License](#license)

---

## Introduction

This project implements a pipeline for fine-tuning a transformer-based language model for text classification tasks. The workflow includes:

- Text preprocessing and cleaning.
- Tokenizing and preparing data for training.
- Addressing class imbalance with weighted loss functions.
- Fine-tuning a lightweight transformer model (`distilbert-base-uncased`).
- Saving the model in `.safetensors` format for efficient and secure use.
- Testing the model using Hugging Face pipelines.

---

## Features

- **Text Cleaning:** Automatically cleans text by removing special characters and extra spaces.
- **Custom Tokenization:** Uses Hugging Face's tokenizer to prepare text for the transformer model.
- **Class Weighting:** Handles imbalanced datasets with custom class weights in the loss function.
- **Fast Training:** Implements mixed-precision training (`fp16`) and optimizes batch sizes for quick iterations.
- **Model Deployment:** Trained models are saved in `.safetensors` format for secure deployment.
- **Submission Generation:** Automatically generates a CSV file with predictions for testing purposes.

---

## Requirements

Make sure the following libraries are installed before running the code:

- `transformers`
- `datasets`
- `torch`
- `scikit-learn`
- `pandas`
- `safetensors`

To install the required libraries, run:

```bash
pip install transformers datasets torch scikit-learn pandas safetensors
