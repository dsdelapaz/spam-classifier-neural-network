# Twitter Spam Detection using CNN-based Text Classification

## Overview

This project implements a **Convolutional Neural Network (CNN)** for spam detection on Twitter data, a **binary text classification problem**. The emphasis is on **NLP preprocessing**, **representation learning**, **class imbalance handling**, and **systematic hyperparameter optimization** rather than application deployment.

The model is trained and evaluated using **PyTorch**, with **Optuna** employed for automated hyperparameter tuning.

---

## Dataset

The dataset consists of **5,572 labeled tweets**, each annotated as either:

* `0` – Non-spam
* `1` – Spam

---

## Preprocessing 

The NLP preprocessing includes:

### Text cleaning

* Lowercasing
* Punctuation removal
* Stopword removal using NLTK

### Tokenization

* Tokenization performed using **spaCy**
* Removal of stopwords and punctuation at the token level

### Vocabulary construction

* Vocabulary built **only on the training set** to avoid data leakage
* Padding token (`<PAD>`) explicitly defined
* Token-to-index mapping created using frequency-based counting

### Sequence preparation

* Variable-length sequences padded using PyTorch utilities
* Padding handled explicitly via `padding_idx` in the embedding layer

---

## EDA

To qualitatively assess textual differences between classes, **word cloud visualizations** were generated for:

* Spam tweets
* Non-spam tweets

This provides insight into token distribution patterns before modeling.

---

## Data Splitting Strategy

A **three-way split** is used to ensure reliable evaluation:

* Training set
* Validation set (for hyperparameter tuning and early stopping)
* Test set (held out entirely until final evaluation)


---

## Model Architecture

### CNN-based Text Classifier

The model architecture consists of:

* **Embedding layer**

  * Learnable word embeddings
  * Padding-aware embedding representation

* **Convolutional Layers**

  * Kernel sizes: 2, 4, and 5
  * Each captures different n-gram patterns

* **Global max pooling**

  * Reduces variable-length sequences to fixed-size representations

* **Fully connected layer**

  * Concatenated convolution outputs
  * Binary classification via softmax

* **Dropout**

  * Applied to mitigate overfitting

This architecture follows established CNN approaches for sentence-level text classification.

---

## Class Imbalance Handling

To address class imbalance between spam and non-spam tweets:

* **Inverse class frequency weighting** is applied
* Weights are incorporated directly into the cross-entropy loss function


---

## Hyperparameter Optimization

Hyperparameter tuning is performed using **Optuna**, optimizing validation loss.

### Tuned parameters

* Learning rate (log-scaled)
* Number of training epochs
* Embedding dimensionality
* Dropout rate

### Optimization techniques

* Early stopping based on validation loss
* Trial-wise reporting and pruning
* Memory cleanup between trials for efficiency

A total of **100 optimization trials** were conducted.

---

## Evaluation Metrics

Final model performance is evaluated on the **held-out test set** using:

* Accuracy
* Precision
* Recall
* F1 Score

### Final Test Results

* **Accuracy**: 0.9641
* **Precision**: 0.8571
* **Recall**: 0.8903
* **F1 Score**: 0.8734

These results demonstrate strong performance, particularly in recall, which is critical for spam detection tasks.

---

## Visualization and Analysis

Optuna visualizations are used to analyze:

* Optimization history
* Hyperparameter importance
* Parallel coordinate plots
* Trial timelines
* Intermediate value convergence


---


## Libraries

* Python
* PyTorch
* spaCy, NLTK
* Optuna
* scikit-learn
* matplotlib, seaborn, plotly

---
