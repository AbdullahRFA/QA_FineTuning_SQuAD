# Extractive Question Answering with Fine-Tuned BERT on SQuAD v1.1

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: SQuAD v1.1](https://img.shields.io/badge/Dataset-SQuAD%20v1.1-green.svg)](https://rajpurkar.github.io/SQuAD-explorer/)

## Overview

This repository contains an end-to-end implementation of an **Extractive Question Answering (QA)** system built by fine-tuning Google's pre-trained `bert-base-uncased` model on the **Stanford Question Answering Dataset (SQuAD v1.1)**. 

Unlike text classification tasks that map an entire input sequence to a single discrete label, extractive QA requires the model to comprehend a passage of context and identify the exact continuous span of text (start and end token indices) that answers a given natural language question.

---

## Key Objectives

1. **Architectural Understanding:** Contrast sequence-level text classification with token-level span prediction in extractive QA.
2. **Data Engineering:** Tokenize paired `(question, context)` inputs using sliding-window chunking (`max_length = 384`, `doc_stride = 128`) to handle long contexts without truncating answer spans, while accurately mapping character-level answer offsets to subword token positions.
3. **Model Fine-Tuning:** Train `AutoModelForQuestionAnswering` (`bert-base-uncased`) using the Hugging Face `Trainer` API with linear learning rate scheduling and mixed-precision (`fp16`) optimization.
4. **Post-Processing & Evaluation:** Reconstruct text spans from predicted start/end logits and evaluate model performance using standard SQuAD metrics (**Exact Match** and **F1 Score**), alongside custom out-of-sample inference tests.

---

## Text Classification vs. Extractive Question Answering

| Feature | Text Classification | Extractive Question Answering |
| :--- | :--- | :--- |
| **Input Format** | Single text sequence (e.g., review, article) | Paired `(Question, Context)` separated by `[SEP]` |
| **Output Representation** | Single class probability distribution (via `[CLS]` token) | Two token-level probability distributions: `start_logits` and `end_logits` |
| **Model Head** | Linear layer mapping hidden size to `num_labels` | Linear layer mapping hidden size to `2` outputs (start & end positions) |
| **Loss Function** | Single Cross-Entropy Loss over class labels | Average of Start Cross-Entropy Loss and End Cross-Entropy Loss |
| **Evaluation Metrics** | Accuracy, Precision, Recall, Macro/Weighted F1 | Exact Match (EM) and Token-level Overlap F1 Score |

---

## Dataset & Preprocessing Pipeline

* **Dataset:** [SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad)
  * **Training Set:** 87,599 question-answer pairs across 442 Wikipedia articles.
  * **Validation Set:** 10,570 question-answer pairs across 48 Wikipedia articles.
* **Tokenization & Sliding Window Strategy:**
  * **Tokenizer:** `bert-base-uncased` (`AutoTokenizer`)
  * **Max Sequence Length:** `384` tokens
  * **Document Stride:** `128` tokens (`return_overflowing_tokens=True`)
  * **Offset Mapping:** Character start/end indices are aligned with subword token indices (`sequence_ids()`) to assign `start_positions` and `end_positions`. If an answer span falls outside a sliding window chunk, both positions default to the `[CLS]` token index (`0`).

---

## Model Architecture & Training Configuration

* **Base Checkpoint:** `bert-base-uncased` (110M parameters, 12 Transformer layers, 768 hidden dimension, 12 attention heads)
* **Task Head:** `AutoModelForQuestionAnswering`
* **Hardware Environment:** NVIDIA T4 GPU (Google Colab)

### Hyperparameters

| Parameter | Value |
| :--- | :--- |
| Learning Rate | `3e-5` |
| Optimizer | AdamW (`weight_decay = 0.01`) |
| Learning Rate Scheduler | Linear decay with warmup |
| Batch Size (Train / Eval) | `16` / `16` |
| Training Epochs | `3` |
| Mixed Precision | `fp16 = True` |
| Evaluation Strategy | Per epoch (`eval_strategy="epoch"`) |

---

## Results & Evaluation

### 1. SQuAD v1.1 Validation Performance

Post-processing was performed by selecting the top-20 start and end logit candidates (`n_best_size = 20`), filtering invalid spans (e.g., answers outside context tokens or exceeding `max_answer_length = 30`), and scoring via the official `evaluate` SQuAD metric:

| Metric | Score | Benchmark Reference (`bert-base-uncased`) |
| :--- | :--- | :--- |
| **Exact Match (EM)** | **81.49%** | ~80.8% – 81.2% |
| **F1 Score** | **88.66%** | ~88.5% – 88.7% |

### 2. Custom Inference Verification

To verify generalization beyond validation splits, the fine-tuned model was evaluated on unseen context-question pairs:

* **Test Case 1**
  * **Context:** *"Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person and the only woman to win the Nobel Prize twice, and the only person to win the Nobel Prize in two scientific fields."*
  * **Question:** *"Who was the first woman to win a Nobel Prize?"*
  * **Predicted Answer:** `Marie Curie` *(Exact Match)*

* **Test Case 2**
  * **Context:** *"The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889."*
  * **Question:** *"Where is the Eiffel Tower located?"*
  * **Predicted Answer:** `Paris, France` *(Exact Match)*

---

## Repository Structure

```text
├── QA_FineTuning_SQuAD.ipynb   # Main Jupyter/Colab notebook with full pipeline & report
├── fine_tuned_bert_squad/      # Saved model weights, config, and tokenizer artifacts
└── README.md                   # Project documentation
```

---

## Setup & Usage

### 1. Installation

Install the required dependencies in your Python environment or Google Colab runtime:

```bash
pip install transformers datasets evaluate accelerate torch
```

### 2. Running Inference with the Fine-Tuned Model

Once trained and saved locally to `./fine_tuned_bert_squad`, you can run extractive QA inference using the Hugging Face `pipeline` API or direct PyTorch forward passes:

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="./fine_tuned_bert_squad",
    tokenizer="./fine_tuned_bert_squad"
)

context = (
    "The theory of relativity usually encompasses two interrelated physics theories "
    "by Albert Einstein: special relativity and general relativity, proposed and "
    "published in 1905 and 1915, respectively."
)
question = "Who proposed the theory of relativity?"

result = qa_pipeline(question=question, context=context)
print(f"Answer: {result['answer']} (Confidence: {result['score']:.4f})")
```

---

## Key Takeaways & Limitations

* **Subword Alignment & Sliding Windows:** Properly handling `offset_mapping` and `doc_stride` is critical in extractive QA; without overlapping chunks, answers located near the 384-token boundary would be severed and mislabeled.
* **Span-Level Reasoning:** Fine-tuning shifts BERT's attention heads from general masked language modeling toward entity-question matching and boundary detection.
* **Limitations:** Because SQuAD v1.1 assumes every question is answerable from the given context, this model will always extract a text span even when presented with unanswerable questions (unlike models trained on SQuAD v2.0).

---

## References

1. Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. EMNLP 2016.
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019.
3. Hugging Face Course: [Question Answering with Transformers](https://huggingface.co/learn/nlp-course/chapter7/7).
