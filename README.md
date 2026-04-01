# Disaster Severity Classification System (BiLSTM vs Transformer)

## 📌 Overview

This project focuses on building an **Intelligent Disaster Severity Classification System** that analyzes social media text (tweets) during crisis events and categorizes them into actionable severity levels.

The system is designed to assist in **real-time disaster response and decision-making**, where rapid identification of high-severity situations (e.g., casualties, infrastructure collapse) is critical.

---

## 🎯 Objective

The primary goals of this project are:

* Classify disaster-related tweets into **severity levels**
* Compare **classical deep learning models** vs **modern transformer models**
* Analyze trade-offs in:

  * performance
  * efficiency
  * robustness
* Build a system that can be extended for **real-time crisis monitoring**

---

## 🧠 Problem Statement

During disasters, large volumes of unstructured data (tweets, messages) are generated. Extracting actionable insights from this data is challenging.

We aim to answer:

> *Can machine learning models automatically detect the severity of crisis situations from text?*

---

## 🗂️ Dataset

We use a crisis-related tweet dataset where each tweet is annotated into categories such as:

* Injured or dead people
* Infrastructure damage
* Missing or trapped people
* Donations and aid
* Advice and warnings
* Irrelevant information

---

## 🔄 Label Engineering

Since the dataset does not directly provide severity levels, we transform the categories into **severity classes**:

| Original Category     | Severity |
| --------------------- | -------- |
| Injured / Dead        | Critical |
| Missing / Trapped     | Critical |
| Infrastructure Damage | High     |
| Displaced People      | High     |
| Donations / Aid       | Medium   |
| Advice / Info         | Low      |
| Irrelevant            | Zero     |

Final labels:

```
0 → Irrelevant  
1 → Low  
2 → Medium  
3 → High  
4 → Critical  
```

---

## 🏗️ Model Architectures

### 🔵 Model 1: BiLSTM with Pretrained Embeddings

Pipeline:

```
Tweet → Word Tokenizer → Embedding Matrix → BiLSTM → Dense Layers → Output
```

* Uses pretrained word embeddings
* Captures sequential dependencies
* Represents classical NLP approach

---

### 🔴 Model 2: Transformer (DistilBERT)

Pipeline:

```
Tweet → Subword Tokenizer → Transformer → Classification Head → Output
```

* Uses contextual embeddings
* Handles noisy and unseen text better
* Represents modern NLP approach

---

## ⚙️ Implementation Details

### Preprocessing

* Text cleaning
* Tokenization
* Padding sequences

### Embeddings (BiLSTM model)

* Pretrained embedding vectors used
* Embedding matrix constructed based on dataset vocabulary

### Training

* Loss: Categorical Crossentropy
* Optimizer: Adam
* Evaluation using validation split

---

## 📊 Evaluation Metrics

We evaluate using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix**

### Key Definitions

* **Precision** → How accurate predictions are
* **Recall** → How many real cases are detected
* **F1-score** → Balance between precision and recall
* **Support** → Number of samples per class

---

## 📈 Results (BiLSTM Baseline)

* Accuracy: ~84%
* Strong performance on:

  * Irrelevant
  * Low severity
* Weak performance on:

  * High and Critical severity

### Observations

* Model shows **high recall but low precision for critical cases**
* Indicates tendency to **overpredict high severity**
* Performance impacted by:

  * class imbalance
  * lack of contextual understanding

---

## 🔍 Key Insights

* Classical models struggle with **semantic nuance**
* Static embeddings cannot capture context effectively
* High-severity detection remains challenging

---

## 🚀 Transformer Advantage

Expected improvements with transformer model:

* Better understanding of context
* Improved performance on rare classes
* Reduced false positives in critical cases

---

## ⚖️ Model Comparison (Planned)

| Metric                | BiLSTM   | Transformer |
| --------------------- | -------- | ----------- |
| Accuracy              | Moderate | Higher      |
| Context Understanding | Limited  | Strong      |
| Handling Noisy Text   | Weak     | Strong      |
| Training Cost         | Low      | Higher      |

---

## 🧪 Future Work

* Train and evaluate transformer model
* Perform hyperparameter tuning
* Handle class imbalance more effectively
* Deploy as a real-time API system
* Integrate live data streams

---

## 💡 Applications

* Disaster response systems
* Emergency management platforms
* Social media monitoring tools
* Government and NGO decision support

---

## 🧭 Conclusion

This project demonstrates how machine learning can be used to convert unstructured crisis data into structured, actionable insights.

The comparison between BiLSTM and transformer models highlights the evolution of NLP techniques and their impact on real-world applications.

---

## 📎 How to Run

1. Activate environment:

```
source ml/bin/activate
```

2. Run training:

```
python train_bilstm.py
```

3. (Optional) Run transformer model:

```
python train_bert.py
```

---

## 📌 Author

[Your Name]

---

## ⭐ Key Highlight

> Built an end-to-end disaster intelligence system comparing classical and modern NLP approaches, achieving strong baseline performance and deriving actionable insights for real-world deployment.

---
