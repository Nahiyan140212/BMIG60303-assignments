# Assignment 06 - CNN for Drug Mention Detection
**BMIG60030 - Fall 2025**
**Dataset:** MIMIC-Ext-DrugDetection (Binary Classification)
**Dataset Link:** https://drive.google.com/drive/folders/1WqsR69QCKSfEKGjboBIxQ7YfvtsfDlZR?usp=sharing

---

## Introduction

This assignment implements a Convolutional Neural Network (CNN) for binary classification of drug mentions in clinical text from the MIMIC-Ext-DrugDetection dataset. The dataset contains 804 training samples and 806 validation samples, with balanced classes (50% drug mentions, 50% no drug mentions). The goal was to optimize the CNN architecture by experimenting with different hyperparameters including maximum sequence length (maxlen), training epochs, and word embeddings. I have considered training and validation set only. For the final run I tested few of the unlabeled text from the test.csv

**Hardware:** Google Cloud A100 GPU with mixed precision training (float16/float32) and XLA compilation enabled for optimal performance.

---

## Part 1: Maximum Sequence Length (maxlen) Experiments


### Results

| maxlen | Validation Accuracy | Coverage | Training Time |
|--------|-------------------|----------|---------------|
| 30     | 0.9082 (90.82%)   | 81.7%    | 18.02s        |
| **50** | **0.9156 (91.56%)** | **94.7%** | **7.30s** |
| 75     | 0.9132 (91.32%)   | 98.1%    | 7.60s         |

### Selection Rationale
**I chose maxlen=50** as the optimal value because:
1. Achieved the highest validation accuracy (91.56%)
2. Captures 94.7% of all sentences without truncation
3. Balances context capture with minimal zero-padding overhead
4. Most efficient training time after initial GPU warmup

While maxlen=75 captures slightly more sentences (98.1%), the additional padding for shorter sentences likely introduced noise, slightly degrading performance. Maxlen=30 truncated too many longer clinical sentences, losing important contextual information about drug mentions.

---

## Part 2: Training Epochs Experiments

### Methodology
Using the optimal maxlen=50, I tested four different epoch values to find the balance between model learning and overfitting. The model includes EarlyStopping and ReduceLROnPlateau callbacks to prevent overfitting and adapt the learning rate dynamically.

### Results

| Epochs | Validation Accuracy | Training Time | Notes |
|--------|-------------------|---------------|-------|
| 3      | 0.9020 (90.20%)   | 6.66s         | Stopped early at epoch 2 |
| 5      | 0.9169 (91.69%)   | 6.92s         | Completed all epochs |
| 8      | 0.9218 (92.18%)   | 7.35s         | LR reduced at epoch 6 |
| **10** | **0.9268 (92.68%)** | **7.61s** | **LR reduced at epoch 6 & 9** |

### Selection Rationale
**I chose epochs=10** as the optimal value because:
1. Achieved the best validation accuracy (92.68%)
2. The model benefited from extended training with adaptive learning rate reduction
3. Training progressed steadily across all 10 epochs without significant overfitting
4. EarlyStopping did not trigger, indicating the model continued to improve

The learning rate reduction at epochs 6 and 9 allowed the model to fine-tune its weights more carefully, leading to better generalization. While 8 epochs also performed well (92.18%), the additional 2 epochs provided a meaningful 0.5% accuracy improvement.

---

## Part 4: Word Embedding Comparison

### Methodology
I compared two embedding approaches:
1. **Custom MIMIC Word2Vec:** Domain-specific embeddings trained on the MIMIC clinical text using Skip-gram (300 dimensions, vocabulary size: 2,110 words)
2. **GloVe (glove.6B.300d):** General-purpose pre-trained embeddings from Wikipedia and Gigaword (300 dimensions, vocabulary size: 400,000 words)

Both models used the optimal configuration: maxlen=50, epochs=10, batch_size=128.

### Results

| Embedding Type | Validation Accuracy | Training Behavior | Vocabulary Coverage |
|----------------|-------------------|-------------------|---------------------|
| **Custom MIMIC Word2Vec** | **0.9268 (92.68%)** | Steady convergence | 2,110 clinical terms |
| GloVe | 0.9243 (92.43%) | Faster initial learning, signs of overfitting | 400,000 general terms |

**Difference:** 0.25% improvement with domain-specific embeddings

### Selection Rationale
**I chose Custom MIMIC Word2Vec** as the optimal embedding because:
1. Slightly higher validation accuracy (0.25% improvement)
2. Trained specifically on clinical terminology relevant to drug mentions (e.g., "heroin," "cocaine," "ivdu," "polysubstance," "methadone")
3. More stable training progression without overfitting
4. Better semantic representations for medical abbreviations and clinical context

While GloVe performed competitively (92.43%), it showed signs of overfitting in later epochs (training accuracy approached 100% while validation plateaued). The domain-specific MIMIC embeddings captured clinical nuances that general-purpose embeddings missed, such as medical abbreviations (IVDU = intravenous drug use) and context-specific drug terminology.

---

## Final Optimized Configuration

**Best Hyperparameters:**
- **maxlen:** 50
- **epochs:** 10
- **embeddings:** Custom MIMIC Word2Vec
- **batch_size:** 128 (GPU-optimized)
- **architecture:** Conv1D (250 filters, kernel size 3) → GlobalMaxPooling1D → Dense(250) → Dropout(0.2) → Dense(1, sigmoid)

**Final Performance:**
- **Validation Accuracy:** 92.68%
- **Total Parameters:** 288,251
- **Training Time:** ~8 seconds per configuration

---

### Best Performing Configuration
The **optimal configuration** (maxlen=50, epochs=10, Custom MIMIC Word2Vec) achieved **92.68% validation accuracy**. This configuration excelled because:
- It captured 94.7% of sentence content without excessive padding
- Extended training with an adaptive learning rate allowed thorough model optimization
- Domain-specific embeddings encoded clinical terminology critical for drug detection

The model successfully learned to distinguish drug-related clinical text from general medical notes, as evidenced by sample predictions correctly identifying terms like "cocaine," "heroin," "IVDU," and "opiate abuse."

### Worst Performing Configuration
The **weakest configuration** (maxlen=30, epochs=3, not fully optimized) achieved **90.20% accuracy**. This underperformed because:
- Maxlen=30 truncated 18.3% of sentences, losing contextual information about drug mentions in longer clinical narratives
- Only 3 epochs provided insufficient training time for the model to learn complex patterns fully
- Early stopping at epoch 2 indicated the model needed more training

1. **Domain-specific embeddings matter:** Even with a small vocabulary (2,110 vs. 400,000 words), domain-specific Word2Vec slightly outperformed GloVe by capturing clinical terminology nuances.

2. **Sweet spot for maxlen:** The 95th percentile sentence length was optimal—not too restrictive (maxlen=30) nor too padded (maxlen=75).

3. **GPU acceleration:** A100 GPU with mixed precision training enabled rapid experimentation, completing all experiments in under 2 minutes total.


