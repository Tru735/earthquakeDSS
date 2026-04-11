# EarthquakeDSS Project Context (Updated)

## Project Overview

EarthquakeDSS is a disaster-response decision support system that uses deep learning to classify crisis-related tweets into severity-aware categories so that responders can quickly identify, prioritize, and act on the most urgent information during an earthquake.

## Objectives

- Accurately classify tweets into a small set of predefined severity classes relevant to crisis response.
- Evaluate Bidirectional LSTM (BiLSTM) architectures with domain-specific word embeddings for multi-class text classification.
- Improve generalization and reduce overfitting through data augmentation, regularization, and callback strategies.
- Diagnose and mitigate model weaknesses at the class level using confusion matrices and error analysis.
- Build a reproducible end-to-end pipeline (data loading, preprocessing, modeling, evaluation) suitable for future research and publication.

## Problem Statement

During large-scale crises, platforms like Twitter are flooded with heterogeneous content: noise, reactions, and critical situation updates. Manually scanning these tweets to find life-threatening events, infrastructure damage, or urgent needs is infeasible in real time. The core problem is to develop an automated system that can reliably categorize crisis tweets by severity and information type, enabling responders to filter out noise and focus on the most actionable and high-impact messages.

## Dataset and Labels

### Data Sources

- **Initial compiled dataset**: Loaded from pickled files such as `compiled_data_class.pkl` and later `compileddata_w_severityClass.pkl`, containing `tweet_id`, `tweet_text`, and a fine-grained `label` column derived from CrisisNLP earthquake Twitter data.
- **Cleaned text dataset**: `cleanedtext_data.pkl`, where raw tweets are lowercased, URLs, mentions, "RT" prefixes, and special characters are removed, and hashtags are preserved as plain words.
- **Sequence dataset**: `X_y_sequences.pkl`, containing padded integer sequences (`X`) produced by a Keras `Tokenizer` and the corresponding severity labels (`y`).
- **Embedding matrices**:
  - `embedding_matrix.pkl`: Embedding matrix built from pre-trained CrisisNLP word2vec vectors for the original dataset.
  - `aug_embedding_matrix.pkl`: Embedding matrix aligned with the vocabulary of the augmented dataset.
- **Augmented sequences**: `aug_X_y_sequences.pkl`, storing sequences and labels for the augmented corpus.

### Original Label Space

The original `comp['label']` categories (from CrisisNLP) include:

- `sympathy_and_emotional_support`
- `not_related_or_irrelevant`
- `injured_or_dead_people`
- `infrastructure_and_utilities_damage`
- `caution_and_advice`
- `other_useful_information`
- `donation_needs_or_offers_or_volunteering_services`
- `missing_trapped_or_found_people`
- `displaced_people_and_evacuations`

### Severity Mappings Across Phases

#### Phase 1 – 5-Class Severity Mapping

A first mapping compressed the original categories into 5 ordered classes (0–4):

- **Class 0**: `not_related_or_irrelevant` (noise / non-crisis)
- **Class 1**: `sympathy_and_emotional_support`
- **Class 2**: Mid-severity informational content (e.g., `donation_needs_or_offers_or_volunteering_services`, `caution_and_advice`, `other_useful_information`)
- **Class 3**: High-severity situational updates (`displaced_people_and_evacuations`, `infrastructure_and_utilities_damage`)
- **Class 4**: `injured_or_dead_people` (highest severity)

This mapping created a `severity_class` column with values from 0 to 4.

#### Phase 2 & 3 – 4-Class Severity Mapping

To reduce confusion between similar mid-severity classes, the label space was later redefined into 4 classes using `new_severity_map`:

- **Class 0**: `not_related_or_irrelevant`
- **Class 1**: `sympathy_and_emotional_support`, `other_useful_information`
- **Class 2**: `donation_needs_or_offers_or_volunteering_services`, `caution_and_advice`
- **Class 3**: `injured_or_dead_people`, `missing_trapped_or_found_people`, `displaced_people_and_evacuations`, `infrastructure_and_utilities_damage`

A new DataFrame `new_comp` with this 4-class `severity_class` was used for subsequent model iterations.

---

## Chronological Project Log

### Phase 1 – Initial BiLSTM Model (5 Severity Classes)

**Data Preparation**

- Loaded the compiled dataset into a DataFrame `comp` with `tweet_id`, `tweet_text`, and `label`.
- Cleaned `tweet_id` strings and converted them to integers.
- Applied the first 5-class `severity_map` to derive `severity_class` values in `[0, 4]`.
- Split the sequences and labels into `x_train`, `x_test`, `y_train`, `y_test` using a 75/25 split.

**Model Architecture**

- Used a pre-trained embedding layer initialized from `embedding_matrix.pkl` (CrisisNLP 300‑dimensional embeddings) with weights frozen.
- Stacked two Bidirectional LSTM layers, followed by dropout and dense layers.
- Final layer: Dense with softmax activation over 5 classes.
- Loss: `sparse_categorical_crossentropy`, Optimizer: `adam`.
- Training callbacks: `ModelCheckpoint` to save best weights by validation loss.

**Training & Outcome**

- Trained for up to 50 epochs.
- Achieved overall accuracy of ~**0.78** with a weighted F1-score around **0.78** on the 5-class test set.

**Problems Identified**

- Confusion matrix and classification report showed that some classes were much weaker than others.
- The model particularly struggled to distinguish between two mid-severity classes (e.g., classes 2 and 3 in the 5-class mapping), with lower precision (~0.54) and recall (~0.60) for one class.
- Error analysis indicated that fine-grained boundaries between mid-severity categories were too subtle for the available data, leading to unstable performance on those classes.

---

### Phase 2 – Refined BiLSTM Model (4 Severity Classes)

**Motivation**

- Reduce label ambiguity and merge semantically similar categories to simplify the task.
- Improve performance on poorly performing classes by pooling their examples into broader severity bins.

**Label Remapping**

- Defined `new_severity_map` with 4 classes (0–3) as described above.
- Constructed a new DataFrame `new_comp` with this mapping and re-generated the sequences and labels (`new_x_train`, `new_x_test`, `new_y_train`, `new_y_test`).

**Model 2 Configuration**

- Reused the same overall BiLSTM architecture and CrisisNLP embedding matrix.
- Adjusted the output layer to have 4 units (softmax over 4 classes).
- Kept similar training setup: `adam` optimizer, `sparse_categorical_crossentropy`, `ModelCheckpoint` for best weights.

**Training & Outcome**

- Trained for up to 50 epochs.
- Achieved accuracy of about **0.766** with weighted F1-score ~**0.77** on the 4-class test set.

**Remaining Weaknesses (from error analysis)**

- **Class 2 (donation needs/offers, caution/advice)**:
  - Precision ≈ 0.45, Recall ≈ 0.55, F1 ≈ 0.50.
  - Many false positives: tweets predicted as Class 2 that actually belonged to Class 1.
- **Class 0 (not related or irrelevant)**:
  - Precision ≈ 0.73, Recall ≈ 0.77, F1 ≈ 0.75.
  - Confusion with Class 1, reflecting difficulty separating mild reactions / commentary from truly irrelevant content.
- Overall, there was substantial confusion between **Class 0 and Class 1**, and between **Class 1 and Class 2**, indicating that the model still lacked enough variability and examples to learn these nuanced distinctions.

---

### Phase 3 – BiLSTM with Augmented Dataset (4 Severity Classes)

**Motivation**

- Address class imbalance and lack of linguistic diversity, especially for informative and high-severity classes.
- Improve generalization to new crisis tweets and strengthen performance for minority classes.

**Data Augmentation Strategy**

- Focused on text augmentation for underrepresented classes while preserving semantic meaning.
- Implemented **random insertion** augmentation:
  - For selected tweets (primarily in minority classes), inserted additional words (e.g., synonyms or related tokens) at random positions.
  - Kept the overall meaning and severity label unchanged.
- Constructed an augmented DataFrame `df_train_aug` by:
  - Starting from the original labeled tweets.
  - Generating multiple augmented versions for tweets in minority classes.
  - Preserving the original `tweet_id` for all its augmented variants to maintain lineage, while modifying `tweet_text` and keeping `severity_class` fixed.

**Diagnostics for Augmented Data**

- Verified that `df_train_aug` had:
  - **3486 total rows**.
  - **1545 unique `tweet_id`s**.
  - **1941 duplicated `tweet_id` entries**, indicating that many tweets had 3 additional augmented versions.
- Top `tweet_id`s appeared 4 times each: 1 original + 3 augmented texts.
- Inspected example groups of rows sharing the same `tweet_id` to confirm that the texts differed while labels and IDs were consistent.

**Augmented Sequences and Embeddings**

- Re-applied tokenization and padding to `df_train_aug.clean_text` to create `aug_X` and `aug_y`, stored in `aug_X_y_sequences.pkl`.
- Built `aug_embedding_matrix.pkl` aligned with the augmented vocabulary using CrisisNLP embeddings wherever available.

**Model 3 Configuration**

- Architecturally similar BiLSTM model (`model3`) with:
  - Pre-trained CrisisNLP-based embedding layer initialized from `aug_emb`.
  - Bidirectional LSTMs, dropout, and dense layers.
  - Softmax output over 4 classes.
- Training–validation split: `aug_x_train`, `aug_x_test`, `aug_y_train`, `aug_y_test` with test size 0.2.
- Callbacks:
  - `ModelCheckpoint` (`mcp3`) to save best weights by validation loss.
  - `EarlyStopping` (`es3`) on validation loss with patience, to stop before overfitting.

**Problems Encountered and Fixes**

- **Problem 1 – API / NameError**:
  - Initially used `accuracy(...)` instead of `accuracy_score(...)`, causing `NameError: name 'accuracy' is not defined`.
  - Fixed by importing and using `accuracy_score` from `sklearn.metrics`.

- **Problem 2 – Overfitting due to validation data mismatch**:
  - Early training of `model3` showed extremely high training accuracy (near 100%) but poor validation accuracy (30–40%) and rising validation loss.
  - Root cause: `validation_data` was incorrectly set to `(new_x_test, new_y_test)` from the _original_ dataset, while `aug_x_train`/`aug_y_train` came from the _augmented_ dataset.
  - Fix: changed `model3.fit` to use `validation_split=0.2` on the augmented training data itself, ensuring consistent distribution between train and validation.
  - With `EarlyStopping` and `ModelCheckpoint`, the model now stopped at the optimal epoch and retained the best weights.

- **Problem 3 – Python built-ins shadowed**:
  - During embedding-matrix utilities, lines like `vocab_size = min((max_vocab, len(word_index)) + 1)` and assignments such as `min = ...` or `len = ...` caused `TypeError` and `'int' object is not callable` errors.
  - Fixes included:
    - Correct computation: `vocab_size = min(max_vocab, len(word_index)) + 1`.
    - Deleting mistaken variables (`del min`, `del len`) and renaming to safe identifiers (`min_len`, `num_tokens`).

**Final Training Behaviour**

- With the corrected setup, training curves showed:
  - Training loss decreasing steadily.
  - Validation loss decreasing and then stabilizing without blowing up.
  - Training accuracy high, validation accuracy tracking it more closely (no extreme divergence).

**Final Results (Augmented Model)**

- **Accuracy**: ~**0.821** on `aug_x_test`.
- **Weighted ROC-AUC**: ~**0.959**, indicating excellent discrimination among classes.

**Per-Class Performance (from classification report)**

- **Class 0 – Not related or irrelevant**:
  - F1 ≈ 0.91, recall ≈ 0.99.
  - Model almost perfectly identifies irrelevant tweets, with very few false negatives.

- **Class 3 – High severity (injured/dead, missing/trapped, displaced/evacuated, infrastructure damage)**:
  - F1 ≈ 0.81, precision ≈ 0.85, recall ≈ 0.76.
  - Strong performance on the most critical class, meaning high reliability for urgent situation reports.

- **Class 2 – Donation needs/offers, caution/advice**:
  - F1 ≈ 0.80, recall ≈ 0.90, precision ≈ 0.72.
  - Model captures most true Class 2 tweets but sometimes over-predicts this class, leading to false positives from neighboring categories.

- **Class 1 – Sympathy / Other useful information**:
  - F1 ≈ 0.73, precision ≈ 0.90, recall ≈ 0.61.
  - When the model predicts Class 1 it is usually correct, but it misses a non-trivial portion of genuine Class 1 tweets (under-prediction / lower recall).

**Impact of Data Augmentation and Callbacks**

- Data augmentation substantially increased linguistic diversity for minority classes, reducing overfitting and improving recall for informative and high-severity categories.
- `ModelCheckpoint` and `EarlyStopping` ensured that the best validation-performance weights were saved, even though training itself might continue for more epochs in some runs.
- Compared to earlier phases, augmentation + callbacks delivered higher accuracy, much better ROC-AUC, and more balanced per-class performance.

---

## Git, Repository, and Project Structure

- Project is version-controlled in the GitHub repository `Tru735/earthquakeDSS`.
- After deleting the original local directory and recloning, several Git push issues were encountered:
  - `! [rejected] main -> main (fetch first)` when the remote contained commits not present locally.
  - Resolved via `git pull origin main` or, when intentionally overwriting remote history with the local state, via `git push origin main --force` (used carefully).
  - Occasional `stale info` or `origin/main` ambiguity errors with `--force-with-lease`, addressed by cleaning untracked files (`git clean -fd`), resetting (`git reset --hard HEAD`), and pushing again.
- Repository structure (high level):
  - `data/` and `data_extracted/`: raw and processed data artifacts (often `.pkl`, `.bin`, `.csv`).
  - `models/severity_classification/`: saved model weights (`best_model.weights_bilstm*.weights.h5`).
  - `notebook/`: Jupyter notebooks documenting experiments.
  - `papers/`: space for draft reports and future research paper material.
  - Root-level `README.md`, `.gitignore`, `requirements.txt`.
- `.gitignore` configured to ignore heavy and intermediate data artifacts (for example, `data/*.pkl`) while still tracking code, notebooks, configuration, and select light-weight artifacts needed for reproducibility.

---

## Lessons Learned and Future Directions

- **Label design matters**: Coarse-to-fine experimentation with label mappings (5 classes → 4 classes) helped find a balance between expressiveness and learnability.
- **Domain-specific embeddings are powerful**: CrisisNLP embeddings significantly boosted performance compared to what generic embeddings would likely achieve in this specialized domain.
- **Data augmentation is crucial for imbalanced crisis data**: Random insertion augmentation, when carefully constrained, improved recall on minority classes without sacrificing overall accuracy.
- **Careful debugging saves time**: Avoiding variable names like `len` and `min`, and using explicit diagnostics for duplicates and class distributions, prevented subtle bugs from derailing experiments.
- **Callbacks and validation design are as important as architecture**: Correctly aligning training and validation datasets, and using `EarlyStopping` and `ModelCheckpoint`, was key to obtaining a model that generalizes instead of memorizing.
- **Version control discipline**: Cleaning up Git history and setting a robust `.gitignore` ensured that only meaningful code and configuration are pushed, making the project easier to share and extend.

This markdown file is intended as a comprehensive context artifact for future LLM-based assistance for writing a full paper on the EarthquakeDSS project.
