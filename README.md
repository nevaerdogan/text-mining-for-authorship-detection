# text-mining-for-authorship-detection
A text mining project to classify authors based on Turkish language documents using TF-IDF, n-gram and BERT embeddings with ensemble models.

***

*Author Classification Using Text Mining and Machine Learning*

This project focuses on classifying the author of a given Turkish text using text mining techniques and machine learning models. The solution is built to demonstrate the use of NLP pipelines and feature extraction methods such as TF-IDF, n-gram, and BERT embeddings.

*Project Objective*

The goal of this project is to predict the author of a text among multiple known writers by analyzing linguistic and stylistic patterns in the content. Authorship detection has use cases in areas such as:

-Content attribution
-Social media fraud detection
-Customer segmentation based on written feedback
-Sentiment and intent analysis

*Techniques & Tools*

Language: Python
Libraries: scikit-learn, pandas, numpy, matplotlib, transformers, xgboost

Models:

-Multilayer Perceptron (MLP)
-Random Forest
-XGBoost

Feature Representations:

-TF-IDF (Unigram, Bigram)
-Character-level n-grams
-BERT embeddings (bert-base-turkish-cased)

*Dataset*

The dataset consists of ~1200 Turkish text samples from multiple authors. Each sample includes a paragraph or passage labeled with its actual author. Texts are preprocessed to clean, normalize, and tokenize content before feature extraction.

*Workflow*

1.Data Cleaning

-Lowercasing, punctuation removal, stopword filtering
-Tokenization using Turkish-specific language rules

2.Feature Extraction

-TF-IDF with different n-gram combinations
-BERT embeddings using HuggingFace transformers

3.Model Training

-Supervised classification using MLP, RF, XGBoost
-Evaluation via Accuracy, Precision, Recall, F1-score

4.Performance Comparison

-Metrics logged for each feature/model combination
-BERT + Ensemble yielded the highest F1-score


*Results and Summary*

Evaluated traditional feature extraction methods including Unigram TF-IDF, word-level 2-3gram TF-IDF, and char-level 2-3gram TF-IDF, in combination with various classifiers such as Random Forest, SVM, Naive Bayes, MLP, Decision Tree, and XGBoost. Additionally,implemented deep contextualized embeddings using BERT, and explored its performance both with a standalone MLP classifier and a fusion of BERT with TF-IDF features.

Experimental results revealed that:
•	Among traditional methods, Unigram TF-IDF + MLP and Char-level TF-IDF + MLP yielded competitive accuracy.
•	BERT-based approaches significantly outperformed classical methods, with BERT + TF-IDF Fusion + MLP achieving the highest accuracy of 0.86, followed by BERT + MLP with 0.83.
•	Character-based n-grams were found to be especially useful in capturing stylistic patterns in Turkish language, performing better than word-level n-grams in several cases.
Overall, the experiments demonstrate the effectiveness of deep learning-based textual representations in authorship attribution tasks, especially when combined with traditional linguistic features like TF-IDF.



*Strategic Impact & Use Cases*

This project goes beyond classification — it illustrates how textual data can provide actionable insights. Some potential real-world applications include:
Customer Profiling: Segment users based on writing style or feedback tone.
Internal HR Analytics: Detect stress or dissatisfaction in employee communication.
Content Moderation: Identify authorship in anonymous or deceptive writing.



Neva Erdogan, 
Computer Engineering Student 
🔗 www.linkedin.com/in/nevaerdogan
