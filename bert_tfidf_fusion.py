from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from preprocessing import texts, labels

le = LabelEncoder()
y = le.fit_transform(labels)

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
def bert_encode(text_list):
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        mean_pool = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(mean_pool)
    return np.array(embeddings)

print("BERT vektorleri cikariliyor...")
X_bert = bert_encode(texts)


print(" TF-IDF vektorleri cikariliyor...")
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # isteğe bağlı artar
X_tfidf = tfidf_vectorizer.fit_transform(texts).toarray()


print(" BERT + TF-IDF birlestiriliyor...")
X_combined = np.concatenate([X_bert, X_tfidf], axis=1)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


print(" MLP modeli egitiliyor...")
classifier = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=500, alpha=0.001, random_state=42, verbose=True)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(" Results:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
