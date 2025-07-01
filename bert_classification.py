from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocessing import texts, labels 

# Etiketleri sayısallaştır
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# HuggingFace BERT modeli (Türkçe)
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# BERT'ten vektör çıkaran fonksiyon
def bert_encode(text_list):
    embeddings = []

    for text in text_list:
        # Metni BERT'e uygun hale getir (tokenize et)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Modeli çalıştır (eğitim yok, sadece tahmin)
        with torch.no_grad():
            outputs = model(**inputs)

        # CLS tokenının embedding'ini al (768 boyutlu)
        mean_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(mean_embedding)

    return np.array(embeddings)

# Vektörlere dönüştür
print(" Metinler BERT ile vektörleştiriliyor...")
X = bert_encode(texts)
y = labels_encoded

# Eğitim / Test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Loğistic Regression ile sınıflandırma 
print(" Model eğitiliyor...")
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Tahmin
y_pred = classifier.predict(X_test)

# Sonuçları yazdır
print("Classification Raporu:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
