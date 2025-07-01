import os
import nltk
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')  # For tokenization

stop_words = set(get_stop_words('turkish')) #turkish stopword list

#simplified cleaning function
def clean_text(text):
    text = text.lower()  # Küçük harfe çevir
    text = ''.join(char for char in text if char.isalpha() or char.isspace())  # Sadece harfleri ve boşlukları al
    tokens = text.split()  # Kelimeleri boşlukla ayır (nltk.word_tokenize yerine)
    tokens = [word for word in tokens if word not in stop_words]  # Stopword'leri çıkar
    return ' '.join(tokens)


DATASET_DIR = "dataset_authorship"

texts = []
labels = []

for author_folder in os.listdir(DATASET_DIR):
    author_path = os.path.join(DATASET_DIR, author_folder)
    
    if os.path.isdir(author_path):
        for filename in os.listdir(author_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(author_path, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    raw_text = file.read()
                    cleaned_text = clean_text(raw_text)
                    texts.append(cleaned_text)
                    labels.append(author_folder)

print("Total num of authors:", len(set(labels)))
print("Total num of files:", len(texts))


from sklearn.model_selection import train_test_split

# splitting dataset into 80%-20% train-test 
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

print("\nNum of Training data:", len(X_train))
print("Num of Test data:", len(X_test))


from sklearn.preprocessing import LabelEncoder

# Etiketleri sayıya çevir
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)



# TEMEL TF-IDF  - Basit kelime bazlı TF-IDF (unigram)
vectorizer_unigram = TfidfVectorizer()
X_train_uni = vectorizer_unigram.fit_transform(X_train)
X_test_uni = vectorizer_unigram.transform(X_test)

print("Unigram TF-IDF shape (Train):", X_train_uni.shape)
print("Unigram TF-IDF shape (Test):", X_test_uni.shape)

# Word-level N-Gram TF-IDF (2-gram, 3-gram) 2 ve 3 kelimelik grupları (word n-gram)
vectorizer_word_ngram = TfidfVectorizer(ngram_range=(2, 3))
X_train_word_ngram = vectorizer_word_ngram.fit_transform(X_train)
X_test_word_ngram = vectorizer_word_ngram.transform(X_test)

print("Word 2-3gram TF-IDF shape (Train):", X_train_word_ngram.shape)


#  Character-level N-Gram TF-IDF (2-char, 3-char) Karakter bazlı n-gram (örn: 'ba', 'as', 'la')
vectorizer_char_ngram = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
X_train_char_ngram = vectorizer_char_ngram.fit_transform(X_train)
X_test_char_ngram = vectorizer_char_ngram.transform(X_test)

print("Char 2-3gram TF-IDF shape (Train):", X_train_char_ngram.shape)


#  ******model_classification.py*******


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# ✅ Unigram TF-IDF'yi kullanıyoruz (önceden preprocessing.py'de hazırlanmış)
# X_train_uni, X_test_uni, y_train, y_test hazır olmalı

# MODELLERİ BURAYA LİSTELEYELİM
models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Naive Bayes": MultinomialNB(),
    "MLP": MLPClassifier(max_iter=300),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # uyarı susturucu
}

# 🔁 Her modeli döndürüp eğitiyoruz
for name, model in models.items():
    print(f"\n>>> {name} modeli egitiliyor...")
    model.fit(X_train_uni, y_train)  # Eğit
    y_pred = model.predict(X_test_uni)  # Test
    
    # Sonuçları değerlendir
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f" Accuracy:  {acc:.4f}")
    print(f" Precision: {prec:.4f}")
    print(f" Recall:    {rec:.4f}")
    print(f" F1-score:  {f1:.4f}")

    # Sayıdan yazara dönüştürmek için:
le.inverse_transform(y_pred)
