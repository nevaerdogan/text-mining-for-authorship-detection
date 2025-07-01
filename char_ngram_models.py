from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from preprocessing import texts, labels

# Etiketleri sayıya çevir
le = LabelEncoder()
y = le.fit_transform(labels)

# 2-gram ve 3-gram karakter bazlı TF-IDF çıkar
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,3), max_features=3000)
X = vectorizer.fit_transform(texts)

# Eğitim / test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeller
models = {
    "SVM": SVC(probability=True),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier()
}

# Eğit ve yazdır
for name, model in models.items():
    print(f"\n>>> {name} modeli eğitiliyor (char-gram)...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
