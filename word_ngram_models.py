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

# 2-gram ve 3-gram kelime tabanlı TF-IDF vektörleri çıkar
vectorizer = TfidfVectorizer(ngram_range=(2,3), max_features=5000)
X = vectorizer.fit_transform(texts)

# Veri setini ayır (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelleri tanımla
models = {
    "SVM": SVC(probability=True),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier()
}

# Her modeli eğit ve değerlendir
for name, model in models.items():
    print(f"\n>>> {name} modeli eğitiliyor...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
