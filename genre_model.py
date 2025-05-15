import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Učitavanje i čišćenje podataka
df = pd.read_csv("GoodReads_100k_books.csv")
df = df.dropna(subset=["desc", "genre"])
df["main_genre"] = df["genre"].apply(lambda x: x.split(",")[0].strip())

# 2. Filtriraj na top 20 žanrova
top_20 = df["main_genre"].value_counts().head(20).index
df = df[df["main_genre"].isin(top_20)]

# 3. Podjela podataka
X_train, X_test, y_train, y_test = train_test_split(df["desc"], df["main_genre"],
test_size=0.2, stratify=df["main_genre"], random_state=42)

# 4. TF-IDF
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5. Treniranje modela
models = {

"Logistic Regression": LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

# 6. Spremanje najboljeg modela (npr. Logistic Regression) i TF-IDF vektora
joblib.dump(models["Logistic Regression"], "genre_classifier.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")