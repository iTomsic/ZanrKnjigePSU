import streamlit as st
import joblib

# Učitavanje modela i vektora
model = joblib.load("genre_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("Predikcija žanra knjige")
st.write("Unesi opis knjige da bismo pokušali predvidjeti žanr.")

# Unos opisa
user_input = st.text_area("Opis knjige", "")

if user_input:
    vectorized = tfidf.transform([user_input])
    prediction = model.predict(vectorized)[0]
    st.success(f" Predviđeni žanr: **{prediction}**")