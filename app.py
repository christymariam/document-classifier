from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

docs = [
    "football match and players",
    "cricket world cup",
    "python programming language",
    "machine learning project"
]

labels = ["Sports", "Sports", "Technology", "Technology"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

model = MultinomialNB()
model.fit(X, labels)

st.title("Document Classifier")

text = st.text_area("Enter a document")

if st.button("Classify"):
    prediction = model.predict(
        vectorizer.transform([text])
    )

    st.success(
        f"Predicted Category: {prediction[0]}"
    )
