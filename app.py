from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

docs = [
    "football match and players",
    "cricket world cup tournament",
    "python programming language",
    "machine learning algorithms",
    "hospital doctors and patients",
    "medical treatment and surgery",
    "stock market investment",
    "banking and finance sector",
    "movies and entertainment industry",
    "music concerts and songs"
]

labels = [
    "Sports",
    "Sports",
    "Technology",
    "Technology",
    "Healthcare",
    "Healthcare",
    "Finance",
    "Finance",
    "Entertainment",
    "Entertainment"
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

model = MultinomialNB()
model.fit(X, labels)

st.title("AI Document Classifier")
st.write("Enter any text and the model will predict its category.")
text = st.text_area("Enter text")
if st.button("Classify"):
    prediction = model.predict(
        vectorizer.transform([text])
    )

    st.success(
        f"Predicted Category: {prediction[0]}"
    )
