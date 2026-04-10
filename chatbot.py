import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity

# load data
df = pd.read_csv("data.csv")
vectorizer = joblib.load("vectorizer.pkl")
X = joblib.load("tfidf_matrix.pkl")

# clean input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# recommend products
def recommend_products(user_input, top_n=5):
    text = clean_text(user_input)
    user_vec = vectorizer.transform([text])

    sim = cosine_similarity(user_vec, X)
    top_idx = sim[0].argsort()[-50:]
    similar_df = df.iloc[top_idx]

    top_products = (
        similar_df.groupby("asin")["adjusted_rating"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    results = []
    for product_id in top_products.index:
        product_reviews = similar_df[similar_df["asin"] == product_id]
        words = " ".join(product_reviews["clean_review"].astype(str)).split()
        top_words = list(dict.fromkeys(words))[:5]

        results.append({
            "product_id": product_id,
            "score": float(top_products[product_id]),
            "words": top_words
        })
    return results

# ui
st.title("Review-Based Recommender")
st.write("Enter a review or product preference to get top 5 recommendations.")

st.write("Example inputs:")
st.write("- cheap headphone with good bass")
st.write("- durable cable with good quality")
st.write("- affordable and reliable product")

user_input = st.text_area("Your input")

if st.button("Get Recommendations"):
    if user_input.strip() == "" or user_input.lower() in ["hi", "hello", "hey"]:
        st.warning("Please enter a product preference.")
    else:
        try:
            results = recommend_products(user_input)

            st.success("Top 5 Recommended Products")

            for i, item in enumerate(results, 1):
                st.write(f"Product {i}")
                st.write(f"Product ID: {item['product_id']}")
                st.write(f"Score: {item['score']:.2f}")
                st.write(f"Common review words: {', '.join(item['words'])}")
                st.write("---")

        except Exception as e:
            st.error(f"Error: {e}")