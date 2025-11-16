import streamlit as st
import pandas as pd
from recommender import recommend_movies

st.title("🎬 Movie Recommender System")
st.write("Select a movie and get similar recommendations!")

@st.cache_data
def load_data():
    return pd.read_csv("data/sample_movies.csv")

df = load_data()
movie_list = df["title"].tolist()

option = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend"):
    st.subheader("Top 5 Recommendations:")
    results = recommend_movies(option)
    for i, movie in enumerate(results, start=1):
        st.write(f"**{i}. {movie}**")
