import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from PIL import Image

def load_movies(movies_file):
    """Load movies.dat and return a DataFrame."""
    encoding = detect_encoding(movies_file)
    movies = pd.read_csv(
        movies_file,
        sep='::',
        engine='python',
        header=None,
        names=['MovieID', 'Title', 'Genres'],
        encoding=encoding
    )
    return movies

def load_similarity_matrix(similarity_file):
    """Load the similarity matrix from a pickle file."""
    with open(similarity_file, 'rb') as f:
        S = pickle.load(f)
    return S

def detect_encoding(file_path, num_bytes=10000):
    """Detect file encoding using chardet."""
    import chardet
    with open(file_path, 'rb') as f:
        raw_data = f.read(num_bytes)
    result = chardet.detect(raw_data)
    return result['encoding']

def myIBCF(newuser_ratings, S_matrix, top_n=10):
    prediction_scores = {}
    for movie in S_matrix.index:
        user_rating = newuser_ratings.get(movie, np.nan)
        
        if pd.isna(user_rating):
            S_i = S_matrix.loc[movie].dropna()
            rated_movies = newuser_ratings[newuser_ratings.notna()].index
            relevant_similarities = S_i[S_i.index.isin(rated_movies)]

            if not relevant_similarities.empty:
                user_ratings = newuser_ratings[relevant_similarities.index]
                numerator = (relevant_similarities * user_ratings).sum()
                denominator = relevant_similarities.sum()
                prediction_scores[movie] = numerator / denominator if denominator != 0 else np.nan
            else:
                prediction_scores[movie] = np.nan

    predictions = pd.Series(prediction_scores).dropna().sort_values(ascending=False)
    top_recommendations = predictions.head(top_n).index.tolist()
    recommendations = ['m' + str(movie_id) for movie_id in top_recommendations]

    return recommendations

def display_recommendations(recommendations, movies_df, images_folder):
    st.subheader("Top 10 Movie Recommendations:")
    
    if not recommendations:
        st.info("No recommendations available based on your ratings.")
        return

    cols = st.columns(5)
    
    for idx, rec in enumerate(recommendations):
        movie_id = int(rec[1:])  # Extract MovieID from 'mXXXX'
        movie_info = movies_df[movies_df['MovieID'] == movie_id].iloc[0]
        title = movie_info['Title']
        image_path = os.path.join(images_folder, f"{movie_id}.jpg")
        
        col = cols[idx % 5]
        
        with col:
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=150, caption=title)
            else:
                placeholder_path = os.path.join(images_folder, 'placeholder.jpg')
                if os.path.exists(placeholder_path):
                    placeholder = Image.open(placeholder_path)
                    st.image(placeholder, width=150, caption=title)
                else:
                    st.write(f"**{title}**")
                    st.write("Poster not available.")

    st.markdown("---")


@st.cache_data  
def load_data():
    movies_file = './ml-1m/movies.dat'
    similarity_file = './similarity_matrix_top30.pkl'

    movies = load_movies(movies_file)
    S = load_similarity_matrix(similarity_file)

    return movies, S


def app(scope=None, receive=None, send=None):
    movies_df, similarity_matrix = load_data()

    sample_movie_ids = similarity_matrix.index.tolist()[:5]
    sample_movies = movies_df[movies_df['MovieID'].isin(sample_movie_ids)]

    st.title("Movie Recommender System: Item-Based Collaborative Filtering")

    st.markdown("""
    Please rank at least one movie to get recommendations
    """)

    st.subheader("Rate These Sample Movies:")

    user_ratings = {}

    with st.form(key='rating_form'):
        for idx, row in sample_movies.iterrows():
            movie_id = row['MovieID']
            title = row['Title']
            user_input = st.slider(
                label=title,
                min_value=0,
                max_value=5,
                value=0,
                step=1,
                key=movie_id
            )
            rating = user_input if user_input > 0 else np.nan
            user_ratings[movie_id] = rating

        submit_button = st.form_submit_button(label='Get Recommendations')

    if submit_button:
        user_ratings_series = pd.Series(user_ratings)
        if user_ratings_series.isna().all():
            st.warning("Please rate at least one movie to receive recommendations.")
        else:
            recommendations = myIBCF(
                newuser_ratings=user_ratings_series,
                S_matrix=similarity_matrix,
                top_n=10
            )
            display_recommendations(recommendations, movies_df, './MovieImages')

# Expose the app to rsconnect/shiny
app = app
