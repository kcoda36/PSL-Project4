import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from scipy.sparse import load_npz

# -----------------------------------
# Load Movies Data
# -----------------------------------
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


def detect_encoding(file_path, num_bytes=10000):
    """Detect file encoding using chardet."""
    import chardet
    with open(file_path, 'rb') as f:
        raw_data = f.read(num_bytes)
    result = chardet.detect(raw_data)
    return result['encoding']


# -----------------------------------
# Load Similarity Matrix
# -----------------------------------
def load_similarity_matrix(similarity_file):
    """Load the similarity matrix from an NPZ file as a CSR matrix."""
    S = load_npz(similarity_file)
    return S


@st.cache_data
def load_data():
    """Load movie data and similarity matrix."""
    movies_file = './ml-1m/movies.dat'
    similarity_file = './similarity_matrix_top30.npz'

    # load movies
    movies = load_movies(movies_file)

    # load sparse similarity matrix
    S = load_similarity_matrix(similarity_file)

    return movies, S


# -----------------------------------
# IBCF Algorithm
# -----------------------------------
def myIBCF(newuser_ratings, S_matrix, top_n=10):
    prediction_scores = {}

    for movie_idx in range(S_matrix.shape[0]):
        user_rating = newuser_ratings.get(movie_idx, np.nan)

        if pd.isna(user_rating):
            # Get similarity scores for movie as a sparse row
            S_i = S_matrix.getrow(movie_idx).toarray().flatten()

            # Find movies rated by the user
            rated_movies = newuser_ratings[newuser_ratings.notna()].index

            # Intersection with movies rated by the user
            relevant_similarities = pd.Series(S_i, index=range(len(S_i))).loc[rated_movies]

            if not relevant_similarities.empty:
                # Extract user's ratings for these movies
                user_ratings = newuser_ratings.loc[relevant_similarities.index]

                # Compute the weighted sum
                numerator = (relevant_similarities * user_ratings).sum()
                denominator = relevant_similarities.sum()

                if denominator != 0:
                    prediction = numerator / denominator
                    prediction_scores[movie_idx] = prediction
                else:
                    prediction_scores[movie_idx] = np.nan
            else:
                prediction_scores[movie_idx] = np.nan

    predictions = pd.Series(prediction_scores)
    predictions = predictions.dropna()
    sorted_predictions = predictions.sort_values(ascending=False)
    top_recommendations = sorted_predictions.head(top_n).index.tolist()
    recommendations = ['m' + str(movie_id) for movie_id in top_recommendations]

    return recommendations


# -----------------------------------
# Display Recommendations
# -----------------------------------
def display_recommendations(recommendations, movies_df, images_folder):
    """Display recommended movies with their titles and poster images."""
    st.subheader("Top 10 Movie Recommendations:")
    
    # Check if there are recommendations to display
    if not recommendations:
        st.info("No recommendations available based on your ratings.")
        return

    # Create a layout with 5 columns per row
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
                # Use a placeholder image if available
                placeholder_path = os.path.join(images_folder, 'placeholder.jpg')
                if os.path.exists(placeholder_path):
                    placeholder = Image.open(placeholder_path)
                    st.image(placeholder, width=150, caption=title)
                else:
                    st.write(f"**{title}**")
                    st.write("Poster not available.")

    st.markdown("---")


# -----------------------------------
# Main Streamlit App
# -----------------------------------
movies_df, similarity_matrix = load_data()

# We went with 5 movies to rank. This could be any number though.
sample_movie_ids = list(range(5))  # Select the first 5 movies for simplicity
sample_movies = movies_df[movies_df['MovieID'].isin(sample_movie_ids)]

st.title("Movie Recommender System: Item-Based Collaborative Filtering")

st.markdown("Please rate at least one movie to get recommendations.")

# Display sample movies with rating sliders
st.subheader("Rate These Sample Movies:")

# Initialize a dictionary to store user ratings
user_ratings = {}

# Create a form for user ratings
with st.form(key='rating_form'):
    for idx, row in sample_movies.iterrows():
        movie_id = row['MovieID']
        title = row['Title']
        
        # Display each movie with a slider for rating
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

    # Error handling if user forgets to select a rating
    if user_ratings_series.isna().all():
        st.warning("Please rate at least one movie to receive recommendations.")
    else:
        # Run the IBCF function
        recommendations = myIBCF(
            newuser_ratings=user_ratings_series,
            S_matrix=similarity_matrix,
            top_n=10
        )

        # Display recommendations
        display_recommendations(recommendations, movies_df, './MovieImages')
