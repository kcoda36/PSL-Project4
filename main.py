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

    # Iterate over each movie to predict
    for movie in S_matrix.index:
        # Safely get the user's rating for the movie; default to NaN if not rated
        user_rating = newuser_ratings.get(movie, np.nan)
        
        if pd.isna(user_rating):
            # Get similarity scores for movie i
            S_i = S_matrix.loc[movie].dropna()

            # Find movies rated by the user
            rated_movies = newuser_ratings[newuser_ratings.notna()].index

            # Intersection with movies rated by the user
            relevant_similarities = S_i[S_i.index.isin(rated_movies)]

            if not relevant_similarities.empty:
                # Extract user's ratings for these movies
                user_ratings = newuser_ratings[relevant_similarities.index]

                # Compute the weighted sum
                numerator = (relevant_similarities * user_ratings).sum()
                denominator = relevant_similarities.sum()

                if denominator != 0:
                    prediction = numerator / denominator
                    prediction_scores[movie] = prediction
                else:
                    prediction_scores[movie] = np.nan
            else:
                prediction_scores[movie] = np.nan

    predictions = pd.Series(prediction_scores)

    predictions = predictions.dropna()

    sorted_predictions = predictions.sort_values(ascending=False)

    top_recommendations = sorted_predictions.head(top_n).index.tolist()

    recommendations = ['m' + str(movie_id) for movie_id in top_recommendations]

    return recommendations

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



@st.cache_data  
def load_data():
    movies_file = './ml-1m/movies.dat'
    similarity_file = './similarity_matrix_top30.pkl'

    # load movies
    movies = load_movies(movies_file)

    # load  matrix
    S = load_similarity_matrix(similarity_file)

    return movies, S

movies_df, similarity_matrix = load_data()



# We went with 5 movies to rank. this could be any number though
sample_movie_ids = similarity_matrix.index.tolist()[:5]
sample_movies = movies_df[movies_df['MovieID'].isin(sample_movie_ids)]


st.title("Movie Recommender System: Item-Based Collaborative Filtering")

st.markdown("""
Please rank at least one movie to get reccomendations""")

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

    # error handling if user forgets to select a rating
    if user_ratings_series.isna().all():
        st.warning("Please rate at least one movie to receive recommendations.")
    else:
        # run the myIBCF function
        recommendations = myIBCF(
            newuser_ratings=user_ratings_series,
            S_matrix=similarity_matrix,
            top_n=10
        )

        # Display recommendations
        display_recommendations(recommendations, movies_df, './MovieImages')


