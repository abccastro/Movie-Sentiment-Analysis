import pandas as pd
import numpy as np
import text_preprocessing as tp
import utils
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_top_closest_movie_names(movie_name, df, top_n=5):
    '''
    This function recommends movie titles that have the closest match to the input
    '''
    top_closest_match_name = []
    try:
        movie_names = df.drop_duplicates()
        movie_names = movie_names.reset_index(drop=True)

        vectorizer = utils.open_pickle_file('movie_recommender_vectorizer.pkl')
        movie_title_vectors = vectorizer.transform(movie_names)

        query_vector = vectorizer.transform([movie_name])
        similarity_scores = cosine_similarity(query_vector, movie_title_vectors)

        top_closest_match_idx = np.argsort(similarity_scores[0])[::-1][:top_n]
        top_closest_match_name = movie_names.loc[top_closest_match_idx.tolist()].values
    except Exception as err:
        print(f"ERROR: {err}")

    return top_closest_match_name


def get_movie_recommendation(movie_name, top_n=10):
    '''
    This function provides movie recommendations based on various features of the given movie (input)
    '''
    recommended_movie_list = []
    
    try:
        movie_recommender_metadata = utils.open_dataset_file('movie_recommender_metadata.csv')
        vg_indices = utils.open_pickle_file('movie_recommender_nn_indices.pkl')
        vg_distances = utils.open_pickle_file('movie_recommender_nn_distances.pkl')

        # query movie title
        movie_idx = movie_recommender_metadata.query("title == @movie_name").index  
        
        if movie_idx.empty:
            # If the movie entered by the user doesn't exist in the records, the program will recommend a new movie similar to the input
            top_closest_match_name = get_top_closest_movie_names(movie_name=movie_name, df=movie_recommender_metadata['title'], top_n=5)

            print(f"'{movie_name}' doesn't exist in the records.\n")
            print(f"You may want to try the movies which are the closest match to the input.")

            for movie in top_closest_match_name.tolist():
                print(f"- {movie}")
        
        else:
            # Place in a separate dataframe the indices and distances, then sort the record by distance in ascending order       
            vg_combined_dist_idx_df = pd.DataFrame()
            for idx in movie_idx:
                vg_dist_idx_df = pd.concat([pd.DataFrame(vg_indices[idx]), pd.DataFrame(vg_distances[idx])], axis=1)
                vg_combined_dist_idx_df = pd.concat([vg_combined_dist_idx_df, vg_dist_idx_df])

            vg_combined_dist_idx_df = vg_combined_dist_idx_df.set_axis(['Index', 'Distance'], axis=1, inplace=False)
            vg_combined_dist_idx_df = vg_combined_dist_idx_df.reset_index(drop=True)
            vg_combined_dist_idx_df = vg_combined_dist_idx_df.sort_values(by='Distance', ascending=True)

            movie_list = movie_recommender_metadata.iloc[vg_combined_dist_idx_df['Index']]

            # Remove any duplicate movie names to provide the user with a diverse selection of recommended movies
            movie_list = movie_list.drop_duplicates(subset=['title'], keep='first')
            
            # Remove from the list any game that shares the same name as the input
            movie_list = movie_list[movie_list['title'] != movie_name]

            # Get the first 10 games in the list
            recommended_movie_list = movie_list.head(top_n)
            recommended_movie_list = recommended_movie_list.drop(columns=['imdb_id'], axis=1)
            
            print(f"Top 10 Recommended Movies for '{movie_name}'")
            recommended_movie_list = recommended_movie_list.reset_index(drop=True)

    except Exception as err:
        print(f"ERROR: {err}")

    return recommended_movie_list

if __name__ == '__main__':

    movie_name = input("Enter movie name: ")

    # return a list of recommended movies
    recommended_movie_list = get_movie_recommendation(movie_name, top_n=10)
    print(recommended_movie_list)