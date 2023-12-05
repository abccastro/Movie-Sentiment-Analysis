import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import datetime
import altair as alt
import torch
import text_preprocessing as tp
import utils
import contractions
import spacy
import pickle
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer,util

nltk.download('stopwords')

# def filter_reviews(input_val):
#     # Load the Sentence Transformer model
#     model = SentenceTransformer('msmarco-MiniLM-L6-cos-v5')

#     # Read the DataFrame
#     df = pd.read_csv("./dataset/cleaned_data/reviews.csv")

#     # Sample two rows from each 'movie' group
#     sampled_df = df.groupby('movie', group_keys=False).apply(lambda group: group.sample(n=4))

#     # Extract the review details and create a list of reviews
#     reviews_list = sampled_df["review_detail"].tolist()

#     # Select a subset of reviews for testing
#     test = reviews_list[16200:]

#     # Encode the reviews into embeddings
#     embeddings = model.encode(test)

#     # Encode the input review
#     input_review = input_val
#     emb_input_review = model.encode(input_review)

#     # Initialize an empty list to store matching reviews
#     matching_reviews = []

#     # Set a threshold for similarity, e.g., 0.3
#     threshold = 0.2

#     # Iterate through the embeddings and check cosine similarity
#     for i, emb in enumerate(embeddings):
#         cos_sim = util.cos_sim([emb_input_review], [emb])[0][0]

#         # Check if similarity is above the threshold
#         if cos_sim > threshold:
#             matching_reviews.append((i, test[i], cos_sim))

#     # Create a DataFrame with the matching reviews
#     df_result = pd.DataFrame(matching_reviews, columns=['Index', 'Review', 'Cosine Similarity'])

#     return df_result

# # Example usage
input_review = "Disney"
condition_for_download = False


# def filter_reviews(input_val):
#     # Load the Sentence Transformer model
#     model = SentenceTransformer('msmarco-MiniLM-L6-cos-v5')

#     # Read the DataFrame
#     df = pd.read_csv("./dataset/cleaned_data/reviews.csv")

#     # Sample two rows from each 'movie' group
#     sampled_df = df.groupby('movie', group_keys=False).apply(lambda group: group.sample(n=4))

#     # Extract the review details and create a list of reviews
#     reviews_list = sampled_df["review_detail"].tolist()

#     # Select a subset of reviews for testing
#     test = reviews_list[16200:]
    

#     # Encode the reviews into embeddings
#     embeddings = model.encode(test)

#     # Encode the input review
#     input_review = input_val
#     emb_input_review = model.encode(input_review)

#     # Initialize an empty list to store matching reviews
#     matching_reviews = []

#     # Set a threshold for similarity, e.g., 0.2
#     threshold = 0.2

#     # Iterate through the embeddings and check cosine similarity
#     for i, emb in enumerate(embeddings):
#         cos_sim = util.cos_sim([emb_input_review], [emb])[0][0]

#         # Check if similarity is above the threshold
#         if cos_sim > threshold:
#             matching_reviews.append((i, test[i], cos_sim))
#             # matching_reviews.append((i, test[i], cos_sim,sampled_df.iloc[i]['roberta_sentiment'],))

            

#     # Create a DataFrame with the matching reviews
#     df_result = pd.DataFrame(matching_reviews, columns=['Index', 'Review', 'Cosine Similarity'])
#     # df_result = pd.DataFrame(matching_reviews, columns=['Index', 'Review', 'Cosine Similarity','roberta_sentiment'])

#     # Add 'roberta_sentiment' column to the result DataFrame
#     df_result['roberta_sentiment'] = sampled_df.iloc[df_result['Index']]['roberta_sentiment'].tolist()
#     df_result['titles'] = sampled_df.iloc[df_result['Index']]['movie'].tolist()

#     return df_result

##-----------> ADD EMB TO CSV
# def filter_reviews(input_val):
# def filter_reviews():
#     # Load the Sentence Transformer model
#     model = SentenceTransformer('msmarco-MiniLM-L6-cos-v5')

#     # Read the DataFrame
#     df = pd.read_csv("./dataset/cleaned_data/reviews.csv")

#     # Sample two rows from each 'movie' group
#     # sampled_df = df.groupby('movie', group_keys=False).apply(lambda group: group.sample(n=4))

#     # Extract the review details and create a list of reviews
#     # reviews_list = sampled_df["review_detail"].tolist()

#     # Select a subset of reviews for testing
#     test = df[201:4000]
#     # test = reviews_list[16200:]

#     # Initialize an empty list to store matching reviews
#     matching_reviews = []

#     # Set a threshold for similarity, e.g., 0.2
#     # threshold = 0.2

#     # Iterate through the reviews
#     for i, row in test.iterrows():
#         # Check if review text is non-empty
#         if pd.notna(row['review_detail']) and row['review_detail'].strip():
#             review_text = row['review_detail']

#             # Check if input review is non-empty
#             # if input_val.strip():
#                 # Encode the input review
#                 # emb_input_review = model.encode(input_val)

#                 # Encode the current review
#             emb = model.encode(review_text)

#             # Compute cosine similarity
#             # cos_sim = util.cos_sim([emb_input_review], [emb])[0][0]

#             # Check if similarity is above the threshold
#             # if cos_sim > threshold:
#             metadata = {
#                 'Index': i,
#                 'Review': review_text,
#                 'Emb': emb,
#                 'roberta_sentiment': row['roberta_sentiment'],
#                 'titles': row['movie'],
#                 'date':row['review_date']
#             }
#             matching_reviews.append(metadata)

#     # Create a DataFrame with the matching reviews
#     df_result = pd.DataFrame(matching_reviews)

#     # Specify the path to the existing CSV file
#     existing_csv_path = './emb_file.csv'

#     import os
#     # Check if the file exists before reading
#     if os.path.isfile(existing_csv_path) and os.path.getsize(existing_csv_path) > 0:
#         # Check if the file has the same structure
#         existing_columns = pd.read_csv(existing_csv_path, nrows=1).columns
#         if all(col in existing_columns for col in df_result.columns):
#             # Append df_result to the existing CSV file
#             df_result.to_csv(existing_csv_path, mode='a', header=False, index=False)
#         else:
#             # If the file structure is different, create a new CSV file
#             df_result.to_csv(existing_csv_path, index=False)
        
#         return {"res": "Good-In"}
#     else:
#         # If the file doesn't exist or is empty, create a new CSV file
#         df_result.to_csv(existing_csv_path, index=False)
#         return {"res": "Good-Out"}

# import streamlit as st
# import pandas as pd
# from sentence_transformers import SentenceTransformer

def filter_reviews(progress_callback=None):
    # Load the Sentence Transformer model
    model = SentenceTransformer('msmarco-MiniLM-L6-cos-v5')

    # Read the DataFrame
    # df = pd.read_csv("./dataset/cleaned_data/reviews_with_imdbid.csv")
    df = pd.read_csv("./dataset/cleaned_data/reviews.csv")

    # Initialize an empty list to store matching reviews
    matching_reviews = []

    #  Select a subset of reviews for testing
    test = df[320000:420000]
    # test = reviews_list[16200:]

    # Set a threshold for similarity, e.g., 0.2
    # threshold = 0.2

    # Total number of iterations
    total_iterations = len(test)

    for i, row in test.iterrows():
    # Check if review text is non-empty
        if pd.notna(row['review_detail']) and row['review_detail'].strip():
            review_text = row['review_detail']

            # Encode the current review
            emb = model.encode(review_text)

            metadata = {
                'Index': i,
                'Review': review_text,
                'Emb': emb,
                'roberta_sentiment': row['roberta_sentiment'],
                'titles': row['movie'],
                'date': row['review_date']
            }
            matching_reviews.append(metadata)

            # Report progress
            progress = min((i + 1) / total_iterations, 1.0)  # Ensure progress doesn't exceed 1.0
            if progress_callback:
                progress_callback(progress)

    # Create a DataFrame with the matching reviews
    df_result = pd.DataFrame(matching_reviews)

    # Specify the path to the existing CSV file
    existing_csv_path = './dataset/emb/emb_file.csv'

    # Check if the file exists before reading
    if df_result.empty:
        return {"res": "No Matching Reviews"}

    # Check if the file has the same structure
    existing_columns = pd.read_csv(existing_csv_path, nrows=1).columns

    if all(col in existing_columns for col in df_result.columns):
        # Append df_result to the existing CSV file
        df_result.to_csv(existing_csv_path, mode='a', header=False, index=False)
    else:
        # If the file structure is different, create a new CSV file
        df_result.to_csv(existing_csv_path, index=False)

    return {"res": "Good-In"}

    



###-------> compare EMBEDDINGS
import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# ---->>>>>>>REAL
# 
# def compare_with_existing_embeddings(all_chunks, input_val):
#     # Load the Sentence Transformer model
#     model = SentenceTransformer('msmarco-MiniLM-L6-cos-v5')
    
#     # Load the existing DataFrame with embeddings
#     df = all_chunks.copy()  # Make a copy to avoid modifying the original DataFrame
    
#     # Encode the input review
#     input_emb = model.encode(input_val).astype(np.float32)

#     # Convert the 'Emb' column from string to NumPy array
#     df['Emb'] = df['Emb'].apply(lambda x: np.fromstring(x[1:-1], sep=' ').astype(np.float32))

#     # Initialize an empty list to store matching reviews
#     matching_reviews = []

#     # Set a threshold for similarity, e.g., 0.4
#     threshold = 0.4

#     # Add a Streamlit progress bar
#     progress_bar = st.progress(0)

#     # Iterate through the reviews
#     total_reviews = len(df)
#     for i, row in df.iterrows():
#         # Extract the review details
#         review_text = row['Review']

#         # Encode the current review
#         emb = row['Emb']

#         # Compute cosine similarity
#         cos_sim = util.pytorch_cos_sim(torch.tensor([input_emb]), torch.tensor([emb])).item()

#         # Check if similarity is above the threshold
#         if cos_sim > threshold:
#             metadata = {
#                 'Index': i,
#                 'Review': review_text,
#                 'Cosine Similarity': cos_sim,
#                 'roberta_sentiment': row['roberta_sentiment'],
#                 'titles': row['titles'],
#                 'date': row['date']
#             }
#             matching_reviews.append(metadata)

#         # Update the progress bar
#         progress_bar.progress((i + 1) / total_reviews)

#     # Close the progress bar once processing is complete
#     progress_bar.empty()

#     # Create a DataFrame with the matching reviews
#     df_result = pd.DataFrame(matching_reviews)
#     df_result.sort_values(by='Cosine Similarity', ascending=False)
#     return df_result

# def compare_with_existing_embeddings(all_chunks, input_val,sdate="",edate=""):
#     # Load the Sentence Transformer model
#     model = SentenceTransformer('msmarco-MiniLM-L6-cos-v5')
    
#     # Load the existing DataFrame with embeddings
#     df = all_chunks.copy()  # Make a copy to avoid modifying the original DataFrame

#     # Check if sdate is after edate
#     try:
#         sdate_num = int(sdate)
#         edate_num = int(edate)
#     except ValueError:
#         st.warning("Invalid date format. Please provide valid numerical dates.")
#         return pd.DataFrame()
    
#     if sdate_num > edate_num:
#         st.warning("Start date cannot be after end date. Please provide valid date ranges.")
#         return pd.DataFrame()

#     if not df.empty and sdate_num and edate_num:
#         df['date'] = pd.to_numeric(df['date'], errors='coerce')  # Convert 'date' to numeric
#         df = df[(df['date'] >= sdate_num) & (df['date'] <= edate_num)]
        
#     # Encode the input review
#     input_emb = model.encode(input_val).astype(np.float32)

#     # Convert the 'Emb' column from string to NumPy array
#     df['Emb'] = df['Emb'].apply(lambda x: np.fromstring(x[1:-1], sep=' ').astype(np.float32))

#     # Initialize an empty list to store matching reviews
#     matching_reviews = []

#     # Set a threshold for similarity, e.g., 0.4
#     threshold = 0.4

#     # Add a Streamlit progress bar
#     progress_bar = st.progress(0)
  

#     # Iterate through the reviews
#     total_reviews = len(df)
#     for i, row in df.iterrows():
#         # Extract the review details
#         review_text = row['Review']

#         # Encode the current review
#         emb = row['Emb']

#         # Compute cosine similarity
#         cos_sim = util.pytorch_cos_sim(torch.tensor([input_emb]), torch.tensor([emb])).item()

#         # Check if similarity is above the threshold
#         if cos_sim > threshold:
#             metadata = {
#                 'Index': i,
#                 'Review': review_text,
#                 'cosine_similarity': cos_sim,
#                 'sentiment': row['viwers_sentiment'],
#                 'titles': row['movie_title'],
#                 'date': row['movie_year'],
#             }
#             matching_reviews.append(metadata)

#         # Update the progress bar
#         progress_bar.progress((i + 1) / total_reviews)

#     # Close the progress bar once processing is complete
#     progress_bar.empty()

#     # Create a DataFrame with the matching reviews and sort it
#     df_result = pd.DataFrame(matching_reviews)
    
#     # Filter the DataFrame based on date if both start and end dates are provided
    

#     if not df_result.empty:
#         df_result = df_result.sort_values(by='cosine_similarity', ascending=False)
#         return df_result
#     else:
#         # Handle the case where the DataFrame is empty
#         return pd.DataFrame()

def compare_with_existing_embeddings(all_chunks, input_val, sdate="", edate=""):
    # Load the Sentence Transformer model
    model = SentenceTransformer('msmarco-MiniLM-L6-cos-v5')
    
    # Load the existing DataFrame with embeddings
    df = all_chunks.copy()  # Make a copy to avoid modifying the original DataFrame

    # Check if sdate is after edate
    try:
        sdate_num = int(sdate)
        edate_num = int(edate)
    except ValueError:
        st.warning("Invalid date format. Please provide valid numerical dates.")
        return pd.DataFrame()
    
    if sdate_num > edate_num:
        st.warning("Start date cannot be after end date. Please provide valid date ranges.")
        return pd.DataFrame()

    # Convert 'date' to numeric
    df['movie_year'] = pd.to_numeric(df['movie_year'], errors='coerce')

    # Filter the DataFrame based on date if both start and end dates are provided
    if not df.empty and sdate_num and edate_num:
        df = df[(df['movie_year'] >= sdate_num) & (df['movie_year'] <= edate_num)]
        
    # Encode the input review
    input_emb = model.encode(input_val).astype(np.float32)

    # Convert the 'Emb' column from string to NumPy array
    df['Emb'] = df['Emb'].apply(lambda x: np.fromstring(x[1:-1], sep=' ').astype(np.float32))

    # Initialize an empty list to store matching reviews
    matching_reviews = []

    # Set a threshold for similarity, e.g., 0.4
    threshold = 0.4

    # Add a Streamlit progress bar
    # progress_bar = st.progress(0)

    # Iterate through the reviews
    total_reviews = len(df)
    for i, row in df.iterrows():
        # Extract the review details
        review_text = row['Review']

        # Encode the current review
        emb = row['Emb']

        # Compute cosine similarity
        cos_sim = util.pytorch_cos_sim(torch.tensor([input_emb]), torch.tensor([emb])).item()

        # Check if similarity is above the threshold
        if cos_sim > threshold:
            metadata = {
                'Index': i,
                'Review': review_text,
                'cosine_similarity': cos_sim,
                'sentiment': row['viwers_sentiment'],
                'titles': row['movie_title'],
                'date': row['movie_year'],
            }
            matching_reviews.append(metadata)

        # Update the progress bar
        # progress_bar.progress((i + 1) / total_reviews)

    # Close the progress bar once processing is complete
    # progress_bar.empty()

    # Create a DataFrame with the matching reviews and sort it
    df_result = pd.DataFrame(matching_reviews)
    
    if not df_result.empty:
        df_result = df_result.sort_values(by='cosine_similarity', ascending=False)
        return df_result
    else:
        # Handle the case where the DataFrame is empty
        return pd.DataFrame()


def filter_by_movie_title(df, input_title,sdate="",edate=""):
# Check if the 'movie' column exists in the DataFrame
    # if 'movie_title' not in df.columns:
    #     # raise ValueError("The 'movie' column does not exist in the DataFrame.")
    #     return pd.DataFrame()
    # # Lowercase and strip whitespace from the 'movie_title' column and input title
    # df['movie_title_lower'] = df['movie_title'].str.lower().str.strip()
    # input_title_lower = input_title.lower().strip()

    # # Filter the DataFrame based on the lowercase and stripped input movie title
    # filtered_df = df[df['movie_title_lower'] == input_title_lower]

    # # Drop the temporary lowercase column
    # filtered_df = filtered_df.drop(columns=['movie_title_lower'])

    # return filtered_df

    #  Check if the 'movie_title' column exists in the DataFrame
    if 'movie_title' not in df.columns:
        # raise ValueError("The 'movie_title' column does not exist in the DataFrame.")
        return pd.DataFrame()

    # Lowercase and strip whitespace from the 'movie_title' column and input title
    df['movie_title_lower'] = df['movie_title'].str.lower().str.strip()
    input_title_lower = input_title.lower().strip()

    # Filter the DataFrame based on the lowercase and stripped input movie title
    filtered_df = df[df['movie_title_lower'] == input_title_lower]

    # Check if both start date and end date are provided
    if sdate and edate:
        # Convert dates to numbers (assuming they are strings)
        sdate = int(sdate)
        edate = int(edate)

        # Ensure start date is not higher than end date
        if sdate > edate:
            st.warning("Start date should not be higher than end date.")
            return pd.DataFrame()
        
        filtered_df['release_year'] = pd.to_numeric(filtered_df['release_year'], errors='coerce')

        # Filter the DataFrame based on the date range
        filtered_df = filtered_df[(filtered_df['release_year'] >= sdate) & (filtered_df['release_year'] <= edate)]

    # Drop the temporary lowercase column
    filtered_df = filtered_df.drop(columns=['movie_title_lower'])

    return filtered_df

    # # Iterate through the reviews
    # total_reviews = len(df)
    # for i, row in df.iterrows():
    #     # Extract the review details
    #     review_text = row['Review']

    #     # Encode the current review
    #     emb = row['Emb']

    #     # Compute cosine similarity
    #     cos_sim = util.pytorch_cos_sim(torch.tensor([input_emb]), torch.tensor([emb])).item()

    #     # Check if similarity is above the threshold
    #     if cos_sim > threshold:
    #         # metadata = {
    #         #     'Index': i,
    #         #     'Review': review_text,
    #         #     'cosine_similarity': cos_sim,
    #         #     'roberta_sentiment': row['roberta_sentiment'],
    #         #     'titles': row['titles'],
    #         #     'date': row['date']
    #         # }
    #         metadata = {
    #                     'Index': i,
    #                     'Review': review_text,
    #                     'cosine_similarity': cos_sim,
    #                     'sentiment': row['viwers_sentiment'],
    #                     'titles': row['movie_title'],
    #                     'date':row['movie_year'],
    #                 }
    #         matching_reviews.append(metadata)

    #     # Update the progress bar
    #     progress_bar.progress((i + 1) / total_reviews)

    # # Close the progress bar once processing is complete
    # progress_bar.empty()

    # # Create a DataFrame with the matching reviews and sort it
    # df_result = pd.DataFrame(matching_reviews)
    # if not df_result.empty:
    #     df_result = df_result.sort_values(by='cosine_similarity', ascending=False)
    #     return df_result
    # else:
    #     # Handle the case where the DataFrame is empty
    #     return pd.DataFrame() 



# def compare_with_existing_embeddings(all_chunks, input_val):
#     # Load the Sentence Transformer model
#     model = SentenceTransformer('msmarco-MiniLM-L6-cos-v5')
    
#     # Load the existing DataFrame with embeddings
#     df = all_chunks.copy()  # Make a copy to avoid modifying the original DataFrame
    
#     # Encode the input review
#     input_emb = model.encode(input_val)

#     # Add a Streamlit progress bar
#     progress_bar = st.progress(0)

#     # Initialize an empty list to store matching reviews
#     matching_reviews = []

#     # Set a threshold for similarity, e.g., 0.4
#     threshold = 0.4

#     # Iterate through the reviews
#     total_reviews = len(df)
#     for i, row in df.iterrows():
#         # Extract the review details
#         review_text = row['Review']

#         # Encode the current review
#         emb = row['Emb']

#         # Compute cosine similarity
#         cos_sim = util.pytorch_cos_sim(torch.tensor([input_emb]), torch.tensor([emb])).item()

#         # Check if similarity is above the threshold
#         if cos_sim > threshold:
#             metadata = {
#                 'Index': i,
#                 'Review': review_text,
#                 'Cosine Similarity': cos_sim,
#                 'roberta_sentiment': row['roberta_sentiment'],
#                 'titles': row['titles'],
#                 'date': row['date']
#             }
#             matching_reviews.append(metadata)

#         # Update the progress bar
#         progress_bar.progress((i + 1) / total_reviews)

#     # Close the progress bar once processing is complete
#     progress_bar.empty()

#     # Create a DataFrame with the matching reviews
#     df_result = pd.DataFrame(matching_reviews)
    # df_result.sort_values(by='Cosine Similarity', ascending=False, inplace=True)
    # return df_result


###-------> SHOW ALL ROWS
def read_all(existing_csv_path):
    # Load the existing DataFrame with embeddings
    df = pd.read_csv(existing_csv_path)
    return df



###-------> FILTER CSV WITH MOVIE TITLES
def filter_titles(csv_path, movie_titles):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Initialize an empty list to store matching reviews
    matching_reviews = []

    # Iterate through the reviews
    for i, row in df.iterrows():
        # Extract the movie title and convert to lowercase
        title = row['title']

        # Check if any word in the title (lowercase and stripped) is present in the list of movie titles
        if any(word.strip() in title for word in movie_titles):
            metadata = {
                'imdb_id': row['imdb_id'],
                'title': row['title'],
                'release_year': row['release_year'],
                'genre': row['genre'],
                'production_companies': row['production_companies'],
                'cast': row['cast'],
                'director': row['director'],
                'budget': row['budget'],
                'revenue': row['revenue'],
                'runtime': row['runtime'],
                'vote_average': row['vote_average'],
                'viewers_sentiment': row['viwers_sentiment']
            }
            matching_reviews.append(metadata)

    # Create a DataFrame with the matching reviews
    df_result = pd.DataFrame(matching_reviews)

    return df_result

@st.cache_data
###-------> FILTER CSV WITH MOVIE TITLES
def filter_titles_by_id(csv_path, imdb_id):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Initialize an empty list to store matching reviews
    matching_reviews = []

    # Iterate through the reviews
    for i, row in df.iterrows():
        # Check if the 'imdb_id' matches the specified IMDb ID
        if str(row['imdb_id']).strip() == str(imdb_id).strip():
            metadata = {
                'imdb_id': row['imdb_id'],
                'title': row['title'],
                'release_year': row['release_year'],
                'genre': row['genre'],
                'production_companies': row['production_companies'],
                'cast': row['cast'],
                'director': row['director'],
                'budget': row['budget'],
                'revenue': row['revenue'],
                'runtime': row['runtime'],
                'vote_average': row['vote_average'],
                'viewers_sentiment': row['viwers_sentiment']
            }
            matching_reviews.append(metadata)

    # Create a DataFrame with the matching reviews
    df_result = pd.DataFrame(matching_reviews)

    return df_result

def filter_cast(csv_path, movie_titles):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Initialize a dictionary to store cast lists for each movie
    movie_casts = {title: set() for title in movie_titles}

    # Iterate through the reviews
    for i, row in df.iterrows():
        # Extract the movie title and convert to lowercase
        title = row['title']

        # Check if the current movie is in the specified list
        if title in movie_titles:
            # Extract the cast list and add it to the set for the current movie
            cast_list = eval(row['cast']) if pd.notna(row['cast']) else []
            movie_casts[title].update(cast_list)

    # Find the common cast members
    common_cast_members = set.intersection(*movie_casts.values())

    # Create a dictionary with the common cast members
    result_dict = {
        'Titles': movie_titles,
        'Common_Cast_Members': list(common_cast_members)
    }

    # Convert the dictionary to a DataFrame
    result_df = pd.DataFrame([result_dict])

    return result_df


# def filter_top10_pos(input_val):

#     # Filter only "Positive" and "Strongly Positive" sentiments
#     positive_sentiments = ['Positive', 'Strongly Positive']
#     df_result = df_result[input_val['sentiment'].isin(positive_sentiments)]

#     # Sort the DataFrame by Cosine Similarity in descending order
#     df_result = df_result.sort_values(by='cosine_similarity', ascending=False).head(10)

#     return df_result

def filter_top10_pos(df_result):
    # Filter only "Positive" and "Strongly Positive" sentiments
    positive_sentiments = ['positive', 'strongly positive']
    
    # Check if the DataFrame is not empty and contains the required columns
    if not df_result.empty and 'sentiment' in df_result.columns and 'cosine_similarity' in df_result.columns:
        df_result = df_result[df_result['sentiment'].isin(positive_sentiments)]

        # Sort the DataFrame by Cosine Similarity in descending order
        df_result = df_result.sort_values(by='cosine_similarity', ascending=False).head(10)

        return df_result
    else:
        # Handle the case where the DataFrame is empty or missing required columns
        return pd.DataFrame()


def filter_top10_neg(df_result):
    # Filter only "Positive" and "Strongly Positive" sentiments
    positive_sentiments = ['negative', 'strongly negative']
    
    # # Check if the DataFrame is not empty and contains the required columns
    # if not df_result.empty and 'sentiment' in df_result.columns and 'cosine_similarity' in df_result.columns:
    #     df_result = df_result[df_result['sentiment'].isin(positive_sentiments)]
    if not df_result.empty and 'sentiment' in df_result.columns and 'cosine_similarity' in df_result.columns:
    # Lowercase the 'sentiment' column and create a temporary lowercase column
        df_result['sentiment_lower'] = df_result['sentiment'].str.lower()

        # Filter the DataFrame based on the lowercase 'sentiment' column
        df_result = df_result[df_result['sentiment_lower'].isin(positive_sentiments)]

        # Drop the temporary lowercase column
        df_result = df_result.drop(columns=['sentiment_lower'])

        # Sort the DataFrame by Cosine Similarity in descending order
        df_result = df_result.sort_values(by='cosine_similarity', ascending=False).head(10)

        return df_result
    else:
        # Handle the case where the DataFrame is empty or missing required columns
        return pd.DataFrame()

def filter_top10_neu(df_result):
    # Filter only "Positive" and "Strongly Positive" sentiments
    positive_sentiments = ['neutral']
    
    # Check if the DataFrame is not empty and contains the required columns
    if not df_result.empty and 'sentiment' in df_result.columns and 'cosine_similarity' in df_result.columns:
        df_result = df_result[df_result['sentiment'].isin(positive_sentiments)]

        # Sort the DataFrame by Cosine Similarity in descending order
        df_result = df_result.sort_values(by='cosine_similarity', ascending=False).head(10)

        return df_result
    else:
        # Handle the case where the DataFrame is empty or missing required columns
        return pd.DataFrame()

def filter_top10movie_pos(df_result):
    # Filter only "Positive" and "Strongly Positive" sentiments
    positive_sentiments = ['positive', 'strongly positive']
    
    # Check if the DataFrame is not empty and contains the required columns
    if not df_result.empty and 'viwers_sentiment' in df_result.columns:
        df_result = df_result[df_result['viwers_sentiment'].isin(positive_sentiments)]

        # Sort the DataFrame by Cosine Similarity in descending order
        df_result = df_result.head(10)

        return df_result
    else:
        # Handle the case where the DataFrame is empty or missing required columns
        return pd.DataFrame()


def filter_top10movie_neg(df_result):
    # Filter only "Positive" and "Strongly Positive" sentiments
    positive_sentiments = ['negative', 'Strongly Negative']
    
    # # Check if the DataFrame is not empty and contains the required columns
    # if not df_result.empty and 'sentiment' in df_result.columns:
    #     df_result = df_result[df_result['sentiment'].isin(positive_sentiments)]
    if not df_result.empty and 'viwers_sentiment' in df_result.columns:
    # Lowercase the 'sentiment' column and create a temporary lowercase column
        df_result['sentiment_lower'] = df_result['viwers_sentiment'].str.lower()

        # Filter the DataFrame based on the lowercase 'sentiment' column
        df_result = df_result[df_result['sentiment_lower'].isin(positive_sentiments)]

        # Drop the temporary lowercase column
        df_result = df_result.drop(columns=['sentiment_lower'])

        # Sort the DataFrame by Cosine Similarity in descending order
        df_result = df_result.head(10)

        return df_result
    else:
        # Handle the case where the DataFrame is empty or missing required columns
        return pd.DataFrame()

def filter_top10movie_neu(df_result):
    # Filter only "Positive" and "Strongly Positive" sentiments
    positive_sentiments = ['neutral']
    
    # Check if the DataFrame is not empty and contains the required columns
    if not df_result.empty and 'viwers_sentiment' in df_result.columns:
        df_result = df_result[df_result['viwers_sentiment'].isin(positive_sentiments)]

        # Sort the DataFrame by Cosine Similarity in descending order
        df_result = df_result.head(10)

        return df_result
    else:
        # Handle the case where the DataFrame is empty or missing required columns
        return pd.DataFrame()

def process_input_text(df,input_movie_text,review_text):
    # Split the input text by "/"
    # values = input_text.split("/")

    # Check if there are at least two values
    # if len(values) >= 2:
        # Extract the title and review
    title = input_movie_text
    review = review_text

    # Filter the DataFrame based on the title
    filtered_df = df[df['movie_title'].str.lower().str.strip() == title.lower()]

    # Create a new DataFrame to store the modified data
    new_df = filtered_df.copy()

    # Replace the "Review" column with the review variable
    new_df['Review'] = review

    # Return the new DataFrame
    return new_df
    # else:
    #     # Handle the case where there are not enough values
    #     print("Error: Input text should contain at least two values.")
    #     return pd.DataFrame()
    
# ---------->>>>>>>>>> START

nlp = spacy.load("en_core_web_sm")
list_of_stopwords = set(stopwords.words('english'))

# initializer dictionaries for data preprocessing
emoji_dict = tp.get_emojis()
slang_word_dict = tp.get_slang_words(webscraped=False)


def conduct_text_preprocessing(text, set_n=1):
    '''
    This function contains processes for data cleaning
    '''
    # try:
    if set_n == 1:
        # Remove non-grammatical text
        text = tp.remove_email_address(text)
        text = tp.remove_hyperlink(text)

        # Replace non-ascii characters as there are Python libraries limiting this feature
        text = tp.replace_nonascii_characters(text)

        # Replace emojis with English word/s
        text = emoji_dict.replace_keywords(text)

        # Handle contractions
        text = contractions.fix(text)

        # Replace slang words
        text = slang_word_dict.replace_keywords(text)
    
    elif set_n == 3:
        # Remove non-alphanumeric characters except for the following
        text = tp.remove_non_alphanumeric_char(text)

        # Remove leading and trailing whitespaces
        text = text.strip()

        # Replace multiple whitespaces with a single space
        text = tp.replace_whitespace(text)

        # Remove stopwords
        text = tp.remove_stopwords(text, list_of_stopwords)

    # except Exception as err:
    #     print(f"ERROR: {err}")
    #     print(f"Input Text: {text}")

    return text


def remove_ner(df):
    '''
    This function removes Named Entity Recognition (NER)s
    '''
    # Dummy dataframe for extracted NERs
    column_names = ['PERSON', 'WORK_OF_ART', 'LOCATION', 'DATE_TIME', 'ORGANIZATION', 'PRODUCT', 'EVENT', 'LANGUAGE']
    movie_name_entities_df = pd.DataFrame(columns=column_names)

    # Extract and remove name entities
    df['Review'], movie_name_entities_df = tp.extract_name_entity(df, nlp, movie_name_entities_df)
    return df


def generate_sentiment(df):
    '''
    This function generates sentiments to the reviews using Naive Bayes model
    '''
    sentiment_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predictions = []

    # try:
    # open pickle file for countvectorizer
    vectorizer = utils.open_pickle_file('sentiment_analysis_vectorizer.pkl')
    
    # open pickle file for naive bayes model
    model = utils.open_pickle_file('sentiment_analysis_model.pkl')
    
    # predict the sentiment of the review
    vectorized_text = vectorizer.transform(df)
    predictions = model.predict(vectorized_text)  
    predictions  = pd.DataFrame(predictions).replace(sentiment_dict)

    # except Exception as err:
    #     print(f"ERROR: {err}")

    return predictions

# ---------->>>>>>>>>> END OF SENTIMENT

# ---------->>>>>>>>>> START OF RECOMMENDATION

def get_top_closest_movie_names(movie_name, df, top_n=10):
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
    
    movie_recommender_metadata = utils.open_dataset_file('movie_metadata.csv')
    vg_indices = utils.open_pickle_file('movie_recommender_nn_indices.pkl')
    vg_distances = utils.open_pickle_file('movie_recommender_nn_distances.pkl')

    # query movie title
    movie_idx = movie_recommender_metadata.query("title == @movie_name").index  
    
    if movie_idx.empty:
        # If the movie entered by the user doesn't exist in the records, the program will recommend a new movie similar to the input
        top_closest_match_name = get_top_closest_movie_names(movie_name=movie_name, df=movie_recommender_metadata['title'], top_n=top_n)

        return pd.DataFrame(top_closest_match_name,columns=["Movie Title"]),0
        # print(f"'{movie_name}' doesn't exist in the records.\n")
        # print(f"You may want to try the movies which are the closest match to the input.")


        # for movie in top_closest_match_name.tolist():
            # print(f"- {movie}")

    
    else:
        # Place in a separate dataframe the indices and distances, then sort the record by distance in ascending order       
        vg_combined_dist_idx_df = pd.DataFrame()
        for idx in movie_idx:
            vg_dist_idx_df = pd.concat([pd.DataFrame(vg_indices[idx]), pd.DataFrame(vg_distances[idx])], axis=1)
            vg_combined_dist_idx_df = pd.concat([vg_combined_dist_idx_df, vg_dist_idx_df])

        vg_combined_dist_idx_df = vg_combined_dist_idx_df.set_axis(['Index', 'Distance'], axis=1)
        # vg_combined_dist_idx_df = vg_combined_dist_idx_df.set_axis(['Index', 'Distance'], axis=1, inplace=False)
        vg_combined_dist_idx_df = vg_combined_dist_idx_df.reset_index(drop=True)
        vg_combined_dist_idx_df = vg_combined_dist_idx_df.sort_values(by='Distance', ascending=True)

        movie_list = movie_recommender_metadata.iloc[vg_combined_dist_idx_df['Index']]

        # Remove any duplicate movie names to provide the user with a diverse selection of recommended movies
        movie_list = movie_list.drop_duplicates(subset=['title'], keep='first')
        
        # Remove from the list any game that shares the same name as the input
        # movie_list = movie_list[movie_list['title'] != movie_name]

        # Get the first 10 games in the list
        recommended_movie_list = movie_list.head(top_n)
        recommended_movie_list = recommended_movie_list.drop(columns=['imdb_id'], axis=1)
        
        # print(f"Top 10 Recommended Movies for '{movie_name}'")
        recommended_movie_list = recommended_movie_list.reset_index(drop=True)

    # except Exception as err:
        # print(f"ERROR: {err}")

    return recommended_movie_list,1

# ---------->>>>>>>>>> END OF RECOMMENDATION

def generate_report(filtered_df):
    # Calculate some hypothetical metrics using the numerical columns
    total_salary = filtered_df['Salary'].sum()
    average_work_hours = filtered_df['Work_Hours'].mean()

    # Create a bar chart
    bar_chart = alt.Chart(filtered_df).mark_bar().encode(
        x='Department',
        y='Salary',
        color='Gender',
        tooltip=['Department', 'Salary']
    ).properties(
        title='Salary by Department (colored by Gender)',
        width=300
    )

    # Create a line chart
    line_chart = alt.Chart(filtered_df).mark_line().encode(
        x='Department',
        y='Work_Hours',
        color='Gender',
        tooltip=['Department', 'Work_Hours']
    ).properties(
        title='Work Hours by Department (colored by Gender)',
        width=300
    )

    # Create a bar chart for Age Distribution by Gender
    age_distribution_chart = alt.Chart(filtered_df).mark_bar().encode(
        x='Age:Q',
        y='count()',
        color='Gender:N'
    ).properties(
        title='Age Distribution by Gender',
        width=300
    )

    # Create a bar chart for Salary Distribution by Department
    salary_distribution_chart = alt.Chart(filtered_df).mark_bar().encode(
        x='Department:N',
        y='average(Salary):Q',
        color='Gender:N'
    ).properties(
        title='Salary Distribution by Department',
        width=300
    )

    with st.expander("Detailed Report", expanded=True):  # Set expanded to True to initially show the report
        # Display the charts in a single row
        st.header("Report:")
        st.write(f"Total Salary: {total_salary}")
        st.write(f"Average Work Hours: {average_work_hours}")
        charts_row1, charts_row2 = st.columns(2)

        with charts_row1:
            st.subheader("Bar Chart:")
            st.altair_chart(bar_chart)

        with charts_row1:
            st.subheader("Line Chart:")
            st.altair_chart(line_chart)

        with charts_row2:
            st.subheader("Age Distribution by Gender:")
            st.altair_chart(age_distribution_chart)

        with charts_row2:
            st.subheader("Salary Distribution by Department:")
            st.altair_chart(salary_distribution_chart)

def main():
    if 'show_match_state' not in st.session_state:
        st.session_state.show_match_state = False
    if 'show_recommend_state' not in st.session_state:
        st.session_state.show_recommend_state = False
    if 'show_review_state' not in st.session_state:
        st.session_state.show_review_state = False
    if 'show_movie_state' not in st.session_state:
        st.session_state.show_movie_state = False
    if 'id' not in st.session_state:
        st.session_state.id = False
    if 'id_table' not in st.session_state:
        st.session_state["id_table"] = pd.DataFrame()
    if "all_chunks" not in st.session_state:
        st.session_state["all_chunks"] = pd.DataFrame()
    if "start" not in st.session_state:
        st.session_state["start"] = False

    # if "id_btn" not in st.session_state:
    #         st.session_state['id_btn'] = False
    # Add a dark theme and background effects to the page
    st.markdown(
        """
        <style>
            body {
                background-color: #1e1e1e;  /* Dark background color */
                color: #f8f8f8;  /* Light text color */
                animation: pulse 5s infinite; /* Pulsating effect */
            }
            
            @keyframes pulse {
                0% {
                    background-color: #1e1e1e;
                }
                50% {
                    background-color: #2a2a2a;  /* Slightly lighter color */
                }
                
                100% {
                    background-color: #1e1e1e;
                }
            }

            .css-1aumxhk {
                transition: all 0.5s ease !important;
            }
            .css-12l3icj {
                transition: all 0.5s ease !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    start = st.title("Data Analysis Dashboard")

    if start:
        if st.session_state.start == False:
            # Read the data from the CSV file
            file_path = './dataset/movie_reviews_merged.csv' # Update with your file path ---> REAL
            # file_path = './dataset/emb/emb_file.csv' # Update with your file path ---> REAL
            # file_path = './dataset/emb/test.csv' # Update with your file path
            # file_path = './dataset/cleaned_data/test.csv' # Update with your file path ---> TEST
            # df_emb = pd.read_csv(file_path)

            # Add a Streamlit progress bar
            progress_bar = st.progress(0)

            # Read the CSV file
            df_emb_iter = pd.read_csv(file_path, iterator=True, chunksize=1000)  # Adjust chunksize as needed

            # Initialize an empty DataFrame to concatenate chunks
            all_chunks = pd.DataFrame()

            # Calculate the total number of chunks
            total_chunks = 0
            for _ in df_emb_iter:
                total_chunks += 1

            # Reset the iterator for reading the file again
            df_emb_iter = pd.read_csv(file_path, iterator=True, chunksize=1000)  # Adjust chunksize as needed

            for i, chunk in enumerate(df_emb_iter):
                # Process each chunk as needed
                # ...

                # Concatenate the chunk to the final DataFrame
                all_chunks = pd.concat([all_chunks, chunk])

                # Update the progress bar
                progress_bar.progress((i + 1) / total_chunks)

            # Close the progress bar once reading is complete
            st.balloons()  # Optional: Display celebratory balloons to signal completion
            progress_bar.empty()
            st.session_state.all_chunks = all_chunks
            st.dataframe(all_chunks.head())
            st.session_state.start = True

    

    # Use st.sidebar for input controls
    with st.sidebar:
        # st.subheader("Filter Options")
        # options_department = df['Department'].unique().tolist()
        # input_department = st.selectbox("Select Department", options_department)

        # options_gender = ["Male", "Female"]
        # input_gender = st.selectbox("Select Gender", options_gender)

        # date_inputs_col = st.columns(2)
        # with date_inputs_col[0]:
        #     input_start_date = st.date_input("Select Start Date", datetime.date.today())

        # with date_inputs_col[1]:
        #     input_end_date = st.date_input("Select End Date", datetime.date.today())

        # def callback():
        #     st.session_state.show_match = True

        def callback():
            st.session_state.show_match_state = True
            st.session_state.show_movie_state = False
            st.session_state.show_review_state = False
        def callback2():
            st.session_state.show_movie_state = True
            st.session_state.show_match_state = False
            st.session_state.show_review_state = False

        def callback3():
            st.session_state.show_movie_state = False
            st.session_state.show_match_state = False
            st.session_state.show_review_state = True

        def callback4():
            st.session_state.show_movie_state = False
            st.session_state.show_match_state = False
            st.session_state.show_review_state = False
            st.session_state.show_recommend_state = True


        # Create a text input box
        st.subheader("Getting Movie Review Sentiment")
        review_movie_text = st.text_input("Enter Movie Title:")
        # st.subheader("Write a Review ")
        review_text = st.text_input("Enter Review:")
        # show_report = st.button("Generate Report")
        show_review = st.button("Submit",on_click=callback3)

        # Create a text input box
        st.header("Aspect Based Sentiment ")
        query_text = st.text_input("Enter Query:")

        st.subheader("Start Date")
        things = st.session_state['all_chunks']
        sdate = things['movie_year'].unique().tolist()

        
        # Sort the list in ascending order
        sdate.sort()
        start_date = st.selectbox("Select Start Date", sdate, key="start_date")

        st.subheader("End Date")
        things = st.session_state['all_chunks']
        edate = things['movie_year'].unique().tolist()

        # Sort the list in ascending order
        edate.sort()
        end_date = st.selectbox("Select End Date", edate, key="end_date")

        
        # show_report = st.button("Generate Report")
        show_match = st.button("Find Matching Reviews",on_click=callback)
      
        



        # Create a text input box
        st.subheader("Write a Movie Title")
        query_movie = st.text_input("Query by Movie Title")

        st.subheader("Start Date")
        things = st.session_state['all_chunks']
        sdate_movie = things['movie_year'].unique().tolist()

        
        # Sort the list in ascending order
        sdate_movie.sort()
        start_date_movie = st.selectbox("Select Start Date", sdate_movie, key="start_date_movie")

        st.subheader("End Date")
        things = st.session_state['all_chunks']
        edate_movie = things['movie_year'].unique().tolist()

        # Sort the list in ascending order
        edate_movie.sort()
        end_date_movie = st.selectbox("Select End Date", edate_movie, key="end_date_movie")


        show_movie = st.button("Sumbit",on_click=callback2)

        # A Nightmare on Elm Street
        st.header("Get a Movie Recommendation")
        movie_rec = st.text_input("Enter Movie Title: ")

        show_recommendations = st.button("Movie Recommendations",on_click=callback4)

    # # Display selected options
    # st.write("### Selected Options:")
    # st.write(f"**Department:** {input_department}")
    # st.write(f"**Gender:** {input_gender}")
    # st.write(f"**Start Date:** {input_start_date}")
    # st.write(f"**End Date:** {input_end_date}")

    

# ----------->>>>>>
    if st.session_state.show_recommend_state:
        # input_review = review_text.strip()


        result_df_2_recommend,result_type = get_movie_recommendation(movie_rec.strip())
        if result_type ==1:
            movie_info = result_df_2_recommend[result_df_2_recommend['title'] == movie_rec.strip()]
            movie_list = result_df_2_recommend[result_df_2_recommend['title'] != movie_rec.strip()]
            st.dataframe(movie_info)
            st.dataframe(movie_list)
        else:
            st.write(f"'{movie_rec}' doesn't exist in the records.")
            st.write("You may want to try the movies which are the closest match to the input.")
            st.dataframe(result_df_2_recommend)

    # Filter and display matching reviews
    # if show_match:
    if st.session_state.show_review_state:
        st.session_state.show_movie_state = False
        st.session_state.show_match_state = False

        input_review = review_text.strip()
        result_df_2_review = process_input_text(st.session_state.all_chunks ,review_movie_text,review_text)
        
        
        # Toy Story/I love this movie!

        st.write(f"**Review:** {review_text}")

        # ----------->>>>>> START OF SENTIMEN
        
        # Conduct data preprocessing

        result_df_2_review["Review"] = result_df_2_review["Review"].apply(lambda x : conduct_text_preprocessing(text=x, set_n=1))
        result_df_2_review["Review"] = remove_ner(result_df_2_review["Review"])

        result_df_2_review["Review"] = result_df_2_review["Review"].apply(lambda x : conduct_text_preprocessing(text=x, set_n=2))
        # result_df_2_review["Review"] = tp.lemmatize_text(result_df_2_review["Review"])
        result_df_2_review["Review"] = tp.lemmatize_text(result_df_2_review["Review"], nlp)


        # # Generate review sentiment
        result_df_2_review['sentiment'] = generate_sentiment(result_df_2_review["Review"])
        st.dataframe(result_df_2_review[["Review","sentiment"]].head(1))
        # st.write(f"**End Date:** {input_end_date}")   

    # ----------->>>>>> END OF SENTIMENT

    if st.session_state.show_match_state:
        st.session_state.show_movie_state = False
        # st.session_state['all_chunks'] = all_chunks

        # --------->>>>>>>> UPLOAD EMBEDDINGS TO CSV

        # progress_bar = st.progress(0)  # Initialize progress bar

        # # Define a callback function to update the progress bar
        # def update_progress(progress):
        #     progress_bar.progress(progress)

        # # Call filter_reviews() with progress tracking
        # result = filter_reviews(progress_callback=update_progress)

        # # Remove the progress bar once the process is complete
        # progress_bar.empty()

        # if result["res"] == "No Matching Reviews":
        #     st.warning("No matching reviews found.")
        # else:
        #     st.success("Matching reviews processed successfully.")
        
        # --------->>>>>>>> UPLOAD EMBEDDINGS TO CSV
        # filter_reviews()
        # result_df = filter_reviews()
        # st.dataframe(result_df)
        # st.subheader(f"Reviews around: {input_review}")
        # st.session_state['show_match'] = True

        
        input_review = query_text.strip()
        result_df_2 = compare_with_existing_embeddings(st.session_state.all_chunks ,input_review,start_date,end_date)
        st.dataframe(result_df_2)
        # Add a download button for top 10 positive reviews
        csv_export_button_pos = st.download_button(
            label=f"All reviews relating to {input_review}",
            data=result_df_2.to_csv().encode(),
            file_name=f'{input_review}_reviews.csv',
            mime='text/csv'
        )


# ----------->>>>>>

        if not result_df_2.empty:
        # Display a bar chart based on sentiment counts
            chart_data = pd.DataFrame(result_df_2['sentiment'].value_counts()).reset_index()
            chart_data.columns = ['Sentiment', 'Count']

            st.subheader("Sentiment Distribution")
            c = alt.Chart(chart_data).mark_bar().encode(
                x='Sentiment',
                y='Count',
                color='Sentiment'
            ).properties(width=800, height=600)

            st.altair_chart(c)

    # # # ----
            # Filter and display top 10 positive reviews
            result_df_tp_10_pos = filter_top10_pos(result_df_2)
            st.subheader(f"Top 10 (Positive) Reviews around: {input_review}")
            st.dataframe(result_df_tp_10_pos)

            

            # Filter and display top 10 negative reviews
            result_df_tp_10_neg = filter_top10_neg(result_df_2)
            st.subheader(f"Top 10 (Negative) Reviews around: {input_review}")
            st.dataframe(result_df_tp_10_neg)

            # Filter and display top 10 negative reviews
            result_df_tp_10_neu = filter_top10_neu(result_df_2)
            st.subheader(f"Top 10 (Neutral) Reviews around: {input_review}")
            st.dataframe(result_df_tp_10_neu)
        else:
            st.dataframe(pd.DataFrame())
        
#         # Add a download button for top 10 negative reviews
#         csv_export_button_neg = st.download_button(
#             label="Download Top 10 (Negative) CSV",
#             data=result_df_tp_10_neg.to_csv().encode(),
#             file_name='top10_negative_reviews.csv',
#             mime='text/csv'
#         )

        # result_df_3 = read_all("./dataset/emb_file.csv")
        # st.dataframe(result_df_3)

        # st.title("Movie Information Viewer")

        # Input field for IMDb ID
        

        # st.subheader(f"Movie Details")
        # result_df_4 = filter_titles("./dataset/cleaned_data/movie_metadata.csv", ["Paris, Texas ","Red Dawn","Thor","Star Wars"])
        # st.dataframe(result_df_4)
        
        # st.subheader(f"Common Cast Members")
        # result_df_5 = filter_cast("./dataset/cleaned_data/movie_metadata.csv", ["Thor","Star Wars: Episode III - Revenge of the Sith"])
        # st.dataframe(result_df_5)

        # st.subheader(f"Common Cast Members")
        # result_df_6 = filter_titles_by_id("./dataset/cleaned_data/movie_metadata.csv", "800369")
        # st.dataframe(result_df_6)


        #     # Input field for IMDb ID
        # imdb_id_input = st.text_input("Enter IMDb ID:", key="imdb_id_input")


        # id_btn = st.button("Filter by IMDb ID")
        
        # if id_btn:
        #     st.session_state["id"] = True
        # # Check if IMDb ID is provided
        #     if imdb_id_input:
                
                
        #         # Filter titles based on the entered IMDb ID
        #         result_df = filter_titles_by_id("./dataset/cleaned_data/movie_metadata.csv", imdb_id_input)
        #         st.session_state["id_table"] = result_df
        #         # Display the result
        #         st.subheader("Movie Information:")
        #         st.dataframe(st.session_state["id_table"])
        #         st.session_state.show_match_state = True


        
    if st.session_state.show_movie_state:
        st.session_state.show_match_state = False

        # --------->>>>>>>> UPLOAD EMBEDDINGS TO CSV

        # progress_bar = st.progress(0)  # Initialize progress bar

        # # Define a callback function to update the progress bar
        # def update_progress(progress):
        #     progress_bar.progress(progress)

        # # Call filter_reviews() with progress tracking
        # result = filter_reviews(progress_callback=update_progress)

        # # Remove the progress bar once the process is complete
        # progress_bar.empty()

        # if result["res"] == "No Matching Reviews":
        #     st.warning("No matching reviews found.")
        # else:
        #     st.success("Matching reviews processed successfully.")
        
        # --------->>>>>>>> UPLOAD EMBEDDINGS TO CSV
        # filter_reviews()
        # result_df = filter_reviews()
        # st.dataframe(result_df)
        # st.subheader(f"Reviews around: {input_review}")
        # st.session_state['show_match'] = True

        
        input_review_movie = query_movie.strip()
        result_df_movie = filter_by_movie_title(st.session_state.all_chunks ,input_review_movie,start_date_movie,end_date_movie)
        st.dataframe(result_df_movie)
        # Add a download button for top 10 positive reviews
        csv_export_button_pos = st.download_button(
            label=f"All reviews relating to {input_review_movie}",
            data=result_df_movie.to_csv().encode(),
            file_name=f'{input_review_movie}_reviews.csv',
            mime='text/csv'
        )


# ----------->>>>>>

        # Display a bar chart based on sentiment counts
        chart_data = pd.DataFrame(result_df_movie['viwers_sentiment'].value_counts()).reset_index()
        chart_data.columns = ['Sentiment', 'Count']

        st.subheader("Sentiment Distribution")
        c = alt.Chart(chart_data).mark_bar().encode(
            x='Sentiment',
            y='Count',
            color='Sentiment'
        ).properties(width=800, height=600)

        st.altair_chart(c)

# # # # ----
        # Filter and display top 10 positive reviews
        result_df_tp_10movie_pos = filter_top10movie_pos(result_df_movie)
        st.subheader(f"Top 10 (Positive) Reviews around: {input_review_movie}")
        st.dataframe(result_df_tp_10movie_pos)

        

        # Filter and display top 10 negative reviews
        result_df_tp_10movie_neg = filter_top10movie_neg(result_df_movie)
        st.subheader(f"Top 10 (Negative) Reviews around: {input_review_movie}")
        st.dataframe(result_df_tp_10movie_neg)

        # Filter and display top 10 negative reviews
        result_df_tp_10movie_neu = filter_top10movie_neu(result_df_movie)
        st.subheader(f"Top 10 (Neutral) Reviews around: {input_review_movie}")
        st.dataframe(result_df_tp_10movie_neu)
# --------
# if st.session_state.show_recommend_state:
#         st.session_state.show_match_state = False
        
#         # Add a download button for top 10 negative reviews
#         csv_export_button_neg = st.download_button(
#             label="Download Top 10 (Negative) CSV",
#             data=result_df_tp_10_neg.to_csv().encode(),
#             file_name='top10_negative_reviews.csv',
#             mime='text/csv'
#         )

        # result_df_3 = read_all("./dataset/emb_file.csv")
        # st.dataframe(result_df_3)

        # st.title("Movie Information Viewer")

        # Input field for IMDb ID
        

        # st.subheader(f"Movie Details")
        # result_df_4 = filter_titles("./dataset/cleaned_data/movie_metadata.csv", ["Paris, Texas ","Red Dawn","Thor","Star Wars"])
        # st.dataframe(result_df_4)
        
        # st.subheader(f"Common Cast Members")
        # result_df_5 = filter_cast("./dataset/cleaned_data/movie_metadata.csv", ["Thor","Star Wars: Episode III - Revenge of the Sith"])
        # st.dataframe(result_df_5)

        # st.subheader(f"Common Cast Members")
        # result_df_6 = filter_titles_by_id("./dataset/cleaned_data/movie_metadata.csv", "800369")
        # st.dataframe(result_df_6)


        #     # Input field for IMDb ID
        # imdb_id_input = st.text_input("Enter IMDb ID:", key="imdb_id_input")


        # id_btn = st.button("Filter by IMDb ID")
        
        # if id_btn:
        #     st.session_state["id"] = True
        # # Check if IMDb ID is provided
        #     if imdb_id_input:
                
                
        #         # Filter titles based on the entered IMDb ID
        #         result_df = filter_titles_by_id("./dataset/cleaned_data/movie_metadata.csv", imdb_id_input)
        #         st.session_state["id_table"] = result_df
        #         # Display the result
        #         st.subheader("Movie Information:")
        #         st.dataframe(st.session_state["id_table"])
        #         st.session_state.show_match_state = True


        


    # if show_report:
    #     # Convert input_joining_date to Pandas Timestamp
    #     input_start_date = pd.to_datetime(input_start_date)
    #     input_end_date = pd.to_datetime(input_end_date)

    #     # Filter DataFrame based on selected values
    #     filtered_df = df[
    #         (df['Department'] == input_department) &
    #         (pd.to_datetime(df['Joining_Date']) > pd.to_datetime(input_start_date)) &
    #         (pd.to_datetime(df['Joining_Date']) < pd.to_datetime(input_end_date)) &
    #         (df['Gender'] == input_gender)
    #     ]
    #     with st.expander("Filtered Records", expanded=True):
    #         # Display filtered records and generate graphical report
    #         st.header("Filtered Records:")
    #         if filtered_df.empty:
    #             st.warning("No records found with the selected values.")
    #         else:
    #             st.dataframe(filtered_df)

    #     # Generate and display the report
    #     if not filtered_df.empty:
            # generate_report(filtered_df)

if __name__ == "__main__":
    main()
