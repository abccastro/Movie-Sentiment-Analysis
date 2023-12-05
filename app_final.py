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
import nltk
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer,util
from enum import Enum


class Sentiment(Enum):
    Negative = "0"
    Neutral = "1"
    Positive = "2"


# Language models
nltk.download('stopwords')
list_of_stopwords = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

# initializer dictionaries for data preprocessing
emoji_dict = tp.get_emojis()
slang_word_dict = tp.get_slang_words(webscraped=False)


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
                'Review': review_text,
                'cosine_similarity': cos_sim,
                'sentiment': row['review_sentiment'],
                'titles': row['title'],
                'date': row['movie_year'],
            }
            matching_reviews.append(metadata)

    # Create a DataFrame with the matching reviews and sort it
    df_result = pd.DataFrame(matching_reviews)
    
    if not df_result.empty:
        df_result = df_result.sort_values(by='cosine_similarity', ascending=False)
        return df_result
    else:
        # Handle the case where the DataFrame is empty
        return pd.DataFrame()


def filter_by_movie_title(df, input_title,sdate="",edate=""):
    #  Check if the 'title' column exists in the DataFrame
    if 'title' not in df.columns:
        return pd.DataFrame()

    # Lowercase and strip whitespace from the 'title' column and input title
    df['movie_title_lower'] = df['title'].str.lower().str.strip()
    input_title_lower = input_title.lower().strip()

    # Filter the DataFrame based on the lowercase and stripped input movie title
    filtered_df = df[df['movie_title_lower'] == input_title_lower]

    # # Check if both start date and end date are provided
    # if sdate and edate:
    #     # Convert dates to numbers (assuming they are strings)
    #     sdate = int(sdate)
    #     edate = int(edate)

    #     # Ensure start date is not higher than end date
    #     if sdate > edate:
    #         st.warning("Start date should not be higher than end date.")
    #         return pd.DataFrame()
        
    #     filtered_df['release_year'] = pd.to_numeric(filtered_df['release_year'], errors='coerce')

    #     # Filter the DataFrame based on the date range
    #     filtered_df = filtered_df[(filtered_df['release_year'] >= sdate) & (filtered_df['release_year'] <= edate)]

    # Drop the temporary lowercase column
    filtered_df = filtered_df.drop(columns=['movie_title_lower'])

    return filtered_df


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
                'movie_sentiment': row['movie_sentiment']
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
                'movie_sentiment': row['movie_sentiment']
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


def filter_top10_reviews(df_result, sentiment_type):

    if sentiment_type == Sentiment.Negative.value:
        sentiments = ['negative']
    elif sentiment_type == Sentiment.Neutral.value:
        sentiments = ['neutral']
    elif sentiment_type == Sentiment.Positive.value:
        sentiments = ['positive']

    # # Check if the Drame is not empty and contains the required columns
    if not df_result.empty and 'sentiment' in df_result.columns and 'cosine_similarity' in df_result.columns:
    # Lowercase theataF 'sentiment' column and create a temporary lowercase column
        df_result['sentiment_lower'] = df_result['sentiment'].str.lower()

        # Filter the DataFrame based on the lowercase 'sentiment' column
        df_result = df_result[df_result['sentiment_lower'].isin(sentiments)]

        # Drop the temporary lowercase column
        df_result = df_result.drop(columns=['sentiment_lower'])

        # Sort the DataFrame by Cosine Similarity in descending order
        df_result = df_result.sort_values(by='cosine_similarity', ascending=False).head(10)

        return df_result
    else:
        # Handle the case where the DataFrame is empty or missing required columns
        return pd.DataFrame()


def filter_top10movie_reviews(df_result, sentiment_type):

    if sentiment_type == Sentiment.Negative.value:
        sentiments = ['negative']
    elif sentiment_type == Sentiment.Neutral.value:
        sentiments = ['neutral']
    elif sentiment_type == Sentiment.Positive.value:
        sentiments = ['positive']

    if not df_result.empty and 'review_sentiment' in df_result.columns:
    # Lowercase the 'sentiment' column and create a temporary lowercase column
        df_result['sentiment_lower'] = df_result['review_sentiment'].str.lower()

        # Filter the DataFrame based on the lowercase 'sentiment' column
        df_result = df_result[df_result['sentiment_lower'].isin(sentiments)]

        # Drop the temporary lowercase column
        df_result = df_result.drop(columns=['sentiment_lower'])

        # Sort the DataFrame by Cosine Similarity in descending order
        df_result = df_result.head(10)

        return df_result
    else:
        # Handle the case where the DataFrame is empty or missing required columns
        return pd.DataFrame()


def process_movie_review(df, input_movie_text, review_text):
    # Filter the DataFrame based on the title
    filtered_df = df[df['title'].str.lower().str.strip() == input_movie_text.lower()]

    new_df = pd.DataFrame()

    if len(filtered_df) > 1:
        # Create a new DataFrame to store the modified data
        new_df = filtered_df.head(1)
        new_df['Review'] = review_text

        # Conduct data preprocessing
        new_df["Cleaned_Review"] = new_df["Review"].apply(lambda x : conduct_text_preprocessing(text=x, set_n=1))
        new_df["Cleaned_Review"] = remove_ner(new_df["Cleaned_Review"])
        new_df["Cleaned_Review"] = new_df["Cleaned_Review"].apply(lambda x : conduct_text_preprocessing(text=x, set_n=2))
        # result_df_2_review["Review"] = tp.lemmatize_text(result_df_2_review["Review"])
        new_df["Cleaned_Review"] = tp.lemmatize_text(new_df["Cleaned_Review"], nlp)

        # Generate review sentiment
        new_df['sentiment'] = generate_sentiment(new_df["Cleaned_Review"])
        
    return new_df
       
       
def conduct_text_preprocessing(text, set_n=1):
    '''
    This function contains processes for data cleaning
    '''
    try:
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

    except Exception as err:
        print(f"ERROR: {err}")
        print(f"Input Text: {text}")

    return text


def remove_ner(df):
    '''
    This function removes Named Entity Recognition (NER)s
    '''
    # Dummy dataframe for extracted NERs
    column_names = ['PERSON', 'WORK_OF_ART', 'LOCATION', 'DATE_TIME', 'ORGANIZATION', 'PRODUCT', 'EVENT', 'LANGUAGE']
    movie_name_entities_df = pd.DataFrame(columns=column_names)

    # Extract and remove name entities
    df['Cleaned_Review'], movie_name_entities_df = tp.extract_name_entity(df, nlp, movie_name_entities_df)
    return df


def generate_sentiment(df):
    '''
    This function generates sentiments to the reviews using Naive Bayes model
    '''
    sentiment_dict = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predictions = []

    try:
        # open pickle file for countvectorizer
        vectorizer = utils.open_pickle_file('sentiment_analysis_vectorizer.pkl')
        # open pickle file for naive bayes model
        model = utils.open_pickle_file('sentiment_analysis_model.pkl')
        
        # predict the sentiment of the review
        vectorized_text = vectorizer.transform(df)
        predictions = model.predict(vectorized_text)  
        predictions  = sentiment_dict.get(predictions[0])

    except Exception as err:
        print(f"ERROR: {err}")

    return predictions


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
    try:
        movie_recommender_metadata = utils.open_dataset_file('movie_metadata.csv')
        vg_indices = utils.open_pickle_file('movie_recommender_nn_indices.pkl')
        vg_distances = utils.open_pickle_file('movie_recommender_nn_distances.pkl')

        # query movie title
        movie_idx = movie_recommender_metadata.query("title == @movie_name").index  
        
        if movie_idx.empty:
            # If the movie entered by the user doesn't exist in the records, the program will recommend a new movie similar to the input
            top_closest_match_name = get_top_closest_movie_names(movie_name=movie_name, df=movie_recommender_metadata['title'], top_n=top_n)
            return pd.DataFrame(top_closest_match_name, columns=["Movie Title"]),0
        
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
            
            # Get the first 10 games in the list
            recommended_movie_list = movie_list.head(top_n)
            recommended_movie_list = recommended_movie_list.drop(columns=['imdb_id'], axis=1)

            recommended_movie_list = recommended_movie_list.reset_index(drop=True)

    except Exception as err:
        print(f"ERROR: {err}")

    return recommended_movie_list,1


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
            # file_path = './dataset/cleaned_data/test.csv' # Update with your file path ---> TEST
            
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
                # Concatenate the chunk to the final DataFrame
                all_chunks = pd.concat([all_chunks, chunk])

                # Update the progress bar
                progress_bar.progress((i + 1) / total_chunks)

            # Close the progress bar once reading is complete
            st.balloons()  # Optional: Display celebratory balloons to signal completion
            progress_bar.empty()
            st.session_state.all_chunks = all_chunks
            st.dataframe(all_chunks.head(), hide_index=True)
            st.session_state.start = True


    # Use st.sidebar for input controls
    with st.sidebar:

        def callback():
            st.session_state.show_movie_state = False
            st.session_state.show_match_state = True
            st.session_state.show_review_state = False
            st.session_state.show_recommend_state = False

        def callback2():
            st.session_state.show_movie_state = True
            st.session_state.show_match_state = False
            st.session_state.show_review_state = False
            st.session_state.show_recommend_state = False

        def callback3():
            st.session_state.show_movie_state = False
            st.session_state.show_match_state = False
            st.session_state.show_review_state = True
            st.session_state.show_recommend_state = False

        def callback4():
            st.session_state.show_movie_state = False
            st.session_state.show_match_state = False
            st.session_state.show_review_state = False
            st.session_state.show_recommend_state = True

        ########################################################
        #  SIDE PANEL: CONTAINING INPUTS
        ########################################################
        
        things = st.session_state['all_chunks']

        ########## MOVIE REVIEW SENTIMENT ##########
        title_movie = things['title'].unique().tolist()
        title_movie.sort()
        
        st.header("Get Movie Review Sentiment")
        review_movie_text = st.selectbox("Select Movie Title", title_movie, key="review_movie_text")
        review_text = st.text_input("Enter Review:")

        show_review = st.button("Find Sentiment", on_click=callback3)

        ########## MOVIE ASPECTED-BASED SENTIMENT ##########
        sdate = things['movie_year'].unique().tolist()
        sdate.sort()
        edate = things['movie_year'].unique().tolist()
        edate.sort()

        st.header("Movie Aspect-Based Sentiment")
        query_text = st.text_input("Enter Query:")
        start_date = st.selectbox("Select Start Date", sdate, key="start_date")
        end_date = st.selectbox("Select End Date", edate, key="end_date")

        show_match = st.button("Find Reviews", on_click=callback, key="show_match")

        ########## MOVIE REVIEWS ##########
        st.header("Get Movie Reviews")
        query_movie = st.selectbox("Select Movie Title", title_movie, key="query_movie")
        # start_date_movie = st.selectbox("Select Start Date", sdate, key="start_date_movie")
        # end_date_movie = st.selectbox("Select End Date", edate, key="end_date_movie")

        show_movie = st.button("Find Reviews", on_click=callback2, key="show_movie")

        ########## MOVIE RECOMMENDATION ##########
        st.header("Get Movie Recommendation")
        movie_rec = st.text_input("Enter Movie Title: ")
        show_recommendations = st.button("Find Recommendations", on_click=callback4)


    ########################################################
    #  DISPLAY PANEL: DASHBOARD
    ########################################################

    ########## MOVIE RECOMMENDATION ##########
    if st.session_state.show_recommend_state:
        result_df_2_recommend, result_type = get_movie_recommendation(movie_rec.strip())
        if result_type ==1:
            movie_info = result_df_2_recommend[result_df_2_recommend['title'] == movie_rec.strip()]
            movie_list = result_df_2_recommend[result_df_2_recommend['title'] != movie_rec.strip()]

            st.subheader(f"Movie Information")
            st.dataframe(movie_info, hide_index=True)

            st.subheader(f"Top 10 Recommended Movies for '{movie_info.iloc[0]['title']}'")
            st.dataframe(movie_list, hide_index=True)
        else:
            st.write(f"The movie'{movie_rec}' doesn't exist in the records.")
            st.write("You may want to try the movies which are the closest match to the input.")
            st.dataframe(result_df_2_recommend, hide_index=True)

    ########## MOVIE REVIEW SENTIMENT ##########
    if st.session_state.show_review_state:
        
        input_review = review_text.strip()
        st.write(f"**Review:** {review_text}")

        result_df_2_review = process_movie_review(st.session_state.all_chunks, review_movie_text, review_text)
        
        if not result_df_2_review.empty:    
            st.dataframe(result_df_2_review[["Review","sentiment"]].head(1), hide_index=True)
        else:
            st.write(f"The movie '{review_movie_text}' doesn't exist in the records.")

    ########## MOVIE ASPECTED-BASED SENTIMENT ##########
    if st.session_state.show_match_state:

        input_review = query_text.strip()
        result_df_2 = compare_with_existing_embeddings(st.session_state.all_chunks ,input_review,start_date,end_date)
        st.dataframe(result_df_2, hide_index=True)
        # Add a download button for top 10 positive reviews
        csv_export_button_pos = st.download_button(
            label=f"All reviews relating to {input_review}",
            data=result_df_2.to_csv().encode(),
            file_name=f'{input_review}_reviews.csv',
            mime='text/csv'
        )

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

            # Filter and display top 10 positive reviews
            result_df_tp_10_pos = filter_top10_reviews(result_df_2, Sentiment.Positive.value)
            st.subheader(f"Top 10 (Positive) Reviews around: {input_review}")
            st.dataframe(result_df_tp_10_pos)

            # Filter and display top 10 negative reviews
            result_df_tp_10_neg = filter_top10_reviews(result_df_2, Sentiment.Negative.value)
            st.subheader(f"Top 10 (Negative) Reviews around: {input_review}")
            st.dataframe(result_df_tp_10_neg)

            # Filter and display top 10 negative reviews
            result_df_tp_10_neu = filter_top10_reviews(result_df_2, Sentiment.Neutral.value)
            st.subheader(f"Top 10 (Neutral) Reviews around: {input_review}")
            st.dataframe(result_df_tp_10_neu)

        else:
            st.dataframe(pd.DataFrame())
        
    ########## MOVIE REVIEWS ##########
    if st.session_state.show_movie_state:

        input_review_movie = query_movie.strip()
        result_df_movie = filter_by_movie_title(st.session_state.all_chunks, input_review_movie)
        st.subheader(f"Movie Reviews on '{input_review_movie}'")
        st.dataframe(result_df_movie[['Review', 'date', 'review_sentiment']], hide_index=True)
        # Add a download button for top 10 positive reviews
        csv_export_button_pos = st.download_button(
            label=f"All reviews relating to {input_review_movie}",
            data=result_df_movie.to_csv().encode(),
            file_name=f'{input_review_movie}_reviews.csv',
            mime='text/csv'
        )

        # Display a bar chart based on sentiment counts
        chart_data = pd.DataFrame(result_df_movie['review_sentiment'].value_counts()).reset_index()
        chart_data.columns = ['Sentiment', 'Count']

        st.subheader("Sentiment Distribution")
        c = alt.Chart(chart_data).mark_bar().encode(
            x='Sentiment',
            y='Count',
            color='Sentiment'
        ).properties(width=800, height=600)

        st.altair_chart(c)

        # Filter and display top 10 positive reviews
        result_df_tp_10movie_pos = filter_top10movie_reviews(result_df_movie, Sentiment.Positive.value)
        st.subheader(f"Top 10 (Positive) Reviews")
        st.dataframe(result_df_tp_10movie_pos[['Review', 'date']], hide_index=True)

        # Filter and display top 10 negative reviews
        result_df_tp_10movie_neg = filter_top10movie_reviews(result_df_movie, Sentiment.Negative.value)
        st.subheader(f"Top 10 (Negative) Reviews")
        st.dataframe(result_df_tp_10movie_neg[['Review', 'date']], hide_index=True)

        # Filter and display top 10 negative reviews
        result_df_tp_10movie_neu = filter_top10movie_reviews(result_df_movie, Sentiment.Neutral.value)
        st.subheader(f"Top 10 (Neutral) Reviews")
        st.dataframe(result_df_tp_10movie_neu[['Review', 'date']], hide_index=True)


if __name__ == "__main__":
    main()
