import pandas as pd
import numpy as np
import text_preprocessing as tp
import utils
import contractions
import os
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
list_of_stopwords = set(stopwords.words('english'))

# initializer dictionaries for data preprocessing
emoji_dict = tp.get_emojis()
slang_word_dict = tp.get_slang_words(webscraped=False)


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
    df['combined_review'], movie_name_entities_df = tp.extract_name_entity(df, 
                                                                           nlp, 
                                                                           movie_name_entities_df)
    return df


def generate_sentiment(df):
    '''
    This function generates sentiments to the reviews using Naive Bayes model
    '''
    sentiment_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predictions = []

    try:
        # open pickle file for countvectorizer
        vectorizer = utils.open_pickle_file('sentiment_analysis_vectorizer.pkl')
        
        # open pickle file for naive bayes model
        model = utils.open_pickle_file('sentiment_analysis_model.pkl')
        
        # predict the sentiment of the review
        vectorized_text = vectorizer.transform(df)
        predictions = model.predict(vectorized_text)  
        predictions  = pd.DataFrame(predictions).replace(sentiment_dict)

    except Exception as err:
        print(f"ERROR: {err}")

    return predictions


if __name__ == '__main__':
    try:
        # NOTE: This can be changed asking for user input
        movie_reviews_df = utils.open_dataset_file('reviews.csv')
        movie_reviews_df['combined_review'] = movie_reviews_df['review_summary'] + ' ' + movie_reviews_df['review_detail']       

        # Conduct data preprocessing
        print("[START] Data preprocessing")
        print("- Initial text preprocessing")
        movie_reviews_df['combined_review'] = movie_reviews_df['combined_review'].apply(lambda x : conduct_text_preprocessing(text=x, set_n=1))
        print("- Removing NERs")
        movie_reviews_df['combined_review'] = remove_ner(movie_reviews_df['combined_review'])
        print("- Supplemental text preprocessing")
        movie_reviews_df['combined_review'] = movie_reviews_df['combined_review'].apply(lambda x : conduct_text_preprocessing(text=x, set_n=2))
        print("- Applying lemmatization")
        movie_reviews_df['combined_review'] = tp.lemmatize_text(movie_reviews_df['combined_review'], nlp)
        print("[END] Data preprocessing")

        # Generate review sentiment
        print("[START] Generating review sentiment")
        movie_reviews_df['sentiment'] = generate_sentiment(movie_reviews_df['combined_review'])
        print("[END] Generating review sentiment")

        # NOTE: This can be changed like displaying the list in a scrollbar list
        # Save dataFrame to a CSV file
        datetime = utils.get_current_datetime()
        csv_file_path = utils.get_absolute_directory_path() + '/dataset/reviews_with_sentiments_' + datetime + '.csv'
        
        movie_reviews_df = movie_reviews_df.drop(columns=['combined_review'], axis=1)
        movie_reviews_df.to_csv(csv_file_path, index=False)

    except FileNotFoundError:
        print(f"ERROR: File path '{file_path}' not found.")
    except Exception as err:
        print(f"ERROR: {err}")