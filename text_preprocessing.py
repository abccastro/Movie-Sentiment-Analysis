"""
##########################################################################################################################

Text Preprocessing Methods

This Python file contains a collection of text preprocessing methods to clean and prepare text data for natural language 
processing (NLP) tasks. These methods include functions for tasks such as removal of non-grammatical text, lowercasing, 
tokenization, stopword removal, and etc.

Usage:
- Import this file in your Python script.
- Call the desired preprocessing functions with your text data to apply the respective transformation.

Example Usage:
>>> import text_preprocessing as tp
>>> text = "Hello, World! This is an email example@test.com."
>>> clean_text = tp.remove_email_address(text)
>>> print(clean_text)
Output: "Hello World! This is an email "

##########################################################################################################################
"""

import pandas as pd
# import spacy
import re
import urllib3
import utils
import warnings
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk, pos_tag, word_tokenize
import pandas as pd

from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from emot.emo_unicode import EMOTICONS_EMO
from flashtext import KeywordProcessor
from bs4 import BeautifulSoup
from unidecode import unidecode

warnings.filterwarnings("ignore")

def remove_email_address(text):
    pattern = r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,7}\b'    
    return re.sub(pattern, '', text)


def remove_hyperlink(text):
    pattern = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'              
    return re.sub(pattern, '', text)


def replace_html_char(text):
    text = text.replace("&amp;", "and")
    text = text.replace("&lt;", "<")
    return text


def remove_non_alphanumeric_char(text):
    # remove non-alpha numeric characters except for the following
    # > hyphen (-) that is in between alphanumeric
    pattern = r'[.,?!\'"():;\[\]{}@#^$%&*_+=<>\\|`~]|-(?!\w)|(?<!\w)-'
    return re.sub(pattern, ' ', text)


def replace_whitespace(text):
    pattern = r'\s+'
    return re.sub(pattern, ' ', text)


def replace_nonascii_characters(text):
    # List of all non-ascii characters that needs to be replaced
    text = re.sub('[İı]', 'I', text)
    # remove diacritics (accented charactes)
    text = unidecode(text, errors="preserve")
    return text

    
def get_emojis():
    emoji_dict = KeywordProcessor()
    try:
        for k,v in EMOTICONS_EMO.items():
            words = re.split(r',| or ', v.lower())
            # get only the first element. remove "smiley" word.
            word = words[0].replace(" smiley", "")
            k = k.replace("‑", "-")
            # put in a dictionary
            emoji_dict.add_keyword(k, words[0])

        # additional emojis
        emoji_dict.add_keyword("<3", "heart")
        emoji_dict.add_keyword("</3", "heartbroken")
    except Exception as err:
        print(f"ERROR: {err}")

    return emoji_dict


def get_slang_words(webscraped=False):
    # TODO: Put in a config file
    file_path = utils.get_absolute_file_path('slang_words_dictionary.pkl', 'dictionary')
    
    slang_word_dict = KeywordProcessor()
    try:
        if webscraped:
            # build slang words dictionary by webscraping
            slang_word_dict = webscrape_slang_words()
            # update the existing pickle file
            utils.save_pickle_file(slang_word_dict, file_path)
        else:
            # open a pickle file
            slang_word_dict = utils.open_pickle_file(file_path)           
    except Exception as err:
        print(f"ERROR: {err}")
        print(f"Importing slang words dictionary from file..")
        # open a pickle file
        slang_word_dict = utils.open_pickle_file(file_path)
    
    return slang_word_dict


def webscrape_slang_words():
    http = urllib3.PoolManager()
    slang_word_dict = KeywordProcessor()
    try:
        for i in range(97,123):
            # site where the slang words will be scraped
            page = http.request('GET', 'https://www.noslang.com/dictioary/'+chr(i))
            soup = BeautifulSoup(page.data, 'html.parser')

            for elem in soup.findAll('div', class_="dictonary-word"): 
                slang_word = elem.find('abbr').get_text()

                key = slang_word[0 : slang_word.rfind(":")-1]
                value = elem.find('dd', class_="dictonary-replacement").get_text()
                # put in a dictionary
                slang_word_dict.add_keyword(key.lower(), value.lower())
    except Exception as err:
        print(f"ERROR: {err}")
    
    return slang_word_dict


def remove_stopwords(text, list_of_stopwords):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words from the tokenized text
    filtered_tokens = [token for token in tokens if token not in list_of_stopwords and "-" not in token and token.isalpha()]
    # Join the non-stopwords back into a string
    filtered_text = " ".join(filtered_tokens)

    return filtered_text


# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def lemmatize_text(texts):
    lemmatizer = WordNetLemmatizer()
    list_of_lemmatized_texts = []
    try:
        for text in texts:
            tokens = word_tokenize(text)
            lemmatized_texts = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and lemmatizer.lemmatize(token) not in nltk.corpus.stopwords.words('english')]
            list_of_lemmatized_texts.append(" ".join(lemmatized_texts))
    except Exception as err:
        print(f"ERROR: {err}")

    return list_of_lemmatized_texts


# def lemmatize_text(texts, nlp):
#     list_of_lemmatized_texts = []
#     try:
#         for doc in nlp.pipe(texts, n_process=2, batch_size=2000, disable=['ner', 'parser', 'textcat']):
#             lemmatized_texts = []
#             for token in doc:
#                 if token.lemma_ not in nlp.Defaults.stop_words and token.lemma_.isalpha():
#                     lemmatized_texts.append(token.lemma_)
#             list_of_lemmatized_texts.append(" ".join(lemmatized_texts))
#     except Exception as err:
#         print(f"ERROR: {err}")
# # 
#     return list_of_lemmatized_texts


# def get_entity_label(label):
#     if label in ['GPE', 'LOC', 'FAC']:
#         label = 'LOCATION'
#     elif label in ['DATE', 'TIME']:
#         label = 'DATE_TIME'
#     elif label in ['ORG']:
#         label = 'ORGANIZATION'
#     return label


# def extract_name_entity(texts, nlp, name_entities_df):
#     list_of_non_ner = []
#     try:
#         for doc in nlp.pipe(texts, n_process=2, batch_size=2000, disable=['tagger', 'lemmatizer', 'parser', 'textcat']):
#             # Original text
#             doc_text = doc.text_with_ws
#             # Iterate through list of NERs
#             for ent in doc.ents:
#                 entity_text = ent.text
#                 label = get_entity_label(ent.label_)

#                 if label not in ['ORDINAL', 'CARDINAL', 'PERCENT', 'QUANTITY', 'NORP', 'MONEY', 'LAW']:
#                     row_index = len(list_of_non_ner)             
#                     # Check if the row index exists 
#                     if row_index in name_entities_df.index:
#                         # Check if the current value is NaN
#                         if isinstance(name_entities_df.loc[row_index, label], float) and pd.isna(name_entities_df.loc[row_index, label]):
#                             # If NaN, replace it with a new list containing the specified value
#                             name_entities_df.at[row_index, label] = [entity_text]
#                         else:
#                             # If not NaN, append the value to the existing list
#                             name_entities_df.loc[row_index, label].append(entity_text)  
#                     else:
#                         # Add a new row with the specified index and value
#                         name_entities_df = name_entities_df.append(pd.Series({label: [entity_text]}, name=row_index))
#                     # Replace the NER with empty string

#                 if label not in ['NORP', 'MONEY']:
#                     doc_text = doc_text.replace(entity_text, '')

#             # List of text without NER
#             list_of_non_ner.append(doc_text)

#     except Exception as err:
#         print(f"ERROR: {err}")

#     return list_of_non_ner, name_entities_df