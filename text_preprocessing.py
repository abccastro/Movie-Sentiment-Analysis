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

import spacy
import re
import urllib3
import utils

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from emot.emo_unicode import EMOTICONS_EMO
from flashtext import KeywordProcessor
from bs4 import BeautifulSoup
from unidecode import unidecode


def remove_email_address(text):
    pattern = r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,7}\b'    
    return re.sub(pattern, '', text)


def remove_hyperlink(text):
    pattern = r'https?://(?:www\.)?[\w\.-]+(?:\.[a-z]{2,})+(?:/[-\w\.,/]*)*(?:\?[\w\%&=]*)?'
    return re.sub(pattern, '', text)


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
    filename = './dictionary/slang_words_dictionary.pkl'
    
    slang_word_dict = KeywordProcessor()
    try:
        if webscraped:
            # build slang words dictionary by webscraping
            slang_word_dict = webscrape_slang_words()
            # update the existing pickle file
            utils.save_pickle_file(slang_word_dict, filename)
        else:
            # open a pickle file
            slang_word_dict = utils.open_pickle_file(filename)           
    except Exception as err:
        print(f"ERROR: {err}")
        print(f"Importing slang words dictionary from file..")
        # open a pickle file
        slang_word_dict = utils.open_pickle_file(filename)
    
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
    # remove stop words from the tokenized text
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in list_of_stopwords and "-" not in token and token.isalpha()]
    filtered_text = " ".join(filtered_tokens)

    return filtered_text


def lemmatize_text(texts):
    # Load the spaCy language model
    # See: https://spacy.io/usage/models
    nlp = spacy.load("en_core_web_sm")
    
    list_of_lemmatized_texts = []
    # customize spacy pipeline to apply two processors to a batch of 2000 records, and to exclude 'ner', 'parser' and 'textcat'
    for doc in nlp.pipe(texts, n_process=2, batch_size=2000, disable=['ner', 'parser', 'textcat']):
        lemmatized_texts = []
        for token in doc:
            try:
                if token.lemma_ not in nlp.Defaults.stop_words and token.lemma_.isalpha():
                    lemmatized_texts.append(token.lemma_)
            except Exception as err:
                print(f"ERROR: {err}")
                print(f"Text: {token.lemma_}")

        list_of_lemmatized_texts.append(" ".join(lemmatized_texts))

    return list_of_lemmatized_texts