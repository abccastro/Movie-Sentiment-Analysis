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

from nltk.tokenize import word_tokenize
from emot.emo_unicode import EMOTICONS_EMO
from flashtext import KeywordProcessor
from bs4 import BeautifulSoup


def remove_email_address(text):
    pattern = r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,7}\b'    
    return re.sub(pattern, '', text)


def remove_hyperlink(text):
    pattern = r'https?://(?:www\.)?[\w\.-]+(?:\.[a-z]{2,})+(?:/[-\w\.,/]*)*(?:\?[\w\%&=]*)?'
    return re.sub(pattern, '', text)


def remove_non_alphanumeric_char(text):
    # remove non-alpha numeric characters except for the following
    # > hyphen (-) that is in between alphanumeric
    # > dollar sign ($) for currency
    # > percent sign (%) for percentage
    pattern = r'[.,?!\'"():;\[\]{}@#^&*_+=<>\\|`~]|-(?!\w)|(?<!\w)-'
    return re.sub(pattern, ' ', text)


def replace_whitespace(text):
    pattern = r'\s+'
    return re.sub(pattern, ' ', text)


def replace_nonascii_characters(text):
    # List of all non-ascii characters that needs to be replaced
    text = re.sub('[İı]', 'I', text)
    return text


def remove_stopwords(text, list_of_stopwords):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words from the tokenized text
    filtered_tokens = [token for token in tokens if token not in list_of_stopwords and token.isalpha()]
    # Join the non-stopwords back into a string
    filtered_text = " ".join(filtered_tokens)
    return filtered_text


def get_emojis():
    emoji_dict = KeywordProcessor()
    for k,v in EMOTICONS_EMO.items():
        words = re.split(r',| or ', v.lower())
        # get only the first word. remove "smiley" word.
        word = words[0].replace(" smiley", "")
        emoji_dict.add_keyword(k, words[0])

    # additional emojis
    emoji_dict.add_keyword("<3", "heart")
    emoji_dict.add_keyword("</3", "heartbroken")

    return emoji_dict


def webscrape_slang_words():
    http = urllib3.PoolManager()
    slang_word_dict = KeywordProcessor()

    try:
        # TODO: need to save the content in a file
        for i in range(97,123):
            page = http.request('GET', 'https://www.noslang.com/dictionary/'+chr(i))
            soup = BeautifulSoup(page.data, 'html.parser')

            for elem in soup.findAll('div', class_="dictonary-word"): 
                slang_word = elem.find('abbr').get_text()

                key = slang_word[0 : slang_word.rfind(":")-1]
                value = elem.find('dd', class_="dictonary-replacement").get_text()
                slang_word_dict.add_keyword(key.lower(), value.lower())
    except Exception as err:
        print(f"ERROR: {err}")
    
    return slang_word_dict


def lemmatize_text(text, nlp):
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text