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
    
from spellchecker import SpellChecker
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
    pattern = r'[.,?!\'"():;\[\]{}@#^$%&*_+=<>\\|`~]|-(?!\w)|(?<!\w)-'
    return re.sub(pattern, ' ', text)


def replace_whitespace(text):
    pattern = r'\s+'
    return re.sub(pattern, ' ', text)


def replace_nonascii_characters(text):
    # List of all non-ascii characters that needs to be replaced
    text = re.sub('[İı]', 'I', text)
    return text

    
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


def lemmatize_text(texts):
    # Load the spaCy language model
    # See: https://spacy.io/usage/models
    nlp = spacy.load("en_core_web_sm")

    list_of_lemmatized_texts = []
    for doc in nlp.pipe(texts, n_process=2, batch_size=2000, disable=['parser']):
        
        lemmatized_texts = []
        for token in doc:
            try:
                if token.ent_type_:
                    lemmatize_text.append(token.text)
                else:
                    if token.lemma_ not in nlp.Defaults.stop_words and token.lemma_.isalpha():
                        lemmatized_texts.append(token.lemma_)
            except Exception as err:
                print(f"ERROR: {err}")
                print(f"Text: {token.lemma_}")

        list_of_lemmatized_texts.append(" ".join(lemmatized_texts))

    return list_of_lemmatized_texts

def spell_check_text(text):
    # Load the spaCy language model
    nlp = spacy.load("en_core_web_sm")

    # Initialize the spell checker
    spell = SpellChecker()

    list_of_spell_corrected_text = []
    for doc in nlp.pipe(text, n_process=2, batch_size=2000, disable=['parser']):
        spell_corrected_text = []

        for token in doc:
            try:
                if token.ent_type_:
                    # If token is a Named Entity, keep the original token
                    spell_corrected_text.append(token.text)
                else:
                    corrected_token = spell.correction(token.text)
                    if corrected_token is not None and corrected_token != token.text:
                        spell_corrected_text.append(corrected_token)
                    else:
                        spell_corrected_text.append(token.text)  # Keep original token if spelling is correct
            except Exception as err:
                print(f"ERROR: {err}")
                print(f"Text: {token.text}")

        list_of_spell_corrected_text.append(" ".join(spell_corrected_text))

    return list_of_spell_corrected_text
    
