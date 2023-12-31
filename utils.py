"""
##########################################################################################################################

Utility Methods

This Python file contains a set of utility methods aimed at providing reusable functions for common tasks in a program. 
These functions cover a range of operations including file handling, data manipulation, string formatting, and more.

Usage:
- Import this file in your Python script.
- Call the desired utility functions to perform specific tasks within your program.

Example Usage:
>>> import utils
>>> formatted_string = utils.format_string('Hello, {}!', 'World')
>>> print(formatted_string)
Output: "Hello, World!"

########################################################################################################################
"""

import json
import re
import pandas as pd
import numpy as np
import pickle
import os

from datetime import datetime
from flashtext import KeywordProcessor


def extract_specific_key(list_of_dicts, dict_keys):
    """
    Function that takes a list of dictionaries (list_of_dicts) and a list of desired keys (dict_keys). It uses 
    a list comprehension to create a new list of dictionaries, where each dictionary contains only the desired keys.
    """
    return [{key: d[key] for key in dict_keys} for d in list_of_dicts]


def convert_str_to_dict(text, dict_keys):
    """
    Function that converts text in string data type to list of dictionaries
    """
    list_of_dicts = []
    try:
        # pattern to replace single quote to double quotes for json.loads to process the input text
        pattern = r'(?<=[{,: ])\'|\'(?=[{,:])'

        text = text.replace('"', "'")
        text = re.sub(pattern, '"', text)
        list_of_dicts = json.loads(text)
    except Exception as err:
        print(f"ERROR: {err}")
        print(f"Input text: {text}")

    return extract_specific_key(list_of_dicts, dict_keys)


def save_pickle_file(data_object, filename):
    """
    Function that saves a pickle file (will override if it already exists)
    """
    try:
        with open(filename, 'wb') as file:
            pickle.dump(data_object, file)
    except Exception as err:
        print(f"ERROR: {err}")


def open_pickle_file(filename):
    """
    Function that reads a pickle file
    """
    data_object = None
    try:
        filepath = get_absolute_file_path(filename, 'pickle')
        # open a pickle file
        with open(filepath, 'rb') as file:
            data_object = pickle.load(file)
    except Exception as err:
        print(f"ERROR: {err}")

    return data_object

def open_dataset_file(filename):
    """
    Function that reads a pickle file
    """
    data_object = None
    try:
        filepath = get_absolute_file_path(filename, 'dataset')
        # open a pickle file
        df = pd.read_csv(filepath)
    except Exception as err:
        print(f"ERROR: {err}")

    return df


def load_embeddings(filename):
    embeddings_index = {}
    try:
        with open(filename, 'r', encoding="utf-8") as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_index[word] = vector
    except Exception as err:
        print(f"ERROR: {err}")

    return embeddings_index


def create_embedding_matrix(word_index, embeddings_index, EMBEDDING_DIM):
    embedding_matrix = np.zeros(((len(word_index)+1), EMBEDDING_DIM))
    for word, idx in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be changed to zeroes
            embedding_matrix[idx] = embedding_vector
    return embedding_matrix


def create_word2vec_embedding_matrix(word_index, embedding_path, embedding_dim):
    word2vec_model = KeywordProcessor.load_word2vec_format(embedding_path, binary=True)
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec_model:
            embedding_matrix[i] = word2vec_model[word]
    return embedding_matrix


def get_absolute_directory_path():
    return os.path.dirname(os.path.abspath(__file__)) 
    

def get_absolute_file_path(filename, folder):
    if '__file__' in globals():
        # Get the directory of the script file
        nb_dir = get_absolute_directory_path()
    else:
        # Get the directory of the notebook
        nb_dir = os.getcwd() 

    data_dir = os.path.join(nb_dir, folder)
    data_file = os.path.join(data_dir, filename)

    return data_file


def get_current_datetime():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%d%m%Y%H%M")
    return formatted_datetime