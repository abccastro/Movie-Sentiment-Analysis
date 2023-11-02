import json
import re


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

        text = re.replace('"', "'")
        text = re.sub(pattern, '"', text)
        list_of_dicts = json.loads(text)
    except Exception as err:
        print(f"ERROR: {err}")
        print(f"Input text: {text}")

    return extract_specific_key(list_of_dicts, dict_keys)