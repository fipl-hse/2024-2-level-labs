"""
Programming 2024
Seminar 4


Data Type: Dictionary
"""
#pylint: disable=unused-argument
# Common information about dictionaries

# Dictionaries are used to store data values in key:value pairs.

# Create a dict
from typing import Union

example = {
    "brand": "Ford",
    "model": "Mustang",
    "year": 1964
}
print(example)
print('*' * 30)

# Create a dict (second way)
pair_example = dict([(1, 'Hello'), (2, 'there')])
print("\nDictionary with each item as a pair: ")
print(pair_example)
print('*' * 30)

# Add a key:value pair
example = {
    "brand": "Ford",
    "model": "Mustang",
    "year": 1964
}
example['colour'] = 'black'
print(example)
print('*' * 30)

# Remove a key:value pair
example.pop('colour')
print(example)
print('*' * 30)

# Change the value of the given key
example['year'] = 2000
print(example)
print('*' * 30)

# Dict methods (some of them)
# .get(key, default) -> get the value by the given key
# .update(another_dict) -> add key:value pairs from the another_dict
# .items() -> returns a list containing a tuple for each key value pair
# .values() -> returns a list of all the values in the dictionary
# .keys() -> returns a list of all the keys in the dictionary

print('*' * 15 + ' TASKS ' + '*' * 15)


# TASKS

# Task 1:
# easy level
def extract_older_people(people: dict[str, int], threshold: int) -> list[str]:
    """
    Return the names of the people who are older than 'threshold'
    """
    # student realisation goes here


# Function calls with expected result:
# assert extract_older_people({'Andrej': 22, 'Alexander': 28, 'Irine': 20},
#                             20) == ['Andrej', 'Alexander']
# assert extract_older_people({'Hera': 45, 'Zagreus': 25, 'Zeus': 48}, 30) == ['Hera', 'Zeus']

# Task 2:
# easy level
def sum_values(data: dict[str, int]) -> Union[float, int]:
    """
    Given a dictionary in Python,
    write a Python program to find the sum of all Items in the dictionary.
    """
    # student realisation goes here


# Function calls with expected result:
# assert sum_values({'a': 300, 'b': 15, 'c': 430}) == 745

# Task 3
# easy level
def find_key(data: dict[str, int]) -> str:
    """
    Return the key with the maximum value
    """
    # student realisation goes here


# Function calls with expected result:
# assert find_key({'Andrej': 10000, 'Artyom': 15000, 'Alexander': 100000}) == 'Alexander'

# Task 4
# easy level
def remove_duplicates(data: dict[str, int]) -> dict[str, int]:
    """
    Write a function that removes duplicates (key:value pairs with same values)
    """
    # student realisation goes here


# Function calls with expected result
# assert remove_duplicates({
#     'Marat': 10000,
#     'Yaroslav': 15000,
#     'Sasha': 10000}) == {'Yaroslav': 15000}

# Task 5
# medium level
def count_letters(sequence: str) -> dict[str, int]:
    """
    Given a string,
    write a Python program to count the number of times each element is found in the string
    (register is to be ignored)
    """
    # student realisation goes here


# Function calls with expected result:
# assert count_letters('Hello there') == {'h': 2, 'e': 3, 'l': 2, 'o': 1, 't': 1, 'r': 1}

# Task 6
# medium level
def decipher(sentence: str, special_characters: dict[int, str]) -> str:
    """
    You are given a secret message you need to decipher.

    Here are the things you need to know to decipher it:
    For each word:

    the second and the last letter is switched (e.g. Hello becomes Holle)
    the first letter is replaced by its character code (e.g. H becomes 72)

    Note: there are no special characters used, only letters and spaces
    """
    # student realisation goes here


# Function calls with expected result:
# character_decoded_dict = {'H': 72, 'g': 103, 'd': 100, 'R': 82, 's': 115}
# assert decode('72olle 103doo 100ya', character_decoded_dict) == 'Hello good day'
# assert decode('82yade 115te 103o', character_decoded_dict) == 'Ready set go'

# Task 7
# medium level
def bake_cakes(recipe: dict[str, int], ingredients: dict[str, int]) -> int:
    """
    Pete likes to bake cakes. He has some recipes and ingredients.
    Unfortunately he is not good at maths.
    Can you help him find out, how many cakes he can bake considering his recipes?

    Implement a function bake_cakes(), which takes the recipe (dict) and
    the available ingredients (also dict) and returns the maximum number
    of cakes Pete can bake (integer).
    Number of ingredients that are absent from the available ingredients dict to considered as 0.
    """
    # student realisation goes here


# Function calls with expected result:
# assert bake_cakes({'flour': 500, 'sugar': 200, 'eggs': 1},
#            {'flour': 1200, 'sugar': 1200, 'eggs': 5, 'milk': 200}) == 2
# assert bake_cakes({'apples': 3, 'flour': 300, 'sugar': 150, 'milk': 100, 'oil': 100},
#            {'sugar': 500, 'flour': 2000, 'milk': 2000}) == 0