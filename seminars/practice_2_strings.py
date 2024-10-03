"""
Programming 2024
Seminar 2


Data Type: String
"""

# pylint: disable=invalid-name,unused-argument

# Common information about strings
#
# strings are immutable
# strings are iterable
# strings are case-sensitive

# Create a string
#example = 'Hello'  # or "Hello"
#print(example)

# String concatenation
#greeting = example + ' there!'
#print(greeting)

# String multiplication
#several_hellos = example * 5
#print(several_hellos)

# String formatting
# .format() method
#example = '{} there!'.format(example)  # pylint: disable=consider-using-f-string
#print(example)
# f-strings
#example = f'{greeting} - this is the "greeting" variable.'
#print(example)


# String methods (some of them)
# .split() -> split by the delimiter
# .join() - join by the delimiter
# .upper() - uppercase copy of the string
# .lower() - lowercase copy of the string
# .isalpha() - if all the characters in the text are letters
# .strip() - remove the given element (space by default) from the both ends of the string
# .find() - search the string for the specified value (return the index of the first occurrence)

# TASKS

# Task 1:
def multiply_string(s, inter):
    """
    Given a string and a non-negative number,
    display the given string the number of times given in the `how_many`.
    """
    # student realisation goes here




# Task 2:
def front_times(s: str, num: int):
    """
    Given the string, take its three leading characters
    and display them that many times as in `how_many`.
    """
    # student realisation goes here





# Task 3:
def extra_end(s: str):
    """
    Given the string, take its two last characters and display them three times.
    """
    # student realisation goes here





# Task 4:
def make_abba(s1: str, s2: str):
    """
    Given two strings, concatenate them as a reflection.
    """
    # student realisation goes here





# Task 5
def reverse_word(s: str):
    """
    Write a function that takes in a string of one or more words,
    and returns the same string, but with all five or more letter words reversed.

    Strings passed in will consist of only letters and spaces.
    Spaces will be included only when more than one word is present.
    """
    # student realisation goes here



# Task 6
def generate_hashtag(s: str):
    """
    The marketing team is spending way too much time typing in hashtags.
    Let's help them with our own Hashtag Generator!

    Here's the deal:

    It must start with a hashtag (#).
    All words must have their first letter capitalized.
    If the final result is longer than 140 chars it must return false.
    If the input or the result is an empty string it must return false.
    Examples
    " Hello there thanks for trying my quiz"  =>  "#HelloThereThanksForTryingMyQuiz"
    "    Hello     World   "                  =>  "#HelloWorld"
    ""                                        =>  false
    """
    # student realisation goes here

# Task 7:
def combo_string(first_string: str, second_string: str) -> str:
    """
    Given two strings, concatenate like the following: shorter+longer+shorter
    """
    # student realisation goes here


# combo_string('Hello', 'hi') → 'hiHellohi'
# combo_string('hi', 'Hello') → 'hiHellohi'
# combo_string('aaa', 'b') → 'baaab'
# combo_string('', 'bb') → 'bb'
# combo_string('aaa', '1234') → 'aaa1234aaa'
# combo_string('bb', 'a') → 'abba'


# Task 1: advanced
def string_splosion(input_string: str) -> str:
    """
    Given the string, format it like in the example.
    """
    # student realisation goes here


# Function calls with expected result:
# string_splosion('Code') → 'CCoCodCode'
# string_splosion('abc') → 'aababc'
# string_splosion('ab') → 'aab'
# string_splosion('Kitten') → 'KKiKitKittKitteKitten'
# string_splosion('x') → 'x'


# Task 2: advanced
def string_match(first_string: str, second_string: str) -> int:
    """
    Given two strings, find the number of times an arbitrary substring (with length of 2)
    is found at the same position in both strings.
    """
    # student realisation goes here

# Function calls with expected result:
# string_match('xxcaazz', 'xxbaaz') → 3
# NOTE: «xx», «aa» and «az» are found at the same position in both strings
# string_match('abc', 'abc') → 2
# string_match('abc', 'axc') → 0
# string_match('he', 'hello') → 1
# string_match('h', 'hello') → 0
# string_match('aabbccdd', 'abbbxxd') → 1
# string_match('aaxxaaxx', 'iaxxai') → 3
# string_match('iaxxai', 'aaxxaaxx') → 3
