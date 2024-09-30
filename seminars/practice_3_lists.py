"""
Programming 2024
Seminar 3


Data Type: Lists
"""
# pylint: disable=unused-argument


# Common information about lists

# lists are mutable
# lists are iterable

# Create a list
example = [1, 2, 3]
print(example)

# List concatenation, the original list doesn't change
first_list = example + [2, 3, 4]
print(example)
print(first_list)

# List changes
example.append(2)
example.extend([2, 3, 4])
print(example)

# List copy
# import copy


first_test = [1, 2, 3, [1, 2, 3]]
test_copy = first_test.copy()
print(first_test, test_copy)
test_copy[3].append(4)
print(first_test, test_copy)

first_test = [1, 2, 3, [1, 2, 3]]
# test_deepcopy = copy.deepcopy(first_test)
# test_deepcopy[3].append(4)
# print(first_test, test_deepcopy)

# List methods (some of them)
# .insert(index, item) -> inserts the given item on the mentioned index
# .remove(item) - removes the first occurrence of the given item from the list
# .pop() or .pop(index) – removes the item from the given index
# (or the last item) and returns that item
# .index(item) – returns the index of the first occurrence
# .sort() – sorts the list in place i.e modifies the original list
# .reverse() – reverses the list in place
# .copy() – returns a shallow copy of the list

# TASKS

# Task 1:
# easy level


def count_evens(nums: list) -> int:
    """
    Return the number of even ints in the given array.
    """
    how_many_evens = 0
    for number in nums:
        if number % 2 == 0:
            how_many_evens += 1
    return how_many_evens


print(count_evens([2, 1, 2, 3, 4]))
print(count_evens([2, 2, 0]))
print(count_evens([1, 3, 5]))

# Task 2:
# easy level


def sum13(nums: list) -> int:
    """
    Return the sum of the numbers in the array, returning 0 for an empty array.
    Except the number 13 is very unlucky,
    so it does not count and numbers that come after a 13
    also do not count.
    """
    total = 0
    skip = False
    for i in nums:
        if i == 13:
            skip = True
            continue  # Пропускаем 13
        if skip and i != 7:
            continue  # Пропускаем все после 13
        total += i
        if i == 7:
            skip = False  # Возобновляем подсчет после 7
    return total


print(sum13([1, 2, 2, 1]))
print(sum13([1, 1]))
print(sum13([1, 2, 2, 1, 13]))
print(sum13([1, 2, 2, 1, 13, 5, 6]))

# Task 3
# easy level


def sum67(nums: list) -> int:
    """
    Return the sum of the numbers in the array,
    except ignore sections of numbers starting with a 6 and extending to the next 7
    (every 6 will be followed by at least one 7).
    Return 0 for no numbers.
    """
    result = 0
    skip = False

    for num in nums:
        if num == 6:
            skip = True
        if not skip:
            result += num
        if num == 7:
            skip = False

    return result


assert sum67([1, 2, 2, 1]) == 6
assert sum67([1, 1]) == 2
assert sum67([1, 2, 2, 1, 6, 99, 7]) == 6
assert sum67([1, 1, 6, 7, 2]) == 4

# Task 4
# easy level


def create_phone_number() -> str:
    """
    Write a function that accepts an array of 10 integers (between 0 and 9),
    that returns a string of those numbers in the form of a phone number.
    """
    return '({nums[0:3]}) '

# Function calls with expected result:
# create_phone_number([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
# => returns "(123) 456-7890"


# Task 5
# medium level
def check_exam() -> int:
    """
    The first input array is the key to the correct answers to an exam,
    like ["a", "a", "b", "d"].
    The second one contains a student's submitted answers.
    The two arrays are not empty and are the same length.
    Return the score for this array of answers,
    giving +4 for each correct answer,
    -1 for each incorrect answer,
    and +0 for each blank answer, represented as an empty string.
    If the score < 0, return 0.
    """
    # student realization goes here

# Function calls with expected result:
# check_exam(["a", "a", "b", "b"], ["a", "c", "b", "d"]) → 6
# check_exam(["a", "a", "c", "b"], ["a", "a", "b",  ""]) → 7
# check_exam(["a", "a", "b", "c"], ["a", "a", "b", "c"]) → 16
# check_exam(["b", "c", "b", "a"], ["",  "a", "a", "c"]) → 0


# Task 6
# medium level
def who_likes_it() -> str:
    """
    You probably know the "like" system from Facebook and other pages.
    People can "like" blog posts, pictures or other items.
    We want to create the text that should be displayed next to such an item.
    """
    # student realization goes here

# Function calls with expected result:
# []                                -->  "no one likes this"
# ["Peter"]                         -->  "Peter likes this"
# ["Jacob", "Alex"]                 -->  "Jacob and Alex like this"
# ["Max", "John", "Mark"]           -->  "Max, John and Mark like this"
# ["Alex", "Jacob", "Mark", "Max"]  -->  "Alex, Jacob and 2 others like this"


# Task 7
# medium level
def find_anagrams() -> list:
    """
    What is an anagram?
    Two words are anagrams of each other if they both contain the same letters.
    'abba' and 'baab' are anagrams
    'abba' and 'bbaa' are anagrams
    'abba' and 'abbba' are not anagrams
    'abba' and 'abca' are not anagrams
    Write a function that will find all the anagrams of a word from a list.
    """
    # student implementation goes here

# Function calls with expected result:
# find_anagrams('abba') => ['aabb', 'bbaa']
# find_anagrams('racer') => ['carer', 'racer', ...]


# Task 8
# medium level
def scramble() -> bool:
    """
    Complete the function scramble(words: list)
    that returns true if a portion of str1 characters can be rearranged to match str2,
    otherwise returns false.
    """
    # student implementation goes here

# Function calls with expected result:
# scramble(['rkqodlw', 'world']) ==> True
# scramble(['cedewaraaossoqqyt', 'codewars']) ==> True
# scramble(['katas', 'steak']) ==> False
