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
    # student realization goes here
    count = 0
    for i in nums:
        if i % 2 == 0:
            count += 1
    return count


# Function calls with expected result:
# count_evens([2, 1, 2, 3, 4]) → 3
# count_evens([2, 2, 0]) → 3
# count_evens([1, 3, 5]) → 0

print('Task 1')
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
    # student realization goes here
    sum = 0
    skip = False
    for num in nums:
        if num == 13:
            skip = True
        elif not skip:
            sum += num
    return sum


# Function calls with expected result:
# sum13([1, 2, 2, 1]) → 6
# sum13([1, 1]) → 2
# sum13([1, 2, 2, 1, 13]) → 6
# sum13([1, 2, 2, 1, 13, 5, 6]) → 6


print('Task 2')
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
    # student realization goes here
    sum = 0
    skip = False
    for num in nums:
        if num == 6:
            skip = True
        elif num == 7:
            skip = False
        elif not skip:
            sum += num
    return sum


# Function calls with expected result:
# sum67([1, 2, 2]) → 5
# sum67([1, 2, 2, 6, 99, 99, 7]) → 5
# sum67([1, 1, 6, 7, 2]) → 4


print('Task 3')
print(sum67([1, 2, 2]))
print(sum67([1, 2, 2, 6, 99, 99, 7]))
print(sum67([1, 1, 6, 7, 2]))


# Task 4
# easy level
def create_phone_number(nums: list) -> str:
    """
    Write a function that accepts an array of 10 integers (between 0 and 9),
    that returns a string of those numbers in the form of a phone number.
    """
    # student realization goes here
    phone_number = "(" + "".join(str(num) for num in nums[:3]) + ") " + "".join(str(num) for num in nums[3:6]) + "-" + "".join(str(num) for num in nums[6:])
    return phone_number


# Function calls with expected result:
# create_phone_number([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
# => returns "(123) 456-7890"
print('Task 4')
print(create_phone_number([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))
# Task 5


# medium level
def check_exam(correct_answers: list, student_answers: list) -> int:
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

    #student realization goes here
    score = 0
    for i in range(len(correct_answers)):
        if student_answers[i] == "":
            score += 0
        elif student_answers[i] == correct_answers[i]:
            score += 4
        else:
            score -= 1
    return max(0, score)


# Function calls with expected result:
# check_exam(["a", "a", "b", "b"], ["a", "c", "b", "d"]) → 6
# check_exam(["a", "a", "c", "b"], ["a", "a", "b",  ""]) → 7
# check_exam(["a", "a", "b", "c"], ["a", "a", "b", "c"]) → 16
# check_exam(["b", "c", "b", "a"], ["",  "a", "a", "c"]) → 0
print('Task 5')
print(check_exam(["a", "a", "b", "b"], ["a", "c", "b", "d"]))
print(check_exam(["a", "a", "c", "b"], ["a", "a", "b",  ""]))
print(check_exam(["a", "a", "b", "c"], ["a", "a", "b", "c"]))
print(check_exam(["b", "c", "b", "a"], ["",  "a", "a", "c"]))


# Task 6
# medium level
def who_likes_it(names: list) -> str:
    """
    You probably know the "like" system from Facebook and other pages.
    People can "like" blog posts, pictures or other items.
    We want to create the text that should be displayed next to such an item.
    """
    # student realization goes here
    if len(names) == 0:
        return 'no one likes this'
    if len(names) == 1:
        return f'{names[0]} likes this'
    if len(names) == 2:
        return f'{names[0]} and {names[1]} like this'
    if len(names) == 3:
        return f'{names[0]}, {names[1]} and {names[2]} like this'
    if len(names) >= 4:
        return f'{names[0]}, {names[1]} and {len(names) - 2} others like this'

# Function calls with expected result:
# []                                -->  "no one likes this"
# ["Peter"]                         -->  "Peter likes this"
# ["Jacob", "Alex"]                 -->  "Jacob and Alex like this"
# ["Max", "John", "Mark"]           -->  "Max, John and Mark like this"
# ["Alex", "Jacob", "Mark", "Max"]  -->  "Alex, Jacob and 2 others like this"


print('Task 6')
print(who_likes_it([]))
print(who_likes_it(['Peter']))
print(who_likes_it(["Jacob", "Alex"]))
print(who_likes_it(["Max", "John", "Mark"]))
print(who_likes_it(["Alex", "Jacob", "Mark", "Max"]))


# Task 7
# medium level
def find_anagrams(words: list) -> list:
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


print('Task 7')
print(find_anagrams(['abba']))
print(find_anagrams(['racer']))


# Task 8
# medium level
def scramble(words: list) -> bool:
    """
    Complete the function scramble(words: list)
    that returns true if a portion of str1 characters can be rearranged to match str2,
    otherwise returns false.
    """
    # student implementation goes here
    str1 = words[0]
    str2 = words[1]
    for char in str2:
        if char not in str1:
            return False
        str1 = str1.replace(char, '', 1)
    return True


# Function calls with expected result:
# scramble(['rkqodlw', 'world']) ==> True
# scramble(['cedewaraaossoqqyt', 'codewars']) ==> True
# scramble(['katas', 'steak']) ==> False

print('Task 8')
print(scramble(['rkqodlw', 'world']))
print(scramble(['cedewaraaossoqqyt', 'codewars']))
print(scramble(['katas', 'steak']))
