# """
# Programming 2023
# Seminar 5
#
#
# Functions
# """
#
# # Built-in functions:
# # print() -> prints the value into console
# # max() -> finds the maximum element in an array
# # min() -> finds the minimum element in an array
# # dict() -> creates dictionary
# # str() -> casts the value to a string
# # list() -> creates an empty list or converts value to a list
# # type() -> returns type of the value
# # etc.
#
#
# # Create a function
# def function() -> None:
#     """
#     Sample function, prints a string when called
#     """
#     print('Function is called!')
#
#
# # Call a function
# function()
#
#
# # Functions always return value
# def return_hello_world() -> str:
#     """
#     Function returns a string 'Hello world'
#     :return: str
#     """
#     return 'Hello world!'
#
#
# RETURN_VALUE = return_hello_world()
# print(RETURN_VALUE)  # 'Hello world'
#
#
# def print_hello_world() -> None:
#     """
#     Function prints a string 'Hello world'
#     :return: None
#     """
#     print('Hello world!')
#
#
# # RETURN_VALUE = print_hello_world()
# # print(RETURN_VALUE)  # None
#
#
# # By default, functions accept the exact number of arguments
# def function_with_two_arguments(arg1: int, arg2: int) -> None:
#     """
#     Receives and prints 2 arguments
#     """
#     print(f'I received two arguments: {arg1} and {arg2}')
#
#
# # fail: TypeError: function_with_two_arguments() missing 1 required positional argument: 'arg2'
# # function_with_two_arguments(1)
#
# # fail: TypeError: function_with_two_arguments() takes 2 positional arguments but 3 were given
# # function_with_two_arguments(1, 2, 3)
#
# # success: I received two arguments: 1 and 2
# function_with_two_arguments(1, 2)
#
#
# # Default arguments may or may not be passed
# def print_all_arguments(arg1: str, arg2: str, arg3: str = 'Argument 3') -> None:
#     """
#     Receives 2 or 3 arguments, prints 3 arguments
#     """
#     print(f'I received these arguments: {arg1, arg2, arg3}')
#
#
# print_all_arguments('Argument 1', 'Argument 2')
# print_all_arguments('Argument 1', 'Argument 2', 'Argument 4')
#
#
# # Positional vs Keyword arguments
# def who_loves_whom(who: str, whom: str) -> None:
#     """
#     Receives two arguments, order is critical!
#     """
#     print(f'{who} loves {whom}')
#
#
# who_loves_whom('mother', 'daughter')  # mother loves daughter
#
# who_loves_whom(whom='mother', who='daughter')  # daughter loves mother
#
#
# # Local variables vs global variables
# def knowing_function(local_variable: str) -> None:
#     """
#     Prints both local and global variables
#     """
#     print(f'I know the following variables: {local_variable}, {GLOBAL_VARIABLE}')
#
#
# GLOBAL_VARIABLE = 'global variable'
#
# # success: 'I know the following variables: local_variable, global variable'
# knowing_function('local_variable')
#
#
# print('*' * 15 + ' TASKS ' + '*' * 15)
#
#
# # Task 1:
# # easy level
# def calculate_sum(arg_1: int, arg_2: int, arg_3: int) -> int:
#     """
#     Takes 3 integer arguments, outputs their sum
#     :return: int
#     """
#     # student realization goes here
#     return arg_1 + arg_2 + arg_3
#
#
# # Function calls with expected result:
# print(calculate_sum(1, 2, 3))
# # calculate_sum(1, -5, 0) # 4
#
#
# # Task 2:
# # easy level
# def calculate_power(base: int, exponent: int) -> int:
#     """
#     Takes 2 integer arguments: base and exponent
#     :return: int
#     """
#     # student realization goes here
#     return base ** exponent
#
#
# print(calculate_power(2, 3))
# # Function calls with expected result:
# # calculate_power(2, 3) # 8
# # calculate_power(7, 2) # 49
# # calculate_power(1589329, 0) # 1
#
#
# # Task 3:
# # easy level
# def calculate_factorial(number: int) -> int:
#     """
#     Takes 1 integer argument and calculates its factorial value
#     :return: int
#     """
#     # student realization goes here
#     result = 1
#     while number > 0:
#         result *= number
#         number -= 1
#     return result
#
#
# print(calculate_factorial(3))
# print(calculate_factorial(5))
# # Function calls with expected result:
# # calculate_factorial(3) # 6
# # calculate_power(2) # 2
# # calculate_power(0) # 1
#
#
# # Task 4:
# # medium level
# def encode_message(word: str, data: dict[str, int]) -> list[int]:
#     """
#     Takes 2 arguments: a string message and a dictionary, mapping characters to digits.
#     Returns a list of digits.
#     :return: list[int]
#     """
#     # student realization goes here
#     result = []
#     for letter in word:
#         result.append(data[letter])
#     return result
#
#
# print(encode_message('hello', {'h': 1, 'e': 2, 'l': 3, 'o': 4}))
# # Function calls with expected result:
# # encode_message('hello', {'h': 1, 'e': 2, 'l': 3, 'o': 4'}) # [1, 2, 3, 3, 4]
# # encode_message('abba', {'a': 1, 'b': 2, 'c': 3, 'd': 4'}) # [1, 2, 2, 1]
#
#
# # Task 5:
# # medium level
# def scream(phrase: str) -> None:
#     """
#     Takes 1 argument: a string with characters in any case.
#     Prints the capitalized version of a string.
#     :return: None
#     """
#     # student realization goes here
#     print(phrase.upper())
#
#
# scream('I love programming in python')
# # Function calls with expected result:
# # scream('I love programming in python') # 'I LOVE PROGRAMMING IN PYTHON'
# # scream('Functions are amazing') # 'FUNCTIONS ARE AMAZING'
#
#
# # Task 6
# # hard level
# # def is_allowed_to_drive(...) -> bool:
# #     """
# #     Takes 1 or 2 arguments: a dictionary with personal information including age (integer),
# #     age threshold determining if a person is allowed to drive a car.
# #     If the age threshold is not passed, it is equal to 18 y.o.
# #     Function outputs True if a person is allowed to drive and False otherwise.
# #     :return: bool
# #     """
# #     # student realization goes here
#
# # Function calls with expected result:
# # is_allowed_to_drive({'name': 'Kath', 'eyes': 'blue', 'age': 20}, 21) # False
# # is_allowed_to_drive({'name': 'Dean', 'height': '178', 'age': 20}) # True
#
#
# # Task 7
# # hard level
# # def get_fibonacci_sequence(...) -> list[int]:
# #     """
# #     Takes 1 integer argument,
# #     outputs fibonacci sequence of such length, starting from [1, 1,...].
#     Fibonacci sequence is the sequence where each element is equal to the sum of the previous two.
# #     :return: list[int]
# #     """
# #     # student realization goes here
#
#
# # Function calls with expected result:
# # get_fibonacci_sequence(7) # [1, 1, 2, 3, 5, 8, 13]
# # get_fibonacci_sequence(2) # [1, 1]
