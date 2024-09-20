a = 0
b = a
c = 0
print(b, id(b), id(c))
print(a, id(a))
a = a + 1
print(a, b, id(b), id(a))
a += 1
print(a, id(a))



"""
Programming 2024
Seminar 1
Running python application and debugging
"""

# pylint: disable=invalid-name


# Three main ideas:
# 1. Python is a program that launches another program.
#       We pass the code from main.py into the python virtual machine (PVM),
#           where it is executed. For example, >>> python main.py
# 2. Reading the output in the terminal after executing the program will help you
#       find the error. Everything is told in the terminal output,
#           the main thing is to look carefully.
# 3. Debugging is the first skill of a programmer.
#       It helps you to take a step-by-step look at how the program works.


# Debugging exercise. Debug the program and fix errors:
first_num = 4
second_num = 3
#print(f'Numbers: {first_num} and {second_num}')

# Case1: the first exception trigger:
# third_num = first_num / second_num

# Fix the first exception:
second_num = 3
third_num = first_num / second_num
#print(third_num)
#print(f'Numbers: {first_num}, {second_num} and {third_num}')
c = 1.3333333
stroka = str(c)
#print(f(stroka))

a = f"Моё число: {c:.2f}" #сначала в скобках число, если мы хотим отформатировать то {.format(c)} f говорит о float, .2 значит что после точки 2 цифры
print(a)
exit(0)
#типы данных, по которым можно интегрироваться, называются последовательностями
#при умножении на флот сравниваются типы данных , при уможении на отрицательное, но целое число проверки проходят

# Case 2: the second debugging case:
if first_num * second_num == third_num:
    print('First number * second equals third number')
else:
    print('First number * second does not equal third number')

print('Program finished')
