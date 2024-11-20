"""
Programming 2024
Seminar 8


Inheritance
"""

# pylint:disable=missing-class-docstring,too-few-public-methods,pointless-string-statement

"""
Vehicle
    Attributes:
        max_speed
        colour
    Methods:
        move
"""


class Vehicle: ...
"""
Car
    Attribute:
        max_speed
        colour
        fuel
    Methods:
        move
        stay
"""


class Car: ...


LADA = ...


"""
Bicycle
    Attributes:
        number_of_wheels
        colour
        max_speed
    Methods:
        move
        freestyle
"""


class Bicycle(Vehicle): ...


# stels = Bicycle('yellow', 30, 2)
# print(stels.colour)
# stels.move()
# stels.freestyle()


"""
Aircraft
    Attributes:
        number_of_engines
        colour
        max_speed
    Methods:
        move
"""


class Aircraft: ...


# for...