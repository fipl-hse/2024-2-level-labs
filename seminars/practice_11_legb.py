"""
Programming 2024
Seminar 10
Working with enclosing scope
"""

# pylint: disable=missing-function-docstring, unused-argument

from typing import Callable


def wrapper_func() -> Callable:
    cache = {}

    def internal(first: int, second: int) -> int:
        pair = (first, second)
        if pair not in cache:
            print(f"Computing for {pair} ...")
            cache[pair] = first + second
        print("Retrieving results from internal cache...")
        return cache[pair]

    return internal


def cached(target_func: Callable) -> Callable:
    cache = {}

    def internal(*args: int) -> int:
        if args not in cache:
            print(f"Computing for {args} ...")
            cache[args] = sum(args)
        print("Retrieving results from internal cache...")
        return cache[args]

    return internal


@cached
def f(first: int, second: int) -> int:
    return first + second


def main() -> None:
    print("######### Closure-based calls")
    wrapped_func = wrapper_func()
    res = wrapped_func(10, 20)
    res = wrapped_func(10, 20)
    res = wrapped_func(10, 20)
    res = wrapped_func(10, 20)
    res = wrapped_func(10, 20)
    res = wrapped_func(10, 20)
    print(f"Result is {res}")

    print("\n\n\n######### Decorator-based calls")
    res = f(10, 20)
    res = f(10, 20)
    res = f(10, 20)
    res = f(10, 20)
    res = f(10, 20)
    res = f(10, 20)
    print(f"Result is {res}")


if __name__ == "__main__":
    main()
