from __future__ import annotations


def ensure_divisibility(numerator: int, denominator: int) -> None:
    if denominator == 0:
        raise ValueError("denominator must not be 0")
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}")


def divide(numerator: int, denominator: int) -> int:
    ensure_divisibility(numerator, denominator)
    return numerator // denominator
