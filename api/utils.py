#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    utils.py
# @Author:      Kuro
# @Time:        1/18/2025 11:05 AM

# ------------------------------------------------
# Utility: Map of spoken words to digits
# ------------------------------------------------
WORD_TO_DIGIT = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9"
}


# ------------------------------------------------
# Utility: Find exactly 3 consecutive spoken digits
# ------------------------------------------------
def find_three_spoken_digits(text: str):
    """
    Finds exactly 3 consecutive spoken digits in the text and returns them
    as a string (e.g., "four seven one" => "471"). Returns None if not found.
    """
    tokens = text.lower().split()
    consecutive_digits = []

    for token in tokens:
        if token in WORD_TO_DIGIT:
            consecutive_digits.append(WORD_TO_DIGIT[token])
        else:
            consecutive_digits = []  # reset if we hit a non-digit token

        if len(consecutive_digits) == 3:
            return "".join(consecutive_digits)

    return None
