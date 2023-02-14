"""Try Except Code Trainings."""


def divide_vals(numerator, denominator):
    """Calculate the fraction of two numbers.

    Args
    ----
        numerator: (float) numerator of fraction
        denominator: (float) denominator of fraction

    Returns
    -------
        fraction_val: (float) numerator/denominator
    """
    try:
        fraction_val = numerator / denominator
    except ZeroDivisionError:
        fraction_val = "denominator cannot be zero"

    return fraction_val

    # try to return the fraction but if the denominator is zero
    # catch the error and return a string saying:
    # "denominator cannot be zero"


def count_words(text):
    """Count the number of words in a string.

    Args
    ----
        text: (string) string of words

    Returns
    -------
        num_words: (int) number of words in the string
    """
    try:
        num_words = len(text.split())
    except AttributeError:
        num_words = "text argument must be a string"

    return num_words
    # try to split based on spaces and return num of words
    # but if text isnt a string return "text argument must be a string"
