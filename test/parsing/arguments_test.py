from src.parsing.arguments import get_floats

samples = [
    "What is the square root of 16?"
    "Reverse the string 'hello'"
    "Reverse the string 'world'"
    "Substitute the digits in the string 'Hello 34 I'm 233 years old' with 'NUMBERS'"
    "Replace all vowels in 'Programming is fun' with asterisks"
    "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"
    "What is the sum of 2 and 3?"
    "What is the sum of 265 and 345?"
    "Is 4 an even number?"
    "Is 7 an even number?"
    "What is the product of 3 and 5?"
    "What is the product of 12 and 4?"
    "Greet shrek"
    "Greet john"
]


def test_get_floats():
    assert get_floats("What is the square root of 16?") == [16.0]
    assert get_floats("Reverse the string 'hello'") == []
    assert get_floats("Substitute the digits in the string 'Hello 34 I'm 233 years old' with 'NUMBERS'") == [34.0, 233.0]
    assert get_floats("What is the sum of 265 and 345?") == [265.0, 345.0]
    assert get_floats("What is the product of 12 and 0?") == [12.0, 0.0]
    assert get_floats("What is the product of -12 and +1?") == [-12.0, 1.0]
    assert get_floats("What is the product of 12.456 and 0.1?") == [12.456, 0.1]
    assert get_floats("What is the product of 12456 and 0.1?") == [12456, 0.1]
    assert get_floats("What is the product of 00456 and 0.1?") == [456, 0.1]

def test_get_strings():
    pass

def test_get_words():
    pass



