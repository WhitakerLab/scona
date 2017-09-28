#!/usr/bin/env python

"""
This example is taken from the pytest getting started docs
https://docs.pytest.org/en/latest/getting-started.html
(but changed to make it pass, not fail).
Added by Kirstie on 27th Sept, and can be deleted
at any time (I just wanted to see Travis pass!)
"""

# content of test_sample.py
def func(x):
    return x + 1

def test_answer():
    assert func(3) == 4
    
