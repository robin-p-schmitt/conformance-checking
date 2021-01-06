import time


def test_debug():
    """test
    >>> 1 + 1
    2
    """
    if time.time() == 0:
        print("Hi!")
    assert int("1") == 1
