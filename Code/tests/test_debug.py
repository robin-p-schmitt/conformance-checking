import time


def test_debug():
    """test
    >>> 1 + 1
    2
    """
    if time.time() == 0:
        print("Hi!")
    assert "foobar".removeprefix("foo") == "bar"
