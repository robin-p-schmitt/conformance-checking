# This file is just for demonstration purposes.
# Delete it as soon as actual tests are there.
import time
import pytest


def test_debug():
    """test
    >>> 1 + 1
    2
    """
    if time.time() == 0:
        print("Hi!")

    # this is how you can write a test expecting an exception
    with pytest.raises(ValueError, match=r"must be \d+$"):
        raise ValueError("value must be 42")

    assert int("1") == 1
