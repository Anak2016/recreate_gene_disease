def test_one():
    assert 1 == 1

import pytest
@pytest.fixture
def test_two():
    expected = (1,2,3)
    actual = (1,2,3)
    assert expected  == actual

class TestSomeStuff():
    def test_three(self):
        assert {1,2,2} == {1,2}
        assert {1,2,2} == {1,2}
        assert {1,2,2} == {1,2}
