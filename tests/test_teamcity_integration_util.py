# coding=utf-8

import pytest


# Test note: test 12 refactor build step

@pytest.mark.skip(reason="mute for now")
def test_manualy_triggered_fail():
    raise AssertionError


def test_manualy_triggered_PASS():
    assert True
