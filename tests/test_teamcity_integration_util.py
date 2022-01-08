# coding=utf-8

import pytest


# Test note: test 13 manual merge to dev

@pytest.mark.skip(reason="mute for now")
def test_manualy_triggered_fail():
    raise AssertionError


def test_manualy_triggered_PASS():
    assert True
