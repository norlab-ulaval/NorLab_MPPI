# coding=utf-8

import pytest

@pytest.fixture(scope='session')
def setup_virtual_display():
    from pyvirtualdisplay import Display
    virtual_display = Display(visible=False, size=(1400, 900))
    virtual_display.start()
    yield
    virtual_display.stop()
