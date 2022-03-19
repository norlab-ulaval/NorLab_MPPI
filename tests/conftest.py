# coding=utf-8

import pytest
import yaml


@pytest.fixture(scope='session')
def setup_virtual_display():
    from pyvirtualdisplay import Display
    virtual_display = Display(visible=False, size=(1400, 900))
    virtual_display.start()
    yield
    virtual_display.stop()


@pytest.fixture(scope="function")
def setup_mock_config_dict_CartPole():
    """ Test config file specify environment as gym CartPole-v1 """
    config_path = "tests/tests_barebones_mpc/config_files/default_test_config_CartPole-v1.yaml"
    with open(config_path, 'r') as f:
        config_dict = dict(yaml.safe_load(f))
    return config_dict


@pytest.fixture(scope="function")
def setup_mock_config_dict_Pendulum():
    """ Test config file specify environment as gym Pendulum-v1 """
    config_path = "tests/tests_barebones_mpc/config_files/default_test_config_Pendulum-v1.yaml"
    with open(config_path, 'r') as f:
        config_dict = dict(yaml.safe_load(f))
    return config_dict
