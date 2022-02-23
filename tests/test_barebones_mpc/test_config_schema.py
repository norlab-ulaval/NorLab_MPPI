# coding=utf-8
from typing import Any, Dict

import pytest
import yaml


def test_config_file_test_schema_PASS():
    with open('tests/test_barebones_mpc/config_files/default_test_config.yaml', 'r') as f:
        config = dict(yaml.safe_load(f))

    print(config)
    assert type(config) is dict
    assert config['hparam']['experimental-hparam']['experimental_window']


@pytest.mark.skip(reason="wait until `default_test_config.yaml` is done implementing")
def test_config_file_default_PASS():
    with open('src/barebones_mpc/config_files/default.yaml', 'r') as f:
        config = dict(yaml.safe_load(f))
        print(config)
        assert type(config) is dict
        assert config['hparam']['experimental-hparam']['experimental_window']

