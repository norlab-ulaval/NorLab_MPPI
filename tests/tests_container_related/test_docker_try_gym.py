#!/usr/bin/env python3

import pytest
from src.container_related.try_gym import verify_gym_classic_control_install, verify_gym_box2d_install



# @pytest.mark.skip(reason="tmp mute")
def test_verify_gym_classic_control_install_PASS(setup_virtual_display):
    verify_gym_classic_control_install()
    return None

# @pytest.mark.skip(reason="tmp mute")
def test_verify_gym_box2d_install_PASS(setup_virtual_display):
    verify_gym_box2d_install()
    return None

