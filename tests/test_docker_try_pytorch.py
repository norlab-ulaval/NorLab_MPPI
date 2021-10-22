#!/usr/bin/env python3

import pytest
from try_pytorch_copy import verify_pytorch_install, verify_pytorch_cuda_install # (CRITICAL) todo:investigate >> temporary fix (!)


def test_verify_pytorch_install_PASS():
    verify_pytorch_install()
    return None


def test_verify_pytorch_cuda_install():
    verify_pytorch_cuda_install()
    return None


def test_fail():
    with pytest.raises(AssertionError):
        raise AssertionError
