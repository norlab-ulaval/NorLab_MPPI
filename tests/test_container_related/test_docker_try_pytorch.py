#!/usr/bin/env python3

import pytest
from src.container_related.try_pytorch import verify_pytorch_install, verify_pytorch_cuda_install


def test_verify_pytorch_install_PASS():
    verify_pytorch_install()
    return None


def test_verify_pytorch_cuda_install():
    verify_pytorch_cuda_install()
    return None
