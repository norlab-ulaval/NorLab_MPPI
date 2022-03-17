#!/usr/bin/env python3

import pytest

from src.container_related.ds_python_utility import is_run_in_DS_arm64_darwin_architecture
from src.container_related.try_pytorch import verify_pytorch_install, verify_pytorch_cuda_install


def test_verify_pytorch_install_PASS():
    verify_pytorch_install()
    return None


@pytest.mark.skipif(is_run_in_DS_arm64_darwin_architecture(),
                    reason="Cuda is not suported on Dockerized-SNOW container for arm64-Darwin architecture")
def test_verify_pytorch_cuda_install():
    verify_pytorch_cuda_install()
    return None
