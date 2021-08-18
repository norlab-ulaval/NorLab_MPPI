#!/usr/bin/env python3


def verify_pytorch_install() -> None:
    """
    Minimal pytorch install verification. If the consol print a tensor like this, your good

        tensor([[0.3380, 0.3845, 0.3217],
                [0.8337, 0.9050, 0.2650],
                [0.2979, 0.7141, 0.9069],
                [0.1449, 0.1132, 0.1375],
                [0.4675, 0.3947, 0.1426]])

    Ref: https://pytorch.org/get-started/locally/

    :return: None
    """
    try:
        import torch
        x = torch.rand(5, 3)
        print("\n", x)
        print("\n> PyTorch install is good to go!\n")
    except Exception as e:
        print("Something is wrong with PyTorch. It's probably the Dockerized-SNOW python interpreter")
        print(e)

    return None


if __name__ == '__main__':

    verify_pytorch_install()
