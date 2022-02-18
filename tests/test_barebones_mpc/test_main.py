# coding=utf-8

import pytest
from src.utils.ttd_related import is_run_on_a_teamcity_continuous_integration_server


pytestmark = pytest.mark.automated_test

is_run_on_TeamCity_CI_server = is_run_on_a_teamcity_continuous_integration_server()


def command_line_test_error_msg(out):
    return "Module invocated from command line exited with error {}".format(out)


@pytest.mark.skipif(is_run_on_TeamCity_CI_server, reason="Mute on CI server")
def test_command_line_invocation_script_execute_barebones_mpc_PASS():
    from os import system

    # ››› locate script execution in Dockerized-SNOW ››› . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    # import os
    # print("\n:: PWD: ", os.getcwd(), "\n:: ls: ", os.listdir())
    # os.chdir("src")
    # print(":: PWD: ", os.getcwd(), "\n:: ls: ", os.listdir())

    #  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .‹‹‹ locate script execution in Dockerized-SNOW ‹‹‹

    out = system("python3 -m src.barebones_mpc --execute --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)

# def test_manualy_triggered_FAIL():
#     assert False

#
# def test_manualy_triggered_PASS():
#     assert True
