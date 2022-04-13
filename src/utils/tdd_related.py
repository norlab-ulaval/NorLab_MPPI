# coding=utf-8
from os import getenv

import argparse
from typing import Tuple


def is_automated_test() -> bool:
    """ Check if code is executed from a test suite """
    automated_test = False
    try:
        globals_pytestmark_ = globals()['pytestmark']
        if globals_pytestmark_.markname == 'automated_test':
            automated_test = True
    except KeyError:
        pass
    return automated_test


# (NICE TO HAVE) todo:refactor for yaml file>> next bloc ↓↓
# def check_testspec_flag_and_setup_spec(user_spec: ExperimentSpec,
#                                        test_spec: ExperimentSpec) -> Tuple[ExperimentSpec, bool]:
#     """
#     Parse command line arg and check for unittest specific. Return test specification if --testSpec is raise
#     :return: user_spec or test_spec
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-t', '--testSpec', input='store_true', help='Use test spec')
#
#     args = parser.parse_args()
#
#     is_test_run = False
#     spec = user_spec
#
#     if args.testSpec:
#         is_test_run = True
#         spec = test_spec
#
#     return spec, is_test_run


def is_run_on_a_teamcity_continuous_integration_server() -> bool:
    """
    Check if python is executed in the continuous integration server.

    Note: The check is specific to TeamCity server
    """
    try:
        import teamcity as tc

        # if it return 'LOCAL' then it is not running on a TeamCity server
        tc_version = getenv('TEAMCITY_VERSION')

        if tc_version != 'LOCAL':
            print(f'\n:: is running under teamcity TEAMCITY_VERSION={tc_version}')
            return True
        else:
            print(f'\n:: TEAMCITY_VERSION={tc_version} ››› run not executed on CI server')
            return False
    except ImportError:
        print(f'\n:: python is not executed on CI server')
        return False


def show_plot_unless_CI_server_runned(show_plot: bool) -> bool:
    """
    Required to switch off matplotlib `plt.show()` on TeamCity continuous intergation server.

    Note: On `plt.show()`, python open up a window waiting to be close by the user which will stale the TeamCity server
    build queue.

    :param show_plot: set True or False as normal
    :return: 'show_plot' argument if python is executed locally, False otherwise
    """

    try:
        import teamcity as tc

        # if `getenv(...)` it return 'LOCAL' then it is not running on a TeamCity server
        tc_version = getenv('TEAMCITY_VERSION')

        if tc_version != 'LOCAL':
            print(f'\n:: is running under teamcity TEAMCITY_VERSION={tc_version}'
                  f' ››› switching `show_plot` to False\n')
            return False
        else:
            print(f'\n:: TEAMCITY_VERSION={tc_version} ››› run not executed on CI server'
                  f' ››› use user argument show_plot={show_plot}\n')
            return show_plot
    except ImportError:
        print(f'\n:: python is not executed on CI server ››› show_plot={show_plot}\n')
        return show_plot
