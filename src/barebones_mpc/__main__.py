# coding=utf-8

import argparse
import yaml

""" 
"""

# todo:implement hparam
# todo:implement test case hparam
# todo:implement input constraint (action)
# todo:implement state constraint


# todo:description redaction >> next bloc ↓↓
parser = argparse.ArgumentParser(description=(
    "=============================================================================\n"
    ":: Command line option for the NorLab-MPPI barebones_mpc package.\n\n"
    "   blablabla\n"
    "   You can execute the package by using the argument: --blabla "),
    epilog="=============================================================================\n")

parser.add_argument('--config', type=str, default='')
parser.add_argument('--execute', action='store_true', help='Execute barebones MPC')
parser.add_argument('--testSpec', action='store_true', help='Flag for automated continuous integration test')

# # ››› Parser references ›››...........................................................................................
# parser.add_argument('--intDefaultExample', type=int, default=20, help='')
# parser.add_argument('--emptyBoolExample', default=None, type=bool, help='')
# parser.add_argument('--floatExample', type=float, default=1.6, help='')
# parser.add_argument('-b', '--BoolExample', action='store_true', help='')
# # ...........................................................................................‹‹‹ Parser references ‹‹‹


args = parser.parse_args()

# exp_spec = ExperimentSpec()
#
if args.execute:
    # Configure experiment hyper-parameter
    if args.testSpec:
        pass
    else:
        pass
else:
    pass

exit(0)
