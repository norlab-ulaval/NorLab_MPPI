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
# parser.add_argument('-r', '--render_training', action='store_true',
#                     help='(Training option) Watch the agent execute trajectories while he is on traning duty')
#
# parser.add_argument('-p', '--play_for', type=int, default=20,
#                     help='(Playing option) Max playing trajectory, default=20')
#
# parser.add_argument('-d', '--discounted', default=None, type=bool,
#                     help='(Training option) Force training execution with discounted reward-to-go')
#
# parser.add_argument('--harderEnvCoeficient', type=float, default=1.6,
#                     help='Harder environment coeficient (if it can be applied)')
#
# parser.add_argument('--record', action='store_true',
#                     help='(Playing option) Record trained agent playing in a environment')

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
