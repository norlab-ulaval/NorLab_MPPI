#! /bin/bash
virtualenv -p /usr/bin/python3.9 venv
venv/bin/pip3.9 install ipympl
venv/bin/pip3.9 install pandas
venv/bin/pip3.9 install scipy
venv/bin/pip3.9 install pyquaternion
venv/bin/pip3.9 install jupyterlab
venv/bin/jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
venv/bin/jupyter labextension install jupyter-matplotlib --no-build
venv/bin/jupyter lab build
