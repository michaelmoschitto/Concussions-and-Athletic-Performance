#!/bin/bash
echo reminder that kernels are stored in  /home/mmoschit/.local/share/jupyter/kernels/
python3 -m venv .venv
source .venv/bin/activate
pip install ipykernel
ipython kernel install --user --name=kernel
pip install -r requirements.txt
