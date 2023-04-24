#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
pip install ipykernel
ipython kernel install --user --name=kernel2
pip install -r requirements.txt
