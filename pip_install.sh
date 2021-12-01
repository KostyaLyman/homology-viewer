#!/usr/bin/env bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
python --version
which python
pip --version

pip install numpy
pip install scipy
pip install networkx
pip install pandas
pip install -U matplotlib;
pip install notebook

pip install plotly
pip install dash
pip install dash-bootstrap-components
pip install dash-extensions

pip install colour

printf "\n Hooray! \n"
