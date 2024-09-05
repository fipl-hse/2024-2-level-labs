set -ex
source config/common.sh

which python

python -m pip install --upgrade pip
python -m pip install virtualenv
python -m virtualenv venv

configure_script

which python

python -m pip install -r requirements.txt
python -m pip install -r requirements_qa.txt

export PYTHONPATH
