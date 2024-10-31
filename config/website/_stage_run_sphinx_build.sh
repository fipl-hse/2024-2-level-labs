#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running Sphinx build check...'

configure_script

rm -rf dist

sphinx-build -b html -W --keep-going -n . dist
check_if_failed

echo "Sphinx build succeeded."
