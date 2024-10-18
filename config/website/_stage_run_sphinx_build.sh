#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running Sphinx build check...'

configure_script

rm -rf dist

make -C config/website/test_sphinx_project html SPHINXOPTS="-W --keep-going -n"
check_if_failed

echo "Sphinx build succeeded."
