#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running Sphinx build check...'

configure_script

pushd config/website/test_sphinx_project

  # Treat warnings as errors, build full documentation
  rm -rf _build
  make html SPHINXOPTS="-W --keep-going -n"
  check_if_failed

popd

echo "Sphinx build succeeded."
