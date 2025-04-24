#!/bin/bash

sphinx-apidoc -f -e -M -o source ../RANO

THIS_DIR=$(dirname "$(readlink -f "$0")")

SOURCE_DIR=$(realpath "$THIS_DIR/source")
BUILD_DIR=$(realpath "$THIS_DIR/build")

PYTHONEXEC="/home/slicer/bin/Slicer-5.8.1-linux-amd64/Slicer"

${PYTHONEXEC} --no-main-window --python-code "from sphinx.cmd.build import main as sphinxmain; sphinxmain(['-M', 'html', '${SOURCE_DIR}', '${BUILD_DIR}']); exit()"

# open the generated documentation in the default web browser
if command -v xdg-open &> /dev/null
then
    xdg-open "${BUILD_DIR}/html/index.html"
elif command -v open &> /dev/null
then
    open "${BUILD_DIR}/html/index.html"
else
    echo "Please open ${BUILD_DIR}/html/index.html in your web browser."
fi