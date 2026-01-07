#!/bin/bash

THIS_DIR=$(dirname "$(readlink -f "$0")")

APIDOC_SOURCE_DIR=$( realpath "$THIS_DIR/../RANO" )
SOURCE_DIR=$(realpath "$THIS_DIR/source")

echo "APIDOC_SOURCE_DIR: $APIDOC_SOURCE_DIR"
echo "SOURCE_DIR: $SOURCE_DIR"

sphinx-apidoc -f -e -M -o "$SOURCE_DIR" "$APIDOC_SOURCE_DIR"


BUILD_DIR=$(realpath "$THIS_DIR/build")


echo "BUILD_DIR: $BUILD_DIR"

PYTHONEXEC="/home/aaron/bin/Slicer-5.10.0-linux-amd64/Slicer"

${PYTHONEXEC} --no-main-window --python-code 'from slicer.util import pip_install;pip_install("-U sphinx");pip_install("myst_parser");pip_install("sphinx-autoapi");pip_install("sphinx_rtd_theme");pip_install("beautifulsoup4");exit()'
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