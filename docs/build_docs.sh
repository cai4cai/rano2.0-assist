#!/bin/bash

THIS_DIR=$(dirname "$(readlink -f "$0")")

APIDOC_SOURCE_DIR=$( realpath "$THIS_DIR/../RANO" )
SOURCE_DIR=$(realpath "$THIS_DIR/source")

echo "APIDOC_SOURCE_DIR: $APIDOC_SOURCE_DIR"
echo "SOURCE_DIR: $SOURCE_DIR"

sphinx-apidoc -f -e -M -o "$SOURCE_DIR" "$APIDOC_SOURCE_DIR"


BUILD_DIR=$(realpath "$THIS_DIR/build")


echo "BUILD_DIR: $BUILD_DIR"

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