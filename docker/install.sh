#!/bin/bash

set -e
set +x

script_dir=$(cd $(dirname $0) || exit 1; pwd)
PROG=$(basename $0)

err() { echo -e >&2 ERROR: $@\\n; }
die() { err $@; exit 1; }

if [ "$#" -ne 1 ]; then
  die "Usage: $PROG /path/to/Slicer-X.Y.Z-linux-amd64/Slicer"
fi

slicer_executable=$1

################################################################################
# Set up headless environment
source $script_dir/start-xorg.sh
echo "XORG_PID [$XORG_PID]"

################################################################################
# Install the extension install
$slicer_executable \
  --disable-loadable-modules \
  --disable-cli-modules \
  --disable-scripted-loadable-modules \
  -c "em = slicer.app.extensionsManagerModel();em.interactive=False;slicer.app.extensionsManagerModel().installExtensionFromServer('PyTorch', True)"

$slicer_executable \
  --disable-loadable-modules \
  --disable-cli-modules \
  --disable-scripted-loadable-modules \
  --python-script ./rano2.0-assist/install_dependencies.py

# Shutdown headless environment
kill -9 $XORG_PID
rm /tmp/.X10-lock