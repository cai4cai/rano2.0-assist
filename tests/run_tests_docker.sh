#!/bin/bash

THIS_DIR=$(dirname "$(readlink -f "$0")")
${THIS_DIR}/../docker/docker_run.sh "/home/researcher/rano2.0-assist/tests/run_tests.sh"