#!/bin/bash

# the following tests need must be defined in RANO/utils/test_rano.py
tests=(
  #"test_RANO_dicom_CPTAC"
  "test_RANO_dicom_KCL"
  "test_RANO_nifti_MU"
)

# Possible paths to the Slicer executable
SLICER_EXECUTABLE_PATHS=(
  "Slicer"
  "${HOME}/bin/Slicer-5.8.1-linux-amd64/Slicer"
  "/home/researcher/slicer/Slicer-5.8.1-linux-amd64/Slicer"  # path in docker image
)

# check if any of the SLICER_EXECUTABLE_PATHS exist and use the first one found
SLICER_EXECUTABLE=""
for path in "${SLICER_EXECUTABLE_PATHS[@]}"; do
    if [ -f "$path" ]; then
        SLICER_EXECUTABLE="$path"
        break
    fi
done

# check if Slicer executable was found otherwise exit with error
if [ -z "$SLICER_EXECUTABLE" ]; then
    echo "Error: Slicer executable not found in the specified paths."
    exit 1
fi

echo "Using Slicer executable: $SLICER_EXECUTABLE"

THIS_DIR=$(dirname "$(readlink -f "$0")")

test_summary=""
# run tests one by one
for test in "${tests[@]}"; do
    echo "Running $test..."
    $SLICER_EXECUTABLE --no-splash --python-script ${THIS_DIR}/run_tests.py "$test"
    # Check if the Slicer executable ran successfully
    if [ $? -eq 0 ]; then
        test_summary+="Test $test: PASSED\n"
    else
        test_summary+="Test $test: FAILED\n"
    fi
done

# Print the summary of test results
echo -e "\nTest Summary:"
echo -e "$test_summary"
