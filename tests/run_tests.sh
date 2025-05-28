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

REPORTS_DIR="${THIS_DIR}/../Reports"

echo "Reports directory: $REPORTS_DIR"

test_summary=""
# run tests one by one
for test in "${tests[@]}"; do
    echo "Running $test..."
    # run the test script in Slicer
    $SLICER_EXECUTABLE --no-splash --python-script ${THIS_DIR}/run_tests.py "$test"

    # Get all report files in the Reports directory again after running the test
    report_files=("${REPORTS_DIR}"/*/*.pdf)

    # get the last modified date of all report files
    report_dates=()
    for file in "${report_files[@]}"; do
        if [ -f "$file" ]; then
            report_dates+=("$(stat -c %Y "$file")")
        fi
    done

    current_time=$(date +%s)

    # date differences
    date_diffs=()
    for date in "${report_dates[@]}"; do
        diff=$((current_time - date))
        date_diffs+=("$diff")
    done

    # sort the date differences
    sorted_diffs=$(printf "%s\n" "${date_diffs[@]}" | sort -n)
    min_diff=${sorted_diffs[0]}

    # in case of no reports, set min_diff to a large number
    if [ ${#report_dates[@]} -eq 0 ]; then
        min_diff=9999999999
    fi

    # Check if the Slicer executable ran successfully and if the newest report was created within the last 10 seconds
    if [ $? -eq 0 ] && [ "$min_diff" -lt 10 ]; then
        test_summary+="Test $test: PASSED\n"
    else
        test_summary+="Test $test: FAILED\n"
    fi
done

# Print the summary of test results
echo -e "\nTest Summary:"
echo -e "$test_summary"
