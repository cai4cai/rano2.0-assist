#!/usr/bin/env python-real

import os
import sys
import subprocess
import re
from pathlib import Path
from slicer import ScriptedLoadableModule

if __name__ == '__main__':
    # define progress bar steps as fraction of 1
    command_received_progress = 0.1
    first_output_progress = 0.2
    tqdm_start = command_received_progress + first_output_progress
    tqdm_end = 1.0

    command = sys.argv[1]

    # send progress bar start line to stdout
    print("""<filter-start><filter-name>TestFilter</filter-name><filter-comment>ibid</filter-comment></filter-start>""")
    sys.stdout.flush()

    print("""<filter-progress>{}</filter-progress>""".format(command_received_progress))
    sys.stdout.flush()

    # create startup environment for subprocess (to run python outside slicer)
    slicer_path = Path(os.environ["SLICER_HOME"]).resolve()
    PATH_without_slicer = os.pathsep.join(
        [p for p in os.environ["PATH"].split(os.pathsep) if not slicer_path in Path(p).parents])

    startupEnv = {}
    try:
        startupEnv["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    except:
        print("SYSTEMROOT environment variable does not exist...")

    try:
        startupEnv["USERPROFILE"] = os.environ["USERPROFILE"]
    except:
        print("USERPROFILE environment variable does not exist...")

    startupEnv['PATH'] = PATH_without_slicer

    popen = subprocess.Popen(command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True,
                             env=startupEnv,
                             shell=True)
    stdout_all = ""
    stderr_all = ""
    while True:
        loop_idx = 0
        stdout_line = popen.stdout.readline()
        stderr_line = popen.stderr.readline()

        if loop_idx == 0:
            print("""<filter-progress>{}</filter-progress>""".format(first_output_progress))
            sys.stdout.flush()

        if stdout_line:
            stdout_all += "run_command stdout: " + stdout_line

        if stderr_line:
            stderr_all += "run_command stderr: " + stderr_line if not "MRMLIDImageIO" in stderr_line else ""

        if not stdout_line and not stderr_line and popen.poll() is not None:
            break

        # check if tqdm sent a progress line (via stderr)
        pattern = r"[\s]*([\d]+)%\|"
        match = re.findall(pattern, stderr_line)
        if match:
            # send progress bar update to stdout
            tqdm_progress = float(match[0]) / 100.0
            relative_progress = (tqdm_end - tqdm_start) * tqdm_progress + tqdm_start
            print("""<filter-progress>{}</filter-progress>""".format(relative_progress))
            sys.stdout.flush()

        loop_idx += 1

    popen.stderr.close()
    popen.stdout.close()
    return_code = popen.wait()

    print("run_command stdout:", flush=True)
    print(stdout_all, flush=True)
    print("run_command stderr:", file=sys.stderr, flush=True)
    print(stderr_all, file=sys.stderr, flush=True)

    # send progress bar completed to stdout
    #print("""<filter-end><filter-name>TestFilter</filter-name><filter-time>10</filter-time></filter-end>""")
    #sys.stdout.flush()

    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


# add this class so the extension wizard can import the scripted CLI
class run_command(ScriptedLoadableModule):
    pass