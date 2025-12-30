#!/usr/bin/env python-real

import os
import sys
import subprocess
import re
from pathlib import Path
from slicer.ScriptedLoadableModule import ScriptedLoadableModule
import threading
import queue

if __name__ == '__main__':
    # define progress bar steps as fraction of 1
    command_received_progress = 0.1
    first_output_progress = 0.2
    tqdm_start = command_received_progress + first_output_progress
    tqdm_end = 1.0

    command = sys.argv[1]

    # send progress bar start line
    print("""<filter-start><filter-name>TestFilter</filter-name><filter-comment>ibid</filter-comment></filter-start>""")
    sys.stdout.flush()
    print(f"<filter-progress>{command_received_progress}</filter-progress>")
    sys.stdout.flush()

    # startup environment
    slicer_path = Path(os.environ["SLICER_HOME"]).resolve()
    PATH_without_slicer = os.pathsep.join(
        [p for p in os.environ["PATH"].split(os.pathsep) if not slicer_path in Path(p).parents]
    )
    startupEnv = {
        "PATH": PATH_without_slicer,
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "USERPROFILE": os.environ.get("USERPROFILE", "")
    }

    popen = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=startupEnv,
        shell=True
    )

    stdout_all = ""
    stderr_all = ""
    progress_queue = queue.Queue()


    # Reader for stdout
    def read_stdout(pipe):
        nonlocal_stdout = []
        loop_idx = 0
        for line in iter(pipe.readline, ''):
            #print("STDOUT:", line.strip())
            if loop_idx == 0:
                # enqueue initial progress update
                progress_queue.put(first_output_progress)
            if line:
                nonlocal_stdout.append("run_command stdout: " + line)
            loop_idx += 1
        pipe.close()
        # merge back into global
        nonlocal_stdout_str = "".join(nonlocal_stdout)
        if nonlocal_stdout_str:
            progress_queue.put(("stdout_all", nonlocal_stdout_str))


    # Reader for stderr
    def read_stderr(pipe):
        nonlocal_stderr = []
        for line in iter(pipe.readline, ''):
            #print("STDERR:", line.strip())
            if line and "MRMLIDImageIO" not in line:
                nonlocal_stderr.append("run_command stderr: " + line)
            # check for tqdm progress
            match = re.findall(r"[\s]*([\d]+)%\|", line)
            if match:
                tqdm_progress = float(match[0]) / 100.0
                relative_progress = (tqdm_end - tqdm_start) * tqdm_progress + tqdm_start
                progress_queue.put(relative_progress)
        pipe.close()
        nonlocal_stderr_str = "".join(nonlocal_stderr)
        if nonlocal_stderr_str:
            progress_queue.put(("stderr_all", nonlocal_stderr_str))


    # Start threads
    t_out = threading.Thread(target=read_stdout, args=(popen.stdout,))
    t_err = threading.Thread(target=read_stderr, args=(popen.stderr,))
    t_out.start()
    t_err.start()

    # Main loop: poll process and queue
    while popen.poll() is None or not progress_queue.empty():
        try:
            item = progress_queue.get(timeout=0.1)
            if isinstance(item, tuple):
                # accumulate stdout/stderr
                if item[0] == "stdout_all":
                    stdout_all += item[1]
                elif item[0] == "stderr_all":
                    stderr_all += item[1]
            else:
                # progress update
                print(f"<filter-progress>{item}</filter-progress>")
                sys.stdout.flush()
        except queue.Empty:
            pass

    # Wait for threads to finish
    t_out.join()
    t_err.join()

    print("Process finished with return code:", popen.returncode)

    # send progress bar completed
    print("""<filter-end><filter-name>TestFilter</filter-name><filter-time>10</filter-time></filter-end>""")
    sys.stdout.flush()

    if popen.returncode:
        raise subprocess.CalledProcessError(popen.returncode, command)

# add this class so the extension wizard can import the scripted CLI
class run_command(ScriptedLoadableModule):
    pass