import os
debug = False
# set the multi_gpu flag depending on whether "LOCAL_RANK" is in the environment variables
# this depends on whether the script is run using torchrun or not
multi_gpu_flag = True if "LOCAL_RANK" in os.environ else False
