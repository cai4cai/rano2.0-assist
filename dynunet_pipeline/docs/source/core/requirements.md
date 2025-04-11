## Requirements

### Operating system
The tool has been tested with Python 3.8 and 3.10 on Ubuntu 20.04 and Windows 10.

### Python libraries
The following Python libraries are required:

* PyTorch (Installation instructions: [https://pytorch.org/](https://pytorch.org/))
* PyTorch-Ignite (install with: `pip install pytorch-ignite`)
* MONAI 
  * (install from this GitHub fork with: `pip install git+https://github.com/aaronkujawa/MONAI.git@medtronic`)
* NumpPy (install with `pip install numpy`)
* Nibabel (install with `pip install nibabel`)
* tqdm (install with `pip install tqdm`)

### HD-BET
HD-BET is used for brain extraction if the model was trained on skull-stripped images. To install HD-BET directly 
from GitHub, with:
    
    pip install git+https://github.com/MIC-DKFZ/HD-BET 

or for installation from a local copy of the repository:

    git clone https://github.com/MIC-DKFZ/HD-BET.git
    cd HD-BET 
    pip install -e .
    cd ..

The first time HD-BET is run after installation, it should automatically download model weights 
from zenodo.org. If there is a problem and the error "0.model not found", you can download it 
manually [here](https://zenodo.org/record/2540695/#.Y_27oYDP18I) and place the model weights in the location requested in the error message. 

### ANTs 
Furthermore, preprocessing involves image registration which relies on routines available in 
[Advanced Normalization Tools (ANTs)](http://stnava.github.io/ANTs/). The two required pre-compiled binaries 
for Linux/Windows/MacOS can be downloaded 
[here](https://emckclac.sharepoint.com/:f:/r/sites/MT-BMEIS-M-F-P/Shared%20Documents/General/Models%20and%20Data/dynunet_pipeline/ANTs_binaries?csf=1&web=1&e=LLu4eI).
These have to be made accessible from the command line via the commands "antsRegistration" and
"antsApplyTransforms" from the command line.

On Linux, this can be achieved by 
placing the two files under `/usr/bin` (or any other path that is part of the
PATH environment variable) and making them executable with `chmod +x <filename>`.

Alternatively, this can be achieved by 
creating two new text files named "antsRegistration" and "antsApplyTransforms"
under any folder that is part of the PATH environment variable,
with the following file contents:
    
    `#! /bin/sh
    exec <...>/antsRegistration.glnxa64 "$@"`

and
    
    `#! /bin/sh
    exec <...>/antsApplyTransforms.glnxa64 "$@"`

where `<...>` is the absolute path to the folder containing the downloaded Linux binaries.


