# RANO2.0-assist

The RANO2.0-assist is an interactive tool for Response Assessment in Neuro-Oncology (RANO). It is based on the RANO 2.0
guidelines and is designed to assist in the evaluation of glioma. The tool is implemented as a 3D Slicer extension and
provides a user-friendly interface for annotating and measuring tumor response in MRI scans. The pipeline includes
the following steps:
1. **Automatic Segmentation**: The tool uses deep learning models to automatically segment the tumor regions in the MRI scans.
2. **Lesion Matching**: Lesions are matched across different time points to assess changes longitudinally.
2. **Automatic 2D Measurements**: The tool provides automatic measurements of the segmented tumor regions, including
   the calculation of the bidimensional product.
3. **Manual Adjustments**: Users can manually adjust the automatically placed line pairs, add new line pairs, and
   remove unwanted ones.
4. **Response Assessment**: The tool provides a summary of the measurements and allows users to assess the response
   according to the RANO 2.0 guidelines, considering the bidimensional product as well as clinical criteria such as 
   steroid use and clinical status.
5. **Report creation:** The tool generates a report summarizing the measurements and response assessment, which can be 
   saved in PDF format.

## Installation
RANO2.0-assist can be installed as a local 3D Slicer extension, or it can be run in a Docker container. The Docker
container includes all the necessary dependencies and can be run on any machine with Docker installed that supports
GPU acceleration. The local installation requires 3D Slicer and the necessary Python packages to be installed.

### Local installation

#### Requirements
Follow the links to install the required software:
- [3D Slicer](https://download.slicer.org/) (tested with version 5.8.1)
- [PyTorch](https://pytorch.org/get-started/locally/) (tested with version 2.4)

The following Python packages can be installed using pip. The versions listed below are the ones that have been tested with this extension.
- numpy (tested with version 2.0.2) (pip install numpy)
- scikit-image (tested with version 0.24.0) (pip install scikit-image)
- numba (tested with version 0.60.0) (pip install numba)
- nibabel (tested with version 5.3.2) (pip install nibabel)
- tqdm (tested with version 4.67.1) (pip install tqdm))
- pyyaml (tested with version 6.0.2) (pip install pyyaml)
- 
- MONAI (install from https://github.com/aaronkujawa/MONAI/tree/rano) (pip install git+https://github.com/aaronkujawa/MONAI.git@rano)
- pytorch-ignite (tested with version pytorch-ignite 0.5.2) (pip install pytorch-ignite)
- tensorboard (tested with version 2.19.0) (pip install tensorboard)
- ANTsPy (tested with version 0.5.4) (pip install antspyx
- HD-BET (requires this specific commit) (pip install git+https://github.com/MIC-DKFZ/HD-BET@3e3d2a5993e8288f2eae928744ffb496cfc7c651)

#### Segmentation models
The segmentation models need to be downloaded from (TODO: add link to Zenodo models) and placed in the following directory:
```
dynunet_pipeline/data/tasks
```
such that the directory structure looks like this:
```
└──rano2.0-assist
   └──dynunet_pipeline/data/tasks`
      ├── task4000_brats24
      ├── task4001_brats24allmod
      └── task...
```

#### Test data
The test data is also available on Zenodo (TODO: add link to Zenodo test data). The test data should be placed in the
following directory:
```
data/input_data
```

#### Add the extension to 3D Slicer

To add the extension to 3D Slicer, follow these steps:
1. Open 3D Slicer and select "Developer Tools" --> "Extension Manager" in the Modules drop down menu or search for
   "Extension Manager" using the search tool.

<img src="_static/select_extension_wizard.png" alt="drawing" width="300"/>

2. In the Extension Manager, click "Select Extension" and select the "rano2.0-assist" repository folder.

3. Confirm the import of the two modules.

<img src="_static/confirm_module_import.png" alt="drawing" width="200"/>

4. Start the RANO module by selecting "Tools" --> "RANO" in the Modules drop down menu or search for
   "RANO" using the search tool.

### Installation with Docker
For Docker installation, follow instructions here: [docker/README.md](../../docker/README.md).


### Usage

#### Running the tool
- Start 3D Slicer

- Start the RANO module by selecting "Tools" --> "RANO" in the Modules drop down menu or search for
  "RANO" using the search tool

#### Loading the image data

##### 1. Loading the test data
- Loading the test data ist only available in developer mode. To enable developer mode, go to the "Edit" menu and select 
"Application Settings". 

- Under "Developer" tab, check the "Enable developer mode" checkbox. Restart of 3D Slicer is required.

- At the top of the RANO module, you will find the "Reload & Test" box, which is for developer use only.

- The "Test Cases" box allow you to load the test data saved under `data/test_data' in the repository. The folder structure
is as follows:
```
└── rano2.0-assist
    └── data
        └── test_data
            ├── KCH-internal
            │   ├── Patient_003
            │   ├── Patient_006
            │   └── ...
            │
            └── MU-Glioma-Post
                ├── PatientID_003
                ├── PatientID_004
                └── ...
```

- Select the patient and time points in the Test Cases Box and click "Load" to load the data. 

<img src="_static/testCasesBox.png" alt="drawing" width="300"/>

This will load all required scans in the "Inputs" box

##### 2. Loading your own data
- Drag and drop the MRI scans onto the 3D Slicer window. 3D Slicer will open a dialog to load the images.

- Alternatively you can follow the "Add Data" or "Add Dicom Data" buttons in the "Add Data" box. 

- After the images are loaded, select the volumes in the "Inputs" box. Note that all visible input channels
are required for automatic segmentation.


#### Automatic segmentation

4. Automatic segmentation: make sure the "Affine registration" checkbox is checked
and the Input is skull-stripped checkbox is unchecked. Select the model "task4001" and
"Create new segmentation" under "Output Segmentation":

<img src="_static/autoSegBox.png" alt="drawing" width="300"/>

Click "Calculate Segmentation" to run the segmentation model.

5. In the "Automatic 2D Measurements" box choose one of the predicted segments
into which the orthogonal line pairs are to be placed: