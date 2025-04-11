# 3D-Slicer plugin

A 3D-Slicer extension is available with which images can be loaded, inference can be run, and the output
parcellation can be viewed and 3D-rendered in 3D-Slicer. 

## Requirements
* 3D-Slicer (>=version 5.2.1: download [here](https://download.slicer.org/))

## Installation

1. Follow steps 1-3 of the core package instructions: {ref}`core/model_and_data_prep:Preparation of Model and Inference Data`
2. Follow 3D-Slicer installation instructions
3. Start 3D-Slicer
4. Go to Edit --> Application Settings --> Modules
5. Under "Add additional module paths" add the paths to the repository folders
`extensions/3DSlicer-extension/FastParcellation/Full_Brain_Parcellation` and
`extensions/3DSlicer-extension/FastParcellation/run_command`

    ![Add the module paths](Screenshot-add-module-paths.png)

6. Confirm the 3D-Slicer restart


## Run inference
1. After the restart, select the "Full Brain Parcellation" module as shown in this
screenshot:

    ![Start the module](Screenshot-select-FBP-module.png)

2. Load an image into 3D-Slicer. This can be done by drag-and-dropping an image file on the 3D-Slicer module window. A
new window should pop up. Make sure the "Description" of the file is "Volume" and confirm the load with "OK". 
3. In the Fast Brain Parcellation module, select the model and the input image.
- If the input image is already registered to the MNI template, uncheck the "Affine registration" checkbox.
- If the input image is skull-stripped, check the "Input is skull-stripped" checkbox. This will make sure, the correct
registration template is used, and further brain extraction is skipped.

   Then, create a new segmentation under the Segmentation drop-down menu:

   ![Setup the inputs/outputs](Screenshot-module-setup.png)

4. Finally, start the inference with "Apply". After about a minute, the segmentation should be visible in the 3D-Slicer.
In the "Data" module, the individual segments of the segmentation can be inspected and be made visible/invisible.

   ![Inspect the output](Screenshot-3d-rendering.png)

All steps are shown in this GIF:    

   ![slicer-demo](slicer-demo.gif)
