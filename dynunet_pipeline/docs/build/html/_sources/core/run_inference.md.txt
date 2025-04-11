## Run inference (example for Task2120)
Inference can be run with the following script:

    data/tasks/task2120_regnobetprimix/commands/infer.sh

Note: If multiple GPUs are not available, the set multi_gpu=0 in the script.

### Inference arguments
The default inference arguments are saved under `src/args.py`.
These can be overwritten by including task specific arguments saved under `data/tasks/task2120_regnobetprimix/config/infer_args.json`.

For example, if you create a new model checkpoint (.pt file) during training and copied it to
`data/tasks/task2120_regnobetprimix/models/model_fold0`, 
make sure to adjust the following argument in the `infer_args.json` file:

    "checkpoint": "checkpoint_key_metric=0.XXXX.pt"

so that it matches the new model checkpoint name.

The argument `"test_files_dir"` specifies the directory with `.nii.gz` images to perform the inference on.
All paths specified as arguments have to be absolute paths or relative to the task directory 
(here the task directory is: `data/tasks/task2120_regnobetprimix`).

This script will run inference on all images in the `"test_files_dir"` directory and save the predictions in a new directory 
specified with the `"out_dir"` argument (default: `results/inference`).

### Registration to MNI template
The model for Task2120 was trained on images that were registered to the MNI template. Therefore, the test images
need to be registered to the MNI template as well. 
The provided test images for Task2120 are already registered to MNI space. However, if a test image is NOT
registered, the `"reg"` argument needs to be set to `true`.

In addtion, the
`"registration_template_path"` and/or `registration_template_bet_path` argument need to be defined 
to specify a registration template for images with and without brain extraction, respectively:

 * for test images that were not pre-processed with brain extraction, the registration template image should be the
original MNI template: `input/template/MNI_152_mri.nii.gz` (default)

 * for test images that were pre-processed with brain extraction, the registration template image should be the
brain extracted MNI template: `input/template/MNI_152_mri_hd-bet_stripped.nii.gz` (default)

To inform the algorithm that the input images were pre-processed with brain extraction, the `"input_is_bet"` argument 
needs to be set to `true`, otherwise it should be set to `false` (default).

### Brain extraction
If the model was trained on brain extracted images, the test images need to be pre-processed with brain extraction as well.
Task2120 was trained on images without brain extraction, so this step is not required for this task. However, 
if brain-extraction is required, the `"bet"` argument needs to be set to `true` in the `infer_args.json` file.
    
    "bet": true,
    "val_num_workers": 0

The second argument is required because the brain extraction is CUDA based and CUDA with multiprocessing does not work
by default.
This step will happen AFTER the registration to the MNI template, so if both preprocessing steps
(registration and brain extraction) are required, use the `MNI_152_mri.nii.gz` template for registration.

Note that brain-extraction with HD-BET should not be performed on images that have already been brain-extracted, because
this removes brain tissue.
