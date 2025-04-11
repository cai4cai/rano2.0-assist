## Run training (example for Task2120)
Training can be run by executing the script:

    data/tasks/task2120_regnobetprimix/commands/train.sh

Note: If multiple GPUs are not available, the set multi_gpu=0 in the script.


This command will run training and save model checkpoints under:

    data/tasks/task2120_regnobetprimix/results/training/model_fold0

If you want to use the trained model for inference, copy this folder to: 

    data/tasks/task2120_regnobetprimix/results/models

But for now we will continue with the pretrained model that is currently saved under that location.