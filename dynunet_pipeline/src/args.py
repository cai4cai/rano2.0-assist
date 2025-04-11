import json
import os
import config


def add_default_training_arguments(args):
    """
    Adds default training arguments for missing arguments to the args Namespace
    :param args: Namespace object containing the input arguments
    :return: Namespace object with default values added
    """

    default_training_args = {
        # path to the training dataset
        'dataset_dir': 'input/dataset',
        # path to the dataset config file
        'modalities_path': 'config/modalities.json',
        'splits_path': 'config/splits.json' if not config.debug else 'config/splits_debug.json',
        'label_names_path': 'config/label_names.json',
        'train_args_path': 'config/train_args.json',
        'models_root_dir': 'results/training',  # path to the directory where the trained models are saved
        'expr_name': 'experiment',
        'fold': 0,  # 0-4, determines which fold of the training set is used for validation
        'use_prior': False,
        'train_num_workers': 0,
        'val_num_workers': 32,
        'interval': 5,
        'eval_overlap': 0.5,
        'sw_batch_size': 4,
        'window_mode': 'gaussian',
        'pos_sample_num': 1,
        'neg_sample_num': 2,
        'cache_rate': 1.0,
        'learning_rate': 0.01,
        'max_epochs': 1000,
        'checkpoint': None,  # resume training from this checkpoint
        'resume_latest_checkpoint': False,  # resume training from latest checkpoint
        'amp': True,
        'lr_decay': True,
        'tta_val': True,
        'batch_dice': False,
        'determinism_flag': False,
        'determinism_seed': 0,
        'local_rank': int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0

    }

    # add default values for missing parameters
    for k, v in default_training_args.items():
        if k not in args:
            setattr(args, k, v)

    return args


def add_default_inference_arguments(args):
    """
    Adds default inference arguments for missing arguments to the args Namespace
    :param args: Namespace object containing the input arguments
    :return: Namespace object with default values added
    """

    default_inference_args = {
        'out_dir': 'results/inference',
        'reg': False,
        'registration_template_path': '',
        'registration_template_bet_path': '',
        'val_num_workers': 0,
        'eval_overlap': 0.5,
        'sw_batch_size': 1,
        'window_mode': 'gaussian',
        'cache_rate': 1.0,
        'checkpoint': None,
        'amp': True,
        'tta_val': True,
        'bet': False,
        'input_is_bet': False,
        'local_rank': int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    }

    # add default values for missing parameters
    for k, v in default_inference_args.items():
        if k not in args:
            setattr(args, k, v)

    return args


def add_args_from_file(args, args_file):
    """
    Adds arguments from a json file to the args Namespace
    :param args: Namespace object containing the input arguments
    :param args_file: path to the json file containing the arguments
    :return: Namespace object with arguments added
    """
    print(f"Loading additional arguments from {os.path.abspath(args_file)}")
    with open(args_file, "r") as f:
        task_args_orig = json.load(f)
    # remove debug parameters
    task_args = {k: v for k, v in task_args_orig.items() if not "_debug" in k}

    if config.debug:
        # replace parameters with debug parameters
        for k, v in task_args_orig.items():
            if "_debug" in k:
                k_no_debug = k.replace('_debug', '')
                print(f"Debugging active... replacing {k_no_debug} = {task_args[k_no_debug]} with {v}")
                task_args[k_no_debug] = v

    # add task parameters to args
    for k, v in task_args.items():
        if k not in args:
            setattr(args, k, v)

    return args


def add_training_args(args):
    """
    Adds training arguments to the args Namespace
    :param args: Namespace object containing the input arguments
    :return: Namespace object with arguments added
    """
    args = add_args_from_file(args, args.args_file)
    # add default values for missing parameters
    args = add_default_training_arguments(args)
    return args


def add_inference_args(args):
    """
    Adds inference arguments to the args Namespace
    :param args: Namespace object containing the input arguments
    :return: Namespace object with arguments added
    """
    args = add_args_from_file(args, args.args_file)
    # add default values for missing parameters
    args = add_default_inference_arguments(args)

    # add arguments saved during training
    train_args_out_path = os.path.join(args.model_dir, "train_args_out.json")
    args = add_args_from_file(args, train_args_out_path)
    return args
