import os 
import yaml
import numpy as np
import random   
import argparse
import torch

from training.model_with_network     import HeteroGNN
from training.dataset   import load_data
from training.utils     import (
                            does_path_exist, 
                            copy_file, 
                            print_parameter_count, 
                            initialize_model
                        )

from training.train import ( train_fn, 
                            #  train_fn, 
                             train_and_validate, 
                             test_fn,
                             validation_fn ) 


def main():

    parser = argparse.ArgumentParser(description="Train the GCN model")
    parser.add_argument( "name", type=str, help="Results will be saved in training/results/<name>")

    args = parser.parse_args()

    print(f"\nTraining Model without Network")

    model_path      = f"training/model_with_network.py"
    train_path      = f"training/train_with_network.py"
    params_path     = f"training/config/params_with_network.yaml"
    dataset_path    = f"training/noc_dataset.py"

    TRAINING_PARAMS = yaml.safe_load(open(params_path))
    results_path    = TRAINING_PARAMS["RESULTS_DIR"]
    SAVE_PATH       = f"{results_path}/{args.name}"

    print(f"\nSaving Results to {SAVE_PATH}")

    does_path_exist(SAVE_PATH)

    copy_file(model_path,   f"{SAVE_PATH}/model.py")
    copy_file(train_path,   f"{SAVE_PATH}/train.py")
    copy_file(params_path,  f"{SAVE_PATH}/params.yaml")
    copy_file(dataset_path, f"{SAVE_PATH}/dataset.py")

    # Seeds for Reproducibility
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    os.environ["PYTHONHASHSEED"] = str(0)

    # Training Parameters 
    DEVICE = TRAINING_PARAMS["DEVICE"]
    DATA_DIR = TRAINING_PARAMS["DATA_DIR"]
    BATCH_SIZE = TRAINING_PARAMS["BATCH_SIZE"]

    HIDDEN_CHANNELS = TRAINING_PARAMS["HIDDEN_CHANNELS"]
    NUM_MPN_LAYERS = TRAINING_PARAMS["NUM_MPN_LAYERS"]

    LEARNING_RATE = TRAINING_PARAMS["LEARNING_RATE"]
    WEIGHT_DECAY = TRAINING_PARAMS["WEIGHT_DECAY"]  

    EPOCHS = TRAINING_PARAMS["EPOCHS"]
    SAVE_THRESHOLD = TRAINING_PARAMS["SAVE_THRESHOLD"]

    if DEVICE == "cuda":
        if torch.cuda.is_available():
            # Set Seeds for Cuda Reproducibility
            DEVICE = torch.device("cuda")
            torch.use_deterministic_algorithms(True)
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else: 
            raise ValueError("CUDA is not available. Please set DEVICE='cpu' in params.yaml")

    elif DEVICE == "cpu":
        DEVICE = torch.device("cpu")

    else: 
        raise ValueError("DEVICE must be either 'cuda' or 'cpu'")

    print(f"Training on {DEVICE}")

    train_data_dir  = f"{DATA_DIR}/train"
    test_data_dir   = f"{DATA_DIR}/test"

    train_loader, valid_loader  = load_data(
                                    train_data_dir, 
                                    batch_size          = BATCH_SIZE, 
                                    validation_split    = 0.1,
                                    use_noc_dataset     = True,
                                    clasify_task_nodes  = False,
                                )

    test_loader, _              = load_data(
                                    test_data_dir, 
                                    batch_size          = 1, 
                                    validation_split    = 0.0,
                                    use_noc_dataset     = True,
                                    clasify_task_nodes  = False,
                                )

    model = HeteroGNN( hidden_channels=HIDDEN_CHANNELS, 
                       num_mpn_layers=NUM_MPN_LAYERS ).to(DEVICE) 


    initialize_model( model, train_loader, DEVICE )
    print_parameter_count( model )

    loss_fn     = torch.nn.L1Loss().to(DEVICE)
    optimizer   = torch.optim.Adam( model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY )

    train_and_validate( epochs         = EPOCHS, 
                        train_loader   = train_loader, 
                        valid_loader   = valid_loader, 
                        test_loader    = test_loader, 
                        model          = model, 
                        optimizer      = optimizer, 
                        loss_fn        = loss_fn, 
                        device         = DEVICE, 
                        save_path      = SAVE_PATH, 
                        save_threshold = SAVE_THRESHOLD )


if __name__ == "__main__":
    main()
