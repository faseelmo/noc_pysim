import os 
import time
import math
import yaml
import numpy as np
import random   
import argparse
import subprocess

from tqdm               import tqdm
from scipy.stats        import kendalltau

import torch
import torch.nn         as nn
import torch.optim      as optim

from training.model     import HeteroGNN
from training.dataset   import load_data
from training.utils     import (
                            does_path_exist, 
                            copy_file, 
                            plot_and_save_loss, 
                            print_parameter_count, 
                            initialize_model
                        )


parser  = argparse.ArgumentParser(description="Train the GCN model")

parser.add_argument(
    "name",
    type=str,
    help="Name of the experiment/Training." 
    "Results will be saved in training/results/<name>")

args    = parser.parse_args()

does_path_exist(args.name)

results     = f"training/results/{args.name}"
model_path  = f"training/model.py"
train_path  = f"training/train.py"
params_path = f"training/params.yaml"

copy_file(model_path, f"{results}/model.py")
copy_file(train_path, f"{results}/train.py")
copy_file(params_path, f"{results}/params.yaml")

TRAINING_PARAMS = yaml.safe_load(open(params_path))

copy_file("training/noc_dataset.py", f"{results}/noc_dataset.py")


def process_batch(data, model, loss_fn, device):
        
    data    = data.to(device)
    output  = model(data)
        
    target_task = data['task'].y.to(device)
    pred_task   = output['task']
    loss_task   = loss_fn(target_task, pred_task)

    return loss_task


def train_fn(train_loader, model, optimizer, loss_fn, device):
    loop        = tqdm(train_loader, leave=True)
    mean_loss   = []

    model.to(device)
    for batch_idx, data in enumerate(loop):

        loss = process_batch(data, model, loss_fn, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        mean_loss.append(loss.item())

        train_loss = sum(mean_loss) / len(mean_loss)

    return train_loss


def validation_fn(valid_loader, model, loss_fn, device):
    mean_loss = []

    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            loss = process_batch(data, model, loss_fn, device)
            mean_loss.append(loss.item())

    validation_set_loss = sum(mean_loss) / len(mean_loss)

    return validation_set_loss

def test_fn(test_loader, model):
    ground_truth_latency_list   = []
    predicted_latency_list      = []

    model.to('cpu')
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data    = data.to('cpu')
            output  = model(data)

            task_latency_truth = torch.max(data['task'].y).detach().cpu().numpy()
            task_latency_pred = torch.max(output['task']).detach().cpu().numpy()
        
            ground_truth_latency_list.append(task_latency_truth)
            predicted_latency_list.append(task_latency_pred)

    tau, p_value = kendalltau(ground_truth_latency_list, predicted_latency_list)

    return tau, p_value

def save_model(model, epoch, results_dir, test_metric, suffix=""):
    test_metric_int = int( round(test_metric, 3) * 100 )

    model_filename = f"{results_dir}/models/LatNet_{test_metric_int}_{epoch+1}_{suffix}.pth"
    torch.save(model.state_dict(), model_filename)


def main():

    # Seeds for Reproducibility
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(0)

    # Training Parameters  
    NUM_MPN_LAYERS      = TRAINING_PARAMS["NUM_MPN_LAYERS"]
    HIDDEN_CHANNELS     = TRAINING_PARAMS["HIDDEN_CHANNELS"]
    DEVICE              = TRAINING_PARAMS["DEVICE"]

    LEARNING_RATE       = TRAINING_PARAMS["LEARNING_RATE"]
    EPOCHS              = TRAINING_PARAMS["EPOCHS"]
    WEIGHT_DECAY        = TRAINING_PARAMS["WEIGHT_DECAY"]
    BATCH_SIZE          = TRAINING_PARAMS["BATCH_SIZE"]

    LOAD_MODEL          = TRAINING_PARAMS["LOAD_MODEL"]
    MODEL_PATH          = TRAINING_PARAMS["MODEL_PATH"]

    DATA_DIR            = TRAINING_PARAMS["DATA_DIR"]
    SAVE_RESULTS        = f"training/results/{args.name}"
    SAVE_THRESHOLD      = TRAINING_PARAMS["SAVE_THRESHOLD"]


    if DEVICE == "cuda":
        if torch.cuda.is_available():
            # Set Seeds for Cuda Reproducibility
            DEVICE = torch.device("cuda")
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

    print(f"\nTraining on {DEVICE}")


    start_time = time.time()

    train_data_dir  = f"{DATA_DIR}/simulator/train"
    test_data_dir   = f"{DATA_DIR}/simulator/test"


    train_loader, valid_loader  = load_data(
                                    train_data_dir, 
                                    batch_size          = BATCH_SIZE, 
                                    validation_split    = 0.1,
                                    use_noc_dataset     = True,
                                )

    test_loader, _              = load_data(
                                    test_data_dir, 
                                    batch_size          = 1, 
                                    validation_split    = 0.0,
                                    use_noc_dataset     = True,
                                )

    model = HeteroGNN(HIDDEN_CHANNELS, NUM_MPN_LAYERS).to(DEVICE)
    print(f"\nHeteroGNN Model Loaded with {NUM_MPN_LAYERS} MPN Layers and {HIDDEN_CHANNELS} Hidden Channels")

    initialize_model(model, test_loader, DEVICE)
    print_parameter_count(model)

    if LOAD_MODEL:

        model_state_dict = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(model_state_dict)
        print(f"\nPre-Trained Wieghts Loaded\n")

    loss_fn     = nn.MSELoss().to(DEVICE)
    optimizer   = optim.Adam(
                    model.parameters(), 
                    lr=LEARNING_RATE, 
                    weight_decay=WEIGHT_DECAY)

    valid_loss_list     = []
    train_loss_list     = []
    test_metric_list    = []
    saved_test_metric = []

    best_metric = 0

    for epoch in range(EPOCHS):

        train_loss          = train_fn(train_loader, model, optimizer, loss_fn, device=DEVICE)
        valid_loss          = validation_fn(valid_loader, model, loss_fn, device=DEVICE)
        test_metric, pvalue = test_fn(test_loader, model)

        print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {valid_loss}, Kendall's Tau: {test_metric}, P-Value: {round(pvalue,5)}")

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        test_metric_list.append(test_metric)

        plot_and_save_loss(
            train_loss_list, 
            valid_loss_list, 
            test_metric_list, 
            args.name)

        rounded_metric = round(test_metric, 3)
        saved_flag = False

        if test_metric > SAVE_THRESHOLD:

            if test_metric > best_metric and rounded_metric not in saved_test_metric:
                best_metric = test_metric
                saved_test_metric.append(rounded_metric)
                save_model(model, epoch, SAVE_RESULTS, test_metric, suffix="best") 
                saved_flag = True

        if epoch % 10 == 0 :

            end_time        = time.time()
            time_elapsed    = (end_time - start_time) / 60

            if epoch != 0:
                print(f"Training Time: {time_elapsed:.2f} minutes")

            if not saved_flag:
                save_model(model, epoch, SAVE_RESULTS, test_metric, suffix="interval")

    save_model(model, epoch, SAVE_RESULTS, test_metric, suffix="last")

if __name__ == "__main__":
    main()
