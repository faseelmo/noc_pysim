import time
import math
import yaml
import argparse
import subprocess

from tqdm               import tqdm
from scipy.stats        import kendalltau

import torch
import torch.nn         as nn
import torch.optim      as optim

from training.model     import GNN, GNNHetero, GNNHeteroPooling
from training.dataset   import load_data
from training.utils     import (
                            does_path_exist, 
                            copy_file, 
                            plot_and_save_loss, 
                            print_parameter_count, 
                            get_metadata, 
                            initialize_model
                        )

torch.manual_seed(1)

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


DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_PARAMS = yaml.safe_load(open(params_path))

print(f"\nTraining on {DEVICE}")

def process_batch(data, model, loss_fn, is_pooling_model, has_wait_time):
        
    data    = data.to(DEVICE)
    output  = model(data)
    
    if is_pooling_model:
        loss = loss_fn(output, data.y)

    elif has_wait_time and not is_pooling_model: 
        
        is_task_empty = data['task'].y.numel() == 0
        
        if not is_task_empty:
            target_task         = data['task'].y
            pred_task           = output['task']
            loss_task           = loss_fn(target_task, pred_task)
        
        target_task_depend  = data['task_depend'].y
        pred_task_depend    = output['task_depend']
        loss_task_depend    = loss_fn(target_task_depend, pred_task_depend)
        
        if not is_task_empty:
            loss = loss_task + loss_task_depend
        else: 
            loss = loss_task_depend

    if loss.isnan():
        print("Loss is NaN")
        exit()

    return loss


def train_fn(train_loader, model, optimizer, loss_fn, is_pooling_model, has_wait_time):
    loop        = tqdm(train_loader, leave=True)
    mean_loss   = []

    for batch_idx, data in enumerate(loop):

        loss = process_batch(data, model, loss_fn, is_pooling_model, has_wait_time)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        mean_loss.append(loss.item())

    if is_pooling_model:
        train_loss = math.sqrt(sum(mean_loss) / len(mean_loss))
    else: 
        train_loss = sum(mean_loss) / len(mean_loss)

    return train_loss


def validation_fn(valid_loader, model, loss_fn, is_pooling_model, has_wait_time):
    mean_loss = []

    for data in valid_loader:

        loss = process_batch(data, model, loss_fn, is_pooling_model, has_wait_time)
        mean_loss.append(loss.item())

    if is_pooling_model:
        validation_set_loss = math.sqrt(sum(mean_loss) / len(mean_loss))
    else: 
        validation_set_loss = sum(mean_loss) / len(mean_loss)

    return validation_set_loss

def test_fn(test_loader, model, is_pooling_model, has_wait_time):
    ground_truth_latency_list   = []
    predicted_latency_list      = []

    for data in test_loader:
        data    = data.to(DEVICE)
        output  = model(data)

        if is_pooling_model:
            latency_truth   = data.y.item()
            latency_pred    = output.item()

        elif not has_wait_time and not is_pooling_model: 
            latency_truth   = torch.max(data['task'].y).detach().cpu().numpy()
            latency_pred    = torch.max(output).detach().cpu().numpy() 

        elif has_wait_time and not is_pooling_model:
            if data['task'].y.numel() > 0:
                task_latency_truth = torch.max(data['task'].y).detach().cpu().numpy()
                task_latency_pred = torch.max(output['task']).detach().cpu().numpy()
            else:
                task_latency_truth = 0
                task_latency_pred = 0
        
            if data['task_depend'].y.numel() > 0:
                task_depend_latency_truth = torch.max(data['task_depend'].y).detach().cpu().numpy()
                task_depend_latency_pred = torch.max(output['task_depend']).detach().cpu().numpy()
            else:
                task_depend_latency_truth = 0
                task_depend_latency_pred = 0
        
            latency_truth = max(task_latency_truth, task_depend_latency_truth)
            latency_pred = max(task_latency_pred, task_depend_latency_pred)

        ground_truth_latency_list.append(latency_truth)
        predicted_latency_list.append(latency_pred)

    tau, p_value = kendalltau(ground_truth_latency_list, predicted_latency_list)

    return tau, p_value


def main():

    torch.manual_seed(0)

    EPOCHS          = TRAINING_PARAMS["EPOCHS"]
    DATA_DIR        = TRAINING_PARAMS["DATA_DIR"]
    BATCH_SIZE      = TRAINING_PARAMS["BATCH_SIZE"]
    WEIGHT_DECAY    = TRAINING_PARAMS["WEIGHT_DECAY"]
    LOAD_MODEL      = TRAINING_PARAMS["LOAD_MODEL"]
    MODEL_PATH      = TRAINING_PARAMS["MODEL_PATH"]
    SAVE_RESULTS    = f"training/results/{args.name}"
    HIDDEN_CHANNELS = TRAINING_PARAMS["HIDDEN_CHANNELS"]
    IS_HETERO       = TRAINING_PARAMS["IS_HETERO"]
    LEARNING_RATE   = TRAINING_PARAMS["LEARNING_RATE"]
    DO_POOLING      = TRAINING_PARAMS["DO_POOLING"] 
    NUM_MPN_LAYERS  = TRAINING_PARAMS["NUM_MPN_LAYERS"]
    CREATE_DATASET  = TRAINING_PARAMS["CREATE_DATASET"]
    GEN_COUNT       = TRAINING_PARAMS["GEN_COUNT"]   
    MAX_NODES       = TRAINING_PARAMS["MAX_NODES"]
    HAS_WAIT_TIME   = TRAINING_PARAMS["HAS_WAIT_TIME"]
    HAS_SCHEDULER   = TRAINING_PARAMS["HAS_SCHEDULER"]

    start_time = time.time()

    if CREATE_DATASET:
        script_path = "data/create_training_data.sh"
        results     = subprocess.run(
                        [script_path, str(GEN_COUNT), str(MAX_NODES)])

        continue_prompt = input("Dataset created. Continue with training? (yes/no): ")
        if continue_prompt.lower() != "yes":
            exit()

    train_loader, valid_loader  = load_data(
                                    DATA_DIR, 
                                    is_hetero           = IS_HETERO, 
                                    has_wait_time       = HAS_WAIT_TIME,
                                    has_scheduler_node  = HAS_SCHEDULER,
                                    batch_size          = BATCH_SIZE, 
                                    validation_split    = 0.1)

    test_loader, _              = load_data(
                                    f"{DATA_DIR}/test", 
                                    is_hetero           = IS_HETERO, 
                                    has_wait_time       = HAS_WAIT_TIME,
                                    has_scheduler_node  = HAS_SCHEDULER, 
                                    batch_size          = 1, 
                                    validation_split    = 0.0)

    if IS_HETERO:

        metadata    = get_metadata(DATA_DIR, HAS_WAIT_TIME)

        if DO_POOLING:

            print(f"\nGNNHeteroPooling Model Loaded")
            model   = GNNHeteroPooling(HIDDEN_CHANNELS, NUM_MPN_LAYERS, metadata).to(DEVICE)

        else:

            print(f"\nGNNHetero Model Loaded")
            model   = GNNHetero(HIDDEN_CHANNELS, NUM_MPN_LAYERS, metadata).to(DEVICE)

    elif not IS_HETERO:

        print(f"\nGNN Model Loaded")
        model       = GNN(HIDDEN_CHANNELS, NUM_MPN_LAYERS).to(DEVICE)

    initialize_model(model, test_loader)
    print_parameter_count(model)

    if LOAD_MODEL:

        model_state_dict = torch.load(MODEL_PATH)
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

    for epoch in range(EPOCHS):

        train_loss          = train_fn(train_loader, model, optimizer, loss_fn, DO_POOLING, HAS_WAIT_TIME)
        valid_loss          = validation_fn(valid_loader, model, loss_fn, DO_POOLING, HAS_WAIT_TIME)
        test_metric, pvalue = test_fn(test_loader, model, DO_POOLING, HAS_WAIT_TIME)

        print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {valid_loss}, Kendall's Tau: {test_metric}, P-Value: {round(pvalue,5)}")

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        test_metric_list.append(test_metric)

        plot_and_save_loss(
            train_loss_list, 
            valid_loss_list, 
            test_metric_list, 
            args.name)

        if test_metric > 0.95:
        # if (epoch + 1) % 10 == 0 or (epoch + 1) == 1:

            torch.save(model, f"{SAVE_RESULTS}/LatNet_{epoch+1}.pth")
            torch.save(
                model.state_dict(), 
                f"{SAVE_RESULTS}/LatNet_{epoch+1}_state_dict.pth")

            end_time        = time.time()
            time_elapsed    = (end_time - start_time) / 60

            print(f"\n[Saving mode] Total Training Time: {time_elapsed} minutes\n")

    torch.save(model.state_dict(), f"{SAVE_RESULTS}/LatNet_state_dict.pth")
    torch.save(model, f"{SAVE_RESULTS}/LatNet_final.pth")


if __name__ == "__main__":
    main()