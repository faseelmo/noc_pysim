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

from training.model     import GNN, GNNHetero, GNNHeteroPooling, HeteroGNN
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

if TRAINING_PARAMS["USE_NOC_DATASET"]:
    copy_file("training/noc_dataset.py", f"{results}/noc_dataset.py")
else: 
    copy_file("training/dataset.py", f"{results}/dataset.py")

print(f"\nTraining on {DEVICE}")

def process_batch(data, model, loss_fn, is_pooling_model, has_wait_time):
        
    data    = data.to(DEVICE)
    output  = model(data)
    
    if is_pooling_model:
        loss = loss_fn(output, data.y)

    elif has_wait_time and not is_pooling_model: 
        
        is_task_empty = data['task'].y.numel() == 0
        
        if not is_task_empty:
            # Some pe graphs have no tasks 
            target_task         = data['task'].y
            pred_task           = output['task']
            loss_task           = loss_fn(target_task, pred_task)

        if 'task_depend' in data:
            target_task_depend  = data['task_depend'].y
            pred_task_depend    = output['task_depend']
            loss_task_depend    = loss_fn(target_task_depend, pred_task_depend)

            if not is_task_empty:
                loss = loss_task + loss_task_depend
        
            else: 
                loss = loss_task_depend

        else: 
            loss = loss_task


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
        
            task_depend_latency_truth = 0
            task_depend_latency_pred = 0

            if 'task_depend' in data:
                if data['task_depend'].y.numel() > 0:
                    task_depend_latency_truth = torch.max(data['task_depend'].y).detach().cpu().numpy()
                    task_depend_latency_pred = torch.max(output['task_depend']).detach().cpu().numpy()
        
            latency_truth = max(task_latency_truth, task_depend_latency_truth)
            latency_pred = max(task_latency_pred, task_depend_latency_pred)

        ground_truth_latency_list.append(latency_truth)
        predicted_latency_list.append(latency_pred)

    tau, p_value = kendalltau(ground_truth_latency_list, predicted_latency_list)

    return tau, p_value


def main():

    torch.manual_seed(0)

    NUM_MPN_LAYERS      = TRAINING_PARAMS["NUM_MPN_LAYERS"]
    HIDDEN_CHANNELS     = TRAINING_PARAMS["HIDDEN_CHANNELS"]
    USE_NOC_DATASET     = TRAINING_PARAMS["USE_NOC_DATASET"]
    IS_HETERO           = TRAINING_PARAMS["IS_HETERO"]
    DO_POOLING          = TRAINING_PARAMS["DO_POOLING"] 
    HAS_WAIT_TIME       = TRAINING_PARAMS["HAS_WAIT_TIME"]
    HAS_SCHEDULER       = TRAINING_PARAMS["HAS_SCHEDULER"]
    USE_HETERO_WRAPPER  = TRAINING_PARAMS["USE_HETERO_WRAPPER"]

    CREATE_DATASET      = TRAINING_PARAMS["CREATE_DATASET"]
    GEN_COUNT           = TRAINING_PARAMS["GEN_COUNT"]   
    MAX_NODES           = TRAINING_PARAMS["MAX_NODES"]

    LEARNING_RATE       = TRAINING_PARAMS["LEARNING_RATE"]
    EPOCHS              = TRAINING_PARAMS["EPOCHS"]
    WEIGHT_DECAY        = TRAINING_PARAMS["WEIGHT_DECAY"]
    BATCH_SIZE          = TRAINING_PARAMS["BATCH_SIZE"]

    LOAD_MODEL          = TRAINING_PARAMS["LOAD_MODEL"]
    MODEL_PATH          = TRAINING_PARAMS["MODEL_PATH"]

    DATA_DIR            = TRAINING_PARAMS["DATA_DIR"]
    SAVE_RESULTS        = f"training/results/{args.name}"
    SAVE_THRESHOLD      = TRAINING_PARAMS["SAVE_THRESHOLD"]


    start_time = time.time()

    if CREATE_DATASET:
        script_path = "data/create_training_data.sh"
        results     = subprocess.run(
                        [script_path, str(GEN_COUNT), str(MAX_NODES)])

        continue_prompt = input("Dataset created. Continue with training? (yes/no): ")
        if continue_prompt.lower() != "yes":
            exit()

    if USE_NOC_DATASET:
        train_data_dir  = f"{DATA_DIR}/simulator/train"
        test_data_dir   = f"{DATA_DIR}/simulator/test"
    else: 
        train_data_dir = f"{DATA_DIR}"
        test_data_dir  = f"{DATA_DIR}/test"


    train_loader, valid_loader  = load_data(
                                    train_data_dir, 
                                    batch_size          = BATCH_SIZE, 
                                    validation_split    = 0.1,
                                    use_noc_dataset     = USE_NOC_DATASET,
                                    is_hetero           = IS_HETERO, 
                                    has_wait_time       = HAS_WAIT_TIME,
                                    has_scheduler_node  = HAS_SCHEDULER,
                                )

    test_loader, _              = load_data(
                                    test_data_dir, 
                                    batch_size          = 1, 
                                    validation_split    = 0.0,
                                    use_noc_dataset     = USE_NOC_DATASET,
                                    is_hetero           = IS_HETERO, 
                                    has_wait_time       = HAS_WAIT_TIME,
                                    has_scheduler_node  = HAS_SCHEDULER, 
                                )

    if IS_HETERO:

        metadata    = get_metadata(test_data_dir, HAS_WAIT_TIME, USE_NOC_DATASET)

        if DO_POOLING:

            model   = GNNHeteroPooling(HIDDEN_CHANNELS, NUM_MPN_LAYERS, metadata).to(DEVICE)
            print(f"\nGNNHeteroPooling Model Loaded with {NUM_MPN_LAYERS} MPN Layers and {HIDDEN_CHANNELS} Hidden Channels")

        elif USE_HETERO_WRAPPER: 

            model = HeteroGNN(HIDDEN_CHANNELS, NUM_MPN_LAYERS).to(DEVICE)
            print(f"\nHeteroGNN Model Loaded with {NUM_MPN_LAYERS} MPN Layers and {HIDDEN_CHANNELS} Hidden Channels")

        else:

            model   = GNNHetero(HIDDEN_CHANNELS, NUM_MPN_LAYERS, metadata).to(DEVICE)
            print(f"\nGNNHetero Model Loaded with {NUM_MPN_LAYERS} MPN Layers and {HIDDEN_CHANNELS} Hidden Channels")

    elif not IS_HETERO:

        model       = GNN(HIDDEN_CHANNELS, NUM_MPN_LAYERS).to(DEVICE)
        print(f"\nGNN Model Loaded with {NUM_MPN_LAYERS} MPN Layers and {HIDDEN_CHANNELS} Hidden Channels")

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

    saved_test_metric = []

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

        save_multiple = False
        if epoch % 10 == 0:
            save_multiple = True

        if test_metric > SAVE_THRESHOLD or save_multiple:

            test_metric = int( round(test_metric, 2) * 100 )

            if not save_multiple:
                if test_metric in saved_test_metric: 
                    continue

            torch.save(model, f"{SAVE_RESULTS}/LatNet_{test_metric}_{epoch+1}.pth")
            torch.save(
                model.state_dict(), 
                f"{SAVE_RESULTS}/LatNet_{test_metric}_{epoch+1}.pth")

            end_time        = time.time()
            time_elapsed    = (end_time - start_time) / 60

            saved_test_metric.append(test_metric)

            print(f"\n[Saving mode] Total Training Time: {time_elapsed} minutes\n")

    torch.save(model.state_dict(), f"{SAVE_RESULTS}/LatNet_state_dict.pth")
    torch.save(model, f"{SAVE_RESULTS}/LatNet_final.pth")


if __name__ == "__main__":
    main()
