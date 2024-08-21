import time
import math
import yaml
import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import kendalltau

from training.model import GNN, GNNHetero
from training.dataset import load_data
from training.utils import does_path_exist, copy_file, plot_and_save_loss


parser = argparse.ArgumentParser(description="Train the GCN model")
parser.add_argument(
    "name",
    type=str,
    help="Name of the experiment/Training. Results will be saved in training/results/<name>",
)
args = parser.parse_args()

does_path_exist(args.name)

results = f"training/results/{args.name}"
model_path = f"training/model.py"
train_path = f"training/train.py"
params_path = f"training/params.yaml"

copy_file(model_path, f"{results}/model.py")
copy_file(train_path, f"{results}/train.py")
copy_file(params_path, f"{results}/params.yaml")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_PARAMS = yaml.safe_load(open(params_path))


torch.manual_seed(1)
print(f"Training on {DEVICE}")


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)

    mean_loss = []
    for batch_idx, data in enumerate(loop):
        data = data.to(DEVICE)

        output = model(data).squeeze(1)
        loss = loss_fn(output, data.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        mean_loss.append(loss.item())

    train_loss = math.sqrt(sum(mean_loss) / len(mean_loss))
    return train_loss


def validation_fn(test_loader, model, loss_fn, epoch):
    mean_loss = []
    for data in test_loader:
        data = data.to(DEVICE)

        output = model(data).squeeze(1)
        loss = loss_fn(output, data.y)
        mean_loss.append(loss.item())

    validation_set_loss = math.sqrt(sum(mean_loss) / len(mean_loss))
    print(
        f"[{epoch+1}/{TRAINING_PARAMS['EPOCHS']}] Validation Loss is {validation_set_loss}"
    )
    return validation_set_loss


def test_fn(test_loader, model):
    ground_truth_latency_list = []
    predicted_latency_list = []

    for data in test_loader:
        data = data.to(DEVICE)

        output = model(data).squeeze(1)

        ground_truth_latency_list.append(data.y.item())
        predicted_latency_list.append(output.item())

    tau, _ = kendalltau(ground_truth_latency_list, predicted_latency_list)
    return tau


def initialize_model(model, dataloader):
    """Necessary since GraphConv is lazily initialized"""
    data = next(iter(dataloader))
    model(data)


def get_metadata(dataset_path):
    from training.dataset import CustomDataset

    dataset = CustomDataset(dataset_path, is_hetero=True)
    data = dataset[0]

    return data.metadata()


def main():
    EPOCHS = TRAINING_PARAMS["EPOCHS"]
    DATA_DIR = TRAINING_PARAMS["DATA_DIR"]
    BATCH_SIZE = TRAINING_PARAMS["BATCH_SIZE"]
    INPUT_FEATURES = TRAINING_PARAMS["INPUT_FEATURES"]
    WEIGHT_DECAY = TRAINING_PARAMS["WEIGHT_DECAY"]
    LOAD_MODEL = TRAINING_PARAMS["LOAD_MODEL"]
    MODEL_PATH = TRAINING_PARAMS["MODEL_PATH"]
    SAVE_RESULTS = f"training/results/{args.name}"
    HIDDEN_CHANNELS = TRAINING_PARAMS["HIDDEN_CHANNELS"]
    IS_HETERO = TRAINING_PARAMS["IS_HETERO"]

    start_time = time.time()

    torch.manual_seed(0)

    train_loader, valid_loader = load_data(
        DATA_DIR, is_hetero=IS_HETERO, batch_size=BATCH_SIZE, validation_split=0.1
    )

    test_loader, _ = load_data(
        f"{DATA_DIR}/test", is_hetero=IS_HETERO, batch_size=1, validation_split=0.0
    )

    if IS_HETERO:
        print(f"\n----Heterogenous GNN----")
        metadata = get_metadata(DATA_DIR)
        model = GNNHetero(hidden_channels=HIDDEN_CHANNELS, metadata=metadata).to(DEVICE)

    elif not IS_HETERO:
        print(f"\n----Homogenous GNN----")
        model = GNN(hidden_channels=5).to(DEVICE)

    initialize_model(model, test_loader)

    print(
        f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    learning_rate = 0.01  # 5e-4

    if LOAD_MODEL:
        model_state_dict = torch.load(MODEL_PATH)
        model.load_state_dict(model_state_dict)
        print(f"\nPre-Trained Wieghts Loaded\n")

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.MSELoss().to(DEVICE)

    valid_loss_list = []
    train_loss_list = []
    test_metric_list = []
    for epoch in range(EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        valid_loss = validation_fn(valid_loader, model, loss_fn, epoch)

        test_metric = test_fn(test_loader, model)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        test_metric_list.append(test_metric)
        plot_and_save_loss(
            train_loss_list, valid_loss_list, test_metric_list, args.name
        )

        if (epoch + 1) % 50 == 0 or (epoch + 1) == 1:
            torch.save(model, f"{SAVE_RESULTS}/LatNet_{epoch+1}.pth")
            torch.save(
                model.state_dict(), f"{SAVE_RESULTS}/LatNet_{epoch+1}_state_dict.pth"
            )
            end_time = time.time()
            time_elapsed = (end_time - start_time) / 60
            print(f"\nTraining Time: {time_elapsed} minutes\n")

    torch.save(model.state_dict(), f"{SAVE_RESULTS}/LatNet_state_dict.pth")
    torch.save(model, f"{SAVE_RESULTS}/LatNet_final.pth")
    end_time = time.time()
    time_elapsed = (end_time - start_time) / 60

    print(f"\nFinal Training Time: {time_elapsed} minutes\n")


if __name__ == "__main__":
    main()
