def does_path_exist(model_name):
    import os
    import yaml

    training_params = yaml.safe_load(open("training/params.yaml"))
    dir_path = os.path.join(training_params["RESULTS_DIR"], model_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Folder '{dir_path}' created.")
    else:
        print(f"Folder '{dir_path}' already exists.")
        continue_prompt = input("Do you want to continue? (yes/no): ")
        if continue_prompt.lower() != "yes":
            exit()


def copy_model_to_results(model_name):
    import shutil
    import os

    path = os.path.join("training", "results", model_name)
    shutil.copy2("training/model.py", path)
    print(f"model.py copied to {path}")


def plot_and_save_loss(train_loss, valid_loss, model_name):
    import matplotlib.pyplot as plt
    import pickle

    epochs = range(1, len(train_loss) + 1)
    plt.yscale("log")
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, valid_loss, label="Validation Loss")
    plt.title("Training and Validation Loss (Log Scale)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    plt.savefig(f"training/results/{model_name}/validation_plot_log.png")
    plt.clf()

    loss_dict = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
    }
    with open(f"training/results/{model_name}/loss_dict.pkl", "wb") as file:
        pickle.dump(loss_dict, file)
