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


def plot_and_save_loss(train_loss, valid_loss, test_metric, model_name):
    import matplotlib.pyplot as plt
    import pickle

    epochs = range(1, len(train_loss) + 1)

    fig, ax1 = plt.subplots()

    ax1.set_yscale("log")
    ax1.plot(epochs, train_loss, label="Training Loss")
    ax1.plot(epochs, valid_loss, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Log Scale)")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(epochs, test_metric, "g-", label="Kendall's Tau")
    ax2.set_ylabel("Kendall's Tau")
    ax2.tick_params(axis="y")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.title("Training and Validation Loss (Log Scale) with Kendall's Tau")
    plt.savefig(
        f"training/results/{model_name}/validation_plot_log_with_kendalls_tau.png"
    )
    plt.clf()

    loss_dict = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "kendalls_tau": test_metric,
    }
    with open(
        f"training/results/{model_name}/loss_dict_with_kendalls_tau.pkl", "wb"
    ) as file:
        pickle.dump(loss_dict, file)
