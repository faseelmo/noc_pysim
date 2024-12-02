import os
import re
import argparse
import pickle
import pandas as pd

import matplotlib
matplotlib.use("pgf")  # Ensure this is set before importing pyplot
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",  # Use pdflatex
        "text.usetex": True,          # Enable LaTeX rendering
        "font.family":     "serif",
        "font.serif":      [],         # empty entries should cause the usage of the document fonts
        "pgf.preamble": r"""
            \def\mathdefault#1{#1}  % Prevent issues with \mathdefault
        """,
    }
)

def extract_param_from_dir(dir):
    match = re.search(
        r"(?P<conv>\w+)_L(?P<layer>\d+)_C(?P<width>\d+)(?:_A(?P<aggr>\w+))?_(?P<loss>\w+)",
        dir,
    )
    if match:
        conv = match.group("conv")
        layer = int(match.group("layer"))
        width = int(match.group("width"))
        aggr = match.group("aggr") if match.group("aggr") else None
        loss = match.group("loss")
        return conv, layer, width, aggr, loss


def load_loss_file(dir_path):
    file_path = os.path.join(dir_path, "loss.pkl")
    with open(file_path, "rb") as f:
        loss = pickle.load(f)
    return loss


def plot_tau_for_all(data, plot_type, save_for_latex=False, tikz_filename="plot.tex"):
    plt.figure(figsize=(6, 6))

    fontsize = 20

    for entry in data:
        conv = entry["conv"]
        layer = entry["Layer"]
        width = entry["Width"]
        tau = entry["Tau"]

        if plot_type == "conv":
            if conv == "graphconv": 
                label = "GraphConv"
            else: 
                label = f"{conv.upper()}"
        elif plot_type == "loss":
            label = f"{entry['Loss']}"
        elif plot_type == "width_depth":
            label = f"L={layer}, W={width}"
        elif plot_type == "aggr":
            label = f"{entry['Aggr']}"

        plt.plot(range(1, len(tau) + 1), tau, label=label, linewidth=3)

    plt.xlabel("epoch", labelpad=5, fontsize=fontsize)
    plt.ylabel("application tau", labelpad=8, fontsize=fontsize)
    plt.legend(loc="best", fontsize=fontsize*0.6)  

    plt.xticks(fontsize=fontsize*0.8)  # Larger x-axis tick values
    plt.yticks(fontsize=fontsize*0.8)  # Larger y-axis tick values

    plt.grid(alpha=0.5)

    if save_for_latex:
        save_location = f"/home/faseelchemmadan/uni/repo/thesis/plots/{tikz_filename}"
        plt.savefig(save_location)
        print(f"TikZ plot saved to {save_location}")

    else:
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument(
        "--find_best",
        action="store_true",
        help="Find the parameters with the best result",
    )
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    args = parser.parse_args()

    dirs = os.listdir(args.dir)
    data = []

    for dir in dirs:
        conv, layer, width, aggr, loss = extract_param_from_dir(dir)
        loss_data = load_loss_file(f"{args.dir}/{dir}")
        tau = loss_data["kendalls_tau"]
        best_tau = max(tau)
        data.append(
            {
                "conv": conv,
                "Layer": layer,
                "Width": width,
                "Aggr": aggr,
                "Loss": loss,
                "Tau": tau,
                "Best Tau": best_tau,
            }
        )

    df = pd.DataFrame(data)
    df = df.sort_values(by="Layer")

    if args.find_best:
        df.drop(columns=["Tau"], inplace=True)
        max_tau_row = df.loc[df["Best Tau"].idxmax()]
        print("\nBest Kendall's Tau: \n", max_tau_row)

    if args.plot:
        plot_tau_for_all(
            data, args.dir, save_for_latex=True, tikz_filename=f"{args.dir}.pgf"
        )
