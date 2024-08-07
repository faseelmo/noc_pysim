from data.utils import does_path_contains_files

import os
import shutil
import random
from natsort import natsorted


if __name__ == "__main__":

    test_path = "data/training_data/test"
    print(f"\nCreating test data in {test_path}")

    test_input = "data/training_data/test/input/"
    test_target = "data/training_data/test/target/"
    test_packet_list = "data/training_data/test/packet_list/"

    os.makedirs(test_input, exist_ok=True)
    os.makedirs(test_target, exist_ok=True)
    os.makedirs(test_packet_list, exist_ok=True)

    does_path_contains_files(test_path + "/input")
    does_path_contains_files(test_path + "/target")
    does_path_contains_files(test_path + "/packet_list")

    input_path = "data/training_data/input"
    target_path = "data/training_data/target"
    packet_list_path = "data/training_data/packet_list"

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(packet_list_path, exist_ok=True)

    input_files = natsorted(os.listdir(input_path))

    test_split = int(len(input_files) * 0.05)
    print(f"Test split is {test_split}")

    random_test_files = random.sample(input_files, test_split)

    for file in random_test_files:
        shutil.move(f"{input_path}/{file}", f"{test_input}/{file}")
        shutil.move(f"{target_path}/{file}", f"{test_target}/{file}")
        shutil.move(f"{packet_list_path}/{file}", f"{test_packet_list}/{file}")

    print(
        f"Test data moved to {test_path}. Number of test files is {len(random_test_files)}"
    )
