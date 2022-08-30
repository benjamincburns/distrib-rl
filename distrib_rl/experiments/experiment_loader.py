import pyjson5 as json
import os

from distrib_rl.experiments.model import ExperimentConfig


def load_experiment(file_path) -> ExperimentConfig:
    if file_path is not None:
        if not os.path.exists(file_path):
            print("\nUNABLE TO LOCATE EXPERIMENT FILE IN PATH:\n", file_path, "\n")
            raise FileNotFoundError

    experiment_json = dict(json.load(open(file_path, "r")))
    experiment_config = ExperimentConfig.parse_obj(experiment_json)
    return experiment_config
