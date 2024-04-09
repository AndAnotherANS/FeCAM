import json
import argparse
import os

import neptune as neptune

from trainer import train


def init_neptune(args):
    run = neptune.init_run()
    run["model/parameters"] = args
    run["gridsearch_run_n"] = os.environ["GRIDSEARCH_RUN_N"]
    return run

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    run = init_neptune(args)
    args.update({"neptune": run})

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')

    parser.add_argument('--run_n', type=int, default=0,
                        help='run identifier for gridsearch runs')

    return parser


if __name__ == '__main__':
    main()
