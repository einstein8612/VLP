from models import register_models
from dataset import FPDataset

from time import time

import torch
from torch.utils.data import DataLoader

import argparse
import os

task_registry = register_models()

def main(args):
    torch.manual_seed(args.seed)

    task = args.task
    tasks = task_registry.get_task_names()
    if task not in tasks:
        raise ValueError(f"Task {task} not found. Available tasks: {tasks}")
    
    now = int(time())

    print("Loading model...")
    model_cls = task_registry.get_model_class(args.task)
    model_args = task_registry.get_model_args(args.task)
    model = model_cls(**model_args, device=args.device, seed=args.seed)

    if args.load:
        model.load(args.load)
        print(f"Model loaded from {args.load}")
    else:
        print("Loading dataset...")
        training_dataset = FPDataset(args.dataset + "/train.csv", args.device)

        print("Fitting model...")
        model.fit(training_dataset)

        if args.save:
            print("Saving model...")
            os.makedirs("saved_runs", exist_ok=True)
            model_path = model.save(f"saved_runs/{task}-{now}")
            print(f"Model saved to {model_path}")

    print("Predicting...")
    test_dataset = FPDataset(args.dataset + "/test.csv", args.device)
    loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    X, y = next(iter(loader))

    predictions = model.predict(X)
    average_error = torch.norm(predictions - y, dim=1).mean().item()
    print("Average error:", average_error)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--save", type=bool, required=False, help="Whether to save the model", default=True)
    parser.add_argument("--load", type=str, required=False, help="Model to load")
    parser.add_argument("--device", type=str, required=False, help="Device to use", default="cpu")
    parser.add_argument("--seed", type=int, required=False, help="Seed for randomness", default=42)

    args = parser.parse_args()
    main(args)