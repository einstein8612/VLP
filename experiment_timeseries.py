from tqdm import tqdm
from models import register_models

import numpy as np
import numpy.typing as npt
import torch

import matplotlib.pyplot as plt

import argparse

from dataset.led_age_series_util import calculate_decay_constant

R90_MIN = 10000
R90_MAX = 50000


def generate_test_set(data: npt.NDArray, valid_mask: npt.NDArray, test_set_fraction: float, rng: np.random.Generator) -> tuple[npt.NDArray, npt.NDArray]:
    H, W, leds_n = data.shape

    # Flatten data
    flat_data = data.reshape(W*H, leds_n)

    # Generate LED ids
    led_ids = np.arange(leds_n)[None, :]  # Shape (1, 1, leds_n)
    # Generate random sample indices at each time step
    valid_flat_idxs = np.flatnonzero(valid_mask)

    sample_flat_idxs = rng.choice(valid_flat_idxs, int(H*W*test_set_fraction))

    # Calculate the x and y coordinates of the samples
    ys = sample_flat_idxs // W
    xs = sample_flat_idxs % W
    sample_locs = np.stack((xs, ys), axis=-1)

    # Fetch the samples for each LED at each time step
    # Add a new axis to the sample_flat_idxs to match the shape of led_ids
    sample_flat_idxs = sample_flat_idxs[:, None]

    # Get samples
    # Get the samples for each LED at each timestep with the same sample index. Shape (H*W*fraction, leds_n)
    samples = flat_data[sample_flat_idxs, led_ids]

    return samples, sample_locs


def generate_aged_samples(data: npt.NDArray, valid_mask: npt.NDArray, relative_decay: npt.NDArray, samples_per_timestep: int, rng: np.random.Generator) -> tuple[npt.NDArray, npt.NDArray]:
    timesteps_n, leds_n = relative_decay.shape
    H, W, leds_n = data.shape

    # Flatten data
    flat_data = data.reshape(W*H, leds_n)

    # Generate LED ids
    led_ids = np.arange(leds_n)[None, None, :]  # Shape (1, 1, leds_n)
    # Generate random sample indices at each time step
    valid_flat_idxs = np.flatnonzero(valid_mask)

    sample_flat_idxs = rng.choice(
        valid_flat_idxs, size=(timesteps_n, samples_per_timestep))

    # Calculate the x and y coordinates of the samples
    ys = sample_flat_idxs // W
    xs = sample_flat_idxs % W
    sample_locs = np.stack((xs, ys), axis=-1)

    # Fetch the samples for each LED at each time step
    # Add a new axis to the sample_flat_idxs to match the shape of led_ids
    sample_flat_idxs = sample_flat_idxs[:, :, None]
    sample_flat_idxs = np.broadcast_to(
        sample_flat_idxs, (timesteps_n, samples_per_timestep, leds_n))
    led_ids = np.broadcast_to(
        led_ids, (timesteps_n, samples_per_timestep, leds_n))

    # Age the samples
    # Get the samples for each LED at each timestep with the same sample index. Shape (timesteps, samples_per_timestep, leds_n)
    samples = flat_data[sample_flat_idxs, led_ids]
    # Apply the relative decay to the samples
    aged_samples = samples * relative_decay[:, None, :]

    # Add flickering
    # flickering = rng.choice([0, 1], size=aged_samples.shape, p=[args.flickering_prob, 1 - args.flickering_prob])
    # aged_samples = aged_samples * flickering # Apply flickering to the samples

    return aged_samples, sample_locs


task_registry = register_models()


def main(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    task = args.task
    tasks = task_registry.get_task_names()
    if task not in tasks:
        raise ValueError(f"Task {task} not found. Available tasks: {tasks}")

    print("Loading model...")
    model_cls = task_registry.get_model_class(args.task)
    model_args = task_registry.get_model_args(args.task)
    model = model_cls(**model_args, device=args.device, seed=args.seed)

    model.load(args.load_model)
    print(f"Model loaded from {args.load_model}")

    data = np.load(args.src)
    H, W, leds_n = data.shape
    print(f"Data loaded from {args.src}")

    # Generate valid mask for data
    # Shape (H, W), 1 for valid, 0 for invalid same for all LEDs
    valid_mask = np.ones_like(data[:, :, 0], dtype=float)
    valid_mask[data[:, :, 0] == -1] = 0
    print(f"Generated valid mask ({valid_mask.sum()} valid points)")

    # Generate time steps
    timesteps = np.arange(0, args.time, args.timestep)
    print(f"Generated {len(timesteps)} timesteps")

    r90_hours = rng.integers(R90_MIN, R90_MAX, leds_n)
    decay_ks = calculate_decay_constant(r90_hours)
    # Calculate relative decay for each LED at each time step
    relative_decay = np.exp(-np.outer(timesteps, decay_ks))
    # Add noise to the data
    relative_decay += rng.normal(0, args.std, size=relative_decay.shape)
    print(f"Generated {leds_n} decay constants and their decay scalers")

    aged_samples_X, aged_samples_y = generate_aged_samples(
        data, valid_mask, relative_decay, args.samples_per_timestep, rng)
    aged_samples_X, aged_samples_y = torch.tensor(
        aged_samples_X, dtype=torch.float), torch.tensor(aged_samples_y, dtype=torch.float)*10
    print(
        f"Generated {aged_samples_X.shape[0]*aged_samples_X.shape[1]} samples; {aged_samples_X.shape[1]} samples at {aged_samples_X.shape[0]} timesteps")

    test_X, test_y = generate_test_set(data, valid_mask, 0.2, rng)

    # Translate to PyTorch
    test_X, test_y = torch.tensor(test_X, dtype=torch.float), torch.tensor(
        test_y, dtype=torch.float)*10
    relative_decay = torch.tensor(relative_decay, dtype=torch.float)

    avg_decay = relative_decay.mean(dim=1)
    errors = []
    bar = tqdm(enumerate(timesteps), total=len(timesteps))
    for i, t in bar:
        # Warm up the model with samples at this timestep
        aged_samples_X_t, aged_samples_y_t = aged_samples_X[i,
                                                            :, :], aged_samples_y[i, :, :]

        sample_predictions = model.predict(aged_samples_X_t)
        sample_average_error = torch.norm(
            sample_predictions - aged_samples_y_t, dim=1).mean().item()
        print("Samples had average error:", sample_average_error)

        # Test accuracy at this timestep
        decay_scalars = relative_decay[i, :]
        predictions = model.predict(test_X * decay_scalars, eval=True)
        average_error = torch.norm(predictions - test_y, dim=1).mean().item()
        bar.set_description(
            f"Average error at {t} hours: {average_error:.2f} mm")
        errors.append(average_error)

    _, axs = plt.subplots(2)

    axs[0].plot(timesteps, errors, label=type(model).__name__)
    axs[0].set(xlabel='Time in hours passed', ylabel='Positioning Error (mm)')
    axs[0].set(title='Positioning Error vs. LED Degradation')

    axs[1].plot(timesteps, avg_decay, label="Average Decay")
    axs[1].set(xlabel='Time in hours passed', ylabel='Average Decay Scalar')
    axs[1].set(title='Average Decay Scalar vs. Time')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--src", type=str, required=True, help="Dataset name")
    parser.add_argument("--load_model", type=str,
                        required=True, help="Model to load")
    parser.add_argument(
        "--std", help="Standard deviation of noise", type=float, default=0.005
    )
    parser.add_argument(
        "--time", help="Time in hours to age LEDs", type=float, default=50000)
    parser.add_argument("--samples_per_timestep",
                        help="Number of noisy samples per LED per timestep", type=int, default=100)
    parser.add_argument("--timestep", help="Timestep size",
                        type=int, required=True, default=1000)
    parser.add_argument("--flickering_prob", help="Probability of flickering",
                        type=float, required=False, default=0.001)
    parser.add_argument("--device", type=str, required=False,
                        help="Device to use", default="cpu")
    parser.add_argument("--seed", type=int, required=False,
                        help="Seed for randomness", default=42)

    args = parser.parse_args()
    main(args)
