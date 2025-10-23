#!/usr/bin/env python3
"""
XOR‑pattern synthetic dataset generator.

Features
--------
* 2‑D XOR data (quadrant‑based binary labels).
* Adjustable size, radius, noise, and random seed.
* Optional Matplotlib visualisation (`-p/--plot`).
* Saves the generated arrays to a **pre‑existing** ../data directory.
* Prompts before overwriting an existing file (`--output`).

Usage examples
--------------
$ python xor_generator.py                     # defaults, no plot, default file name
$ python xor_generator.py -n 400 -p           # plot, default file name
$ python xor_generator.py -o my_xor.npz -p    # custom name, plot
$ python xor_generator.py -r 2.5 -s 0.15 -seed 123 -p -o experiment1.npz
"""

import argparse
import os
import sys

import numpy as np


# ----------------------------------------------------------------------
# 1️⃣  DATA GENERATION
# ----------------------------------------------------------------------
def generate_xor_data(
    n_per_quadrant: int = 250,
    radius: float = 1.0,
    noise_std: float = 0.2,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the XOR pattern."""
    rng = np.random.default_rng(random_state)

    # Quadrant centres and their labels
    centers = np.array(
        [
            [+radius, +radius],  # Q1 → 1
            [-radius, +radius],  # Q2 → 0
            [-radius, -radius],  # Q3 → 1
            [+radius, -radius],  # Q4 → 0
        ]
    )
    labels = np.array([1, 0, 1, 0])

    X = np.empty((4 * n_per_quadrant, 2), dtype=np.float64)
    y = np.empty(4 * n_per_quadrant, dtype=np.int64)

    for i, (c, lab) in enumerate(zip(centers, labels)):
        start = i * n_per_quadrant
        end = start + n_per_quadrant
        X[start:end] = rng.normal(loc=c, scale=noise_std, size=(n_per_quadrant, 2))
        y[start:end] = lab

    # Shuffle so the order isn’t quadrant‑sorted
    perm = rng.permutation(X.shape[0])
    X, y = X[perm], y[perm]

    return X, y


# ----------------------------------------------------------------------
# 2️⃣  OPTIONAL PLOTTING
# ----------------------------------------------------------------------
def _maybe_import_matplotlib():
    """Import matplotlib lazily; exit with a friendly message if missing."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(
            "\nERROR: Plotting requested but Matplotlib could not be imported.\n"
            f"Details: {exc}\n"
            "Install it with: pip install matplotlib\n"
        )
        sys.exit(1)


def plot_xor_data(X: np.ndarray, y: np.ndarray) -> None:
    """Scatter‑plot the data, colour‑coded by label."""
    _maybe_import_matplotlib()
    import matplotlib.pyplot as plt  # Imported inside the function on purpose

    cmap = {0: "#1f77b4", 1: "#ff7f0e"}  # blue & orange
    colors = [cmap[label] for label in y]

    plt.figure(figsize=(6, 6))
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=colors,
        s=30,
        alpha=0.8,
        edgecolor="k",
        linewidth=0.3,
    )
    plt.axhline(0, color="gray", lw=0.5, ls="--")
    plt.axvline(0, color="gray", lw=0.5, ls="--")
    plt.title("XOR synthetic dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 3️⃣  SAVE TO PRE‑EXISTING ../data WITH OVERWRITE PROTECTION
# ----------------------------------------------------------------------
def _get_data_dir() -> str:
    """
    Return the absolute path to the expected '../data' directory.
    Raise a clear error if it does not exist.
    """
    data_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))
    if not os.path.isdir(data_dir):
        sys.stderr.write(
            f"\nERROR: Expected data directory '{data_dir}' does not exist.\n"
            "Create it manually (e.g., `mkdir -p ../data`) before running the script.\n"
        )
        sys.exit(1)
    return data_dir


def _prompt_overwrite(filepath: str) -> bool:
    """Ask the user whether to overwrite an existing file."""
    while True:
        resp = input(f"File '{filepath}' already exists. Overwrite? [y/N] ").strip().lower()
        if resp in ("y", "yes"):
            return True
        if resp in ("", "n", "no"):
            return False
        print("Please answer with 'y' (yes) or 'n' (no).")


def save_dataset(X: np.ndarray, y: np.ndarray, filename: str) -> None:
    """Save X and y to '../data/<filename>'. Handles overwrite prompting."""
    data_dir = _get_data_dir()

    # Ensure the filename ends with .npz (NumPy compressed archive)
    if not filename.lower().endswith(".npz"):
        filename += ".npz"
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        if not _prompt_overwrite(filepath):
            print("Save aborted – existing file left untouched.")
            return

    np.savez_compressed(filepath, X=X, y=y)
    print(f"Dataset saved to: {filepath}")


# ----------------------------------------------------------------------
# 4️⃣  MAIN ENTRY POINT
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate (optionally plot) XOR‑pattern 2‑D data and save it."
    )
    parser.add_argument(
        "-n",
        "--samples-per-quadrant",
        type=int,
        default=250,
        help="Points per quadrant (default: 250)",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=1.0,
        help="Distance of each quadrant centre from the origin (default: 1.0)",
    )
    parser.add_argument(
        "-s",
        "--noise",
        type=float,
        default=0.2,
        help="Std‑dev of Gaussian noise (default: 0.2)",
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Show a scatter plot of the generated data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="xor_dataset.npz",
        help="File name (saved under ../data). Extension .npz is added if omitted.",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------
    # Generate data
    # --------------------------------------------------------------
    X, y = generate_xor_data(
        n_per_quadrant=args.samples_per_quadrant,
        radius=args.radius,
        noise_std=args.noise,
        random_state=args.seed,
    )

    # --------------------------------------------------------------
    # Quick sanity printout
    # --------------------------------------------------------------
    print("\nFirst 10 samples (x1, x2, label):")
    for i in range(min(10, len(y))):
        print(f"{X[i,0]: .3f}, {X[i,1]: .3f} → {y[i]}")

    # --------------------------------------------------------------
    # Save the dataset (will abort with a clear error if ../data is missing)
    # --------------------------------------------------------------
    save_dataset(X, y, args.output)

    # --------------------------------------------------------------
    # Optional visualisation
    # --------------------------------------------------------------
    if args.plot:
        plot_xor_data(X, y)


if __name__ == "__main__":
    main()
