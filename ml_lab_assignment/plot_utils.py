"""Plotting helpers for training diagnostics."""

import re
from pathlib import Path

import matplotlib.pyplot as plt


def _get_outputs_dir():
    """Return output directory for persisted plot images."""
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _filename_from_title(title):
    """Create a stable filename from plot title."""
    stem = re.sub(r"[^a-z0-9]+", "_", title.strip().lower()).strip("_")
    return f"{stem}.png"


def _save_current_figure(title):
    """Save current matplotlib figure under outputs/ using title-derived name."""
    out_path = _get_outputs_dir() / _filename_from_title(title)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")


def plot_single_curve(values, xlabel, ylabel, title):
    """Plot a single line curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(values, label=ylabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_current_figure(title)
    plt.show()


def plot_two_curves(train_values, val_values, xlabel, ylabel, title, label1, label2):
    """Plot two curves for train/validation comparison."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_values, label=label1)
    plt.plot(val_values, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_current_figure(title)
    plt.show()
