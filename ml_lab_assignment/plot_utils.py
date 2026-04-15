"""Plotting helpers for training diagnostics."""

import matplotlib.pyplot as plt


def plot_single_curve(values, xlabel, ylabel, title):
    """Plot a single line curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
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
    plt.show()
