import numpy as np


def add_label_noise(y, noise_ratio, num_classes=10):

    y_noisy = y.copy()

    n_samples = len(y)
    n_noisy = int(noise_ratio * n_samples)

    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    for idx in noisy_indices:

        current_label = y_noisy[idx]

        new_label = np.random.randint(0, num_classes)

        while new_label == current_label:
            new_label = np.random.randint(0, num_classes)

        y_noisy[idx] = new_label

    return y_noisy
