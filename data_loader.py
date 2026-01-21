import numpy as np
import pickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data']  # shape (10000, 3072)


def load_cifar10(root_path):

    x_train = []
    y_train = []

    # 5 batchs
    for i in range(1, 6):
        file_path = os.path.join(root_path, f'data_batch_{i}')
        if not os.path.exists(file_path):
            file_path = os.path.join(root_path, f'data_batch_{i}.bin')

        with open(file_path, 'rb') as fo:
            batch_dict = pickle.load(fo, encoding='bytes')
            x_train.append(batch_dict[b'data'])
            y_train += batch_dict[b'labels']

    x_train = np.vstack(x_train)
    y_train = np.array(y_train)

    x_train_norm = x_train.astype('float32') / 255.0

    return x_train_norm, y_train

def add_noise(x_clean, sigma=0.1):

    print(f"adding noise with sigma={sigma}")

    noise = np.random.normal(loc=0.0, scale=sigma, size=x_clean.shape)

    x_noisy = x_clean + noise
    x_noisy = np.clip(x_noisy, 0., 1.)

    return x_noisy
