import numpy as np
from scr.models import AutoencoderScratch
from scr.metrics import calculate_mse


def train_one_epoch(model, x_train, batch_size=32):
    epoch_loss = 0
    for i in range(0, len(x_train), batch_size):
        batch = x_train[i:i + batch_size]
        if len(batch) == 0:
            break

        reconstructed = model.forward(batch)

        # loss calculation
        loss = calculate_mse(batch, reconstructed)
        epoch_loss += loss

        model.backward(batch, reconstructed)

    return epoch_loss / (len(x_train) / batch_size)

def cross_validation(data, k_folds=5, epochs=5, lr=0.01):
    fold_size = len(data) // k_folds
    mse_scores = []

    for i in range(k_folds):
        print(f"fold {i + 1}/{k_folds}...")

        # split data
        val_start = i * fold_size
        val_end = (i + 1) * fold_size
        x_val = data[val_start:val_end]
        x_train = np.concatenate((data[:val_start], data[val_end:]), axis=0)

        model = AutoencoderScratch(input_dim=3072, latent_dim=64, lr=lr)

        for epoch in range(epochs):
            train_one_epoch(model, x_train)

        preds = model.forward(x_val)
        score = calculate_mse(x_val, preds)
        mse_scores.append(score)
        print(f"    fold {i + 1} MSE: {score:.5f}")

    return np.array(mse_scores)
