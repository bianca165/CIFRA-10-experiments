import shap

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.model_selection import KFold

from data_loader import load_cifar10, add_noise
from lib.models import AutoencoderLib
from visualization import plot_loss_curve, plot_latent_clusters, plot_reconstruction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10

def explain_with_shap(model, x_data, device):
    print("\nSHAP explanation")
    model.eval()

    # 100 img
    background = torch.tensor(x_data[:100]).float().to(device)
    test_images = torch.tensor(x_data[100:103]).float().to(device)

    explainer = shap.DeepExplainer(model.encoder, background)
    shap_values = explainer.shap_values(test_images)

    shap_numpy = np.array(shap_values)

    shap_sum = np.abs(shap_numpy).mean(axis=0)

    # for the plot
    shap_view = shap_sum.reshape(-1, 32, 32, 3)
    original_view = x_data[100:103].reshape(-1, 32, 32, 3)

    shap.image_plot(shap_view, original_view)

def train_library_cv(x_noisy, x_clean, k_folds=3, epochs=5):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    mse_scores = []

    print(f"\n{k_folds} - Fold Cross Validation")

    tensor_noisy = torch.tensor(x_noisy).float().to(DEVICE)
    tensor_clean = torch.tensor(x_clean).float().to(DEVICE)

    fold = 0
    for train_idx, val_idx in kfold.split(x_noisy):
        fold += 1

        train_noisy = tensor_noisy[train_idx]
        train_clean = tensor_clean[train_idx]
        val_noisy = tensor_noisy[val_idx]
        val_clean = tensor_clean[val_idx]

        # new model
        model = AutoencoderLib().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # train
        for epoch in range(epochs):
            optimizer.zero_grad()
            recon, _ = model(train_noisy)  # noisy
            loss = criterion(recon, train_clean)  # clean
            loss.backward()
            optimizer.step()

        # ev
        with torch.no_grad():
            val_recon, _ = model(val_noisy)  # noisy validation
            val_loss = criterion(val_recon, val_clean).item()
            mse_scores.append(val_loss)
            print(f"  fold {fold} MSE: {val_loss:.5f}")

    return mse_scores

def main():

    data_path = '../data/cifar-10-batches-py'
    try:
        x_clean, y_true = load_cifar10(data_path)
        x_noisy = add_noise(x_clean, sigma=0.1)
    except FileNotFoundError as e:
        print(e)
        return

    # cv
    cv_scores = train_library_cv(x_noisy, x_clean, k_folds=3, epochs=5)
    mean_score = np.mean(cv_scores)
    std_dev = np.std(cv_scores)
    conf_interval = 1.96 * (std_dev / np.sqrt(len(cv_scores)))

    print(f"\n mean MSE: {mean_score:.5f} (+/- {conf_interval:.5f})")

    # train
    tensor_noisy = torch.tensor(x_noisy).float().to(DEVICE)
    tensor_clean = torch.tensor(x_clean).float().to(DEVICE)

    model = AutoencoderLib().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    loss_history = []

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        reconstructed, _ = model(tensor_noisy)
        loss = criterion(reconstructed, tensor_clean)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        print(f"epoch {epoch + 1}/{EPOCHS}, loss: {loss.item():.4f}")

    # clustering
    with torch.no_grad():
        _, latent = model(tensor_noisy)
        latent_np = latent.cpu().numpy()

    kmeans = KMeans(n_clusters=10, random_state=42)
    labels = kmeans.fit_predict(latent_np)

    sil = silhouette_score(latent_np, labels)
    print(f"\nSilhouette Score: {sil:.4f}")

    plot_loss_curve(loss_history)
    plot_latent_clusters(latent_np, labels)

    model.eval()
    with torch.no_grad():
        sample_input = tensor_noisy[:5]
        sample_output, _ = model(sample_input)
        sample_input_np = sample_input.cpu().numpy()
        sample_output_np = sample_output.cpu().numpy()

    plot_reconstruction(sample_input_np, sample_output_np, n=5)

    ari = adjusted_rand_score(y_true, labels)
    print(f"ARI Score: {ari:.4f}")

    ch_score = calinski_harabasz_score(latent_np, labels)
    print(f"Calinski-Harabasz Index: {ch_score:.4f}")

    explain_with_shap(model, x_noisy, DEVICE)

if __name__ == "__main__":
    main()
