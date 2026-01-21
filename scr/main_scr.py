import shap

import numpy as np
from data_loader import load_cifar10, add_noise
from scr.models import AutoencoderScratch, KMeansScratch
from scr.train_engine import train_one_epoch, cross_validation
from scr.metrics import silhouette_score_manual
from sklearn.metrics import calinski_harabasz_score, adjusted_rand_score
from visualization import plot_reconstruction, plot_loss_curve, plot_latent_clusters
from k_selection import elbow_method, plot_elbow, silhouette_method, plot_silhouette

DATA_PATH = '../data/cifar-10-batches-py'
EPOCHS = 10
BATCH_SIZE = 64

def explain_with_shap_scratch(model, x_data):
    print("\nSHAP explanation KernelExplainer")

    # wrapper
    def predict_fn(data):
        return model.encode(data)

    # 10 img
    background = x_data[:10]
    test_images = x_data[10:12]
    explainer = shap.KernelExplainer(predict_fn, background)

    shap_values = explainer.shap_values(test_images, nsamples=50)
    shap_numpy = np.array(shap_values)
    shap_sum = np.abs(shap_numpy).mean(axis=0)
    shap_view = shap_sum.reshape(-1, 32, 32, 3)
    original_view = test_images.reshape(-1, 32, 32, 3)

    shap.image_plot(shap_view, original_view)

def optimize_hyperparameters(x_train, k_folds=3):
    learning_rates = [0.01, 0.05, 0.1]
    best_lr = None
    best_score = float('inf')

    # grid search
    for lr in learning_rates:
        print(f"\ntesting learning rate: {lr}")
        scores = cross_validation(x_train, k_folds=k_folds, epochs=3, lr=lr)
        mean_mse = np.mean(scores)
        print(f"  mean MSE: {mean_mse:.5f}")

        if mean_mse < best_score:
            best_score = mean_mse
            best_lr = lr

    print(f"\n best lr: {best_lr} (MSE: {best_score:.5f})")
    return best_lr


def main():
    try:
        x_clean, y_true = load_cifar10(DATA_PATH)
        x_noisy = add_noise(x_clean, sigma=0.1)
    except FileNotFoundError as e:
        print(e)
        return

    best_lr = optimize_hyperparameters(x_noisy, k_folds=3)
    scores = cross_validation(x_noisy, k_folds=3, epochs=5, lr=best_lr)

    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    conf_interval = 1.96 * (std_dev / np.sqrt(len(scores)))

    print(f"\nCV mean MSE: {mean_score:.4f} (+/- {conf_interval:.4f})")

    model = AutoencoderScratch(input_dim=3072, latent_dim=128, lr=best_lr)

    loss_history = []

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, x_noisy, batch_size=BATCH_SIZE)
        loss_history.append(loss)
        print(f"epoch {epoch + 1}/{EPOCHS} - loss: {loss:.4f}")

    plot_loss_curve(loss_history)

    reconstructed_images = model.forward(x_noisy[:5])
    plot_reconstruction(x_noisy[:5], reconstructed_images, n=5)

    latent_vectors = model.encode(x_noisy)

    # ks_elbow, wcss = elbow_method(latent_vectors, k_min=2, k_max=12)
    # plot_elbow(ks_elbow, wcss)
    # ks_sil, scores = silhouette_method(latent_vectors, k_min=2, k_max=11)
    # plot_silhouette(ks_sil, scores)

    kmeans = KMeansScratch(k=10)
    kmeans.fit(latent_vectors)

    sil_score = silhouette_score_manual(latent_vectors, kmeans.labels_)
    print(f"\nSilhouette Score: {sil_score:.4f}")

    ch_score = calinski_harabasz_score(latent_vectors, kmeans.labels_)
    print(f"Calinski-Harabasz Index: {ch_score:.4f}")

    ari = adjusted_rand_score(y_true[:len(kmeans.labels_)], kmeans.labels_)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")

    plot_latent_clusters(latent_vectors, kmeans.labels_)

    explain_with_shap_scratch(model, x_noisy)

if __name__ == "__main__":
    main()
