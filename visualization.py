import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def reshape_for_plot(flat_image):
    img = flat_image.reshape(3, 32, 32).transpose(1, 2, 0)
    return np.clip(img, 0, 1)


def plot_image_comparison(clean_images, noisy_images, n=5):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(reshape_for_plot(clean_images[i]))
        plt.title("clean")
        plt.axis("off")

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reshape_for_plot(noisy_images[i]))
        plt.title("noisy")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_reconstruction(original, reconstructed, n=5):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(reshape_for_plot(original[i]))
        plt.title("input")
        plt.axis("off")

        # Plot Reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reshape_for_plot(reconstructed[i]))
        plt.title("reconstructed")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_loss_curve(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label='Training Loss', color='blue')
    plt.title('Autoencoder training loss')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_latent_clusters(latent_vectors, labels):
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='cluster')
    plt.title('Latent space visualization')
    plt.xlabel('comp 1')
    plt.ylabel('comp 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()
