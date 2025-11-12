import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. normalizing data in [0, 1] - color shades are from [0, 255]
x_train_norm = x_train.astype('float32') / 255.0

print(f"Training data size (shape): {x_train_norm.shape}")

# image no (50000)
# 1: row (32)
# 2: column (32)
# 3: color channel (3)
mean = np.mean(x_train_norm, axis=(0, 1, 2))
std = np.std(x_train_norm, axis=(0, 1, 2))

print("\nAggregated channel statistics")
print(f"Mea (R, G, B): {mean}")
print(f"Standard deviation (R, G, B): {std}")
print(f"Val Min: {np.min(x_train_norm)}")  # should be 0.0
print(f"Val Max: {np.max(x_train_norm)}")  # should be 1.0

# Această secțiune se traduce direct în Tabela ta de Statistici.

# Reshape data
x_train_flat = x_train_norm.reshape(-1, 3)

plt.figure(figsize=(15, 5))

# R=0, G=1, B=2
colors = ['Red', 'Green', 'Blue']

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.hist(x_train_flat[:, i], bins=50, color=colors[i].lower(), alpha=0.7)
    plt.title(f'Pixel distribution - Channel {colors[i]}')
    plt.xlabel('Val Pixel (0-1)')
    plt.ylabel('Frecquincy')
    plt.grid(axis='y', alpha=0.5)

plt.tight_layout()
plt.show()

# Interpretare: Comentează forma distribuțiilor (probabil o distribuție bimodală
# sau înclinată, nu perfect normală).

img_mean = np.mean(x_train_norm, axis=(1, 2))

correlation_matrix = np.corrcoef(img_mean, rowvar=False)

print("\n# correlation matrix (R, G, B)")
print(correlation_matrix)

plt.figure(figsize=(6, 5))
plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(ticks=[0, 1, 2], labels=['Red', 'Green', 'Blue'])
plt.yticks(ticks=[0, 1, 2], labels=['Red', 'Green', 'Blue'])
plt.title('Color channel correlation matrix')
plt.show()

# Interpretare: Valorile apropiate de 1.0 indică o corelație puternică (dacă R crește, G crește).
# În CIFAR-10, te aștepți la o corelație pozitivă puternică între canale.

# Independence check (scatter plot)
plt.figure(figsize=(6, 5))
plt.scatter(x_train_flat[::500, 0], x_train_flat[::500, 1], s=2, alpha=0.5, color='purple')
plt.title("Independence check: Red vs Green channel")
plt.xlabel("Red intensity")
plt.ylabel("Green intensity")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# add noise sigma = 0.1)
sigma = 0.1
zgomot = np.random.normal(loc=0.0, scale=sigma, size=x_train_norm.shape)
x_train_noisy = x_train_norm + zgomot

x_train_noisy = np.clip(x_train_noisy, 0., 1.)

# 5 clean vs noisy
n_afisari = 5
plt.figure(figsize=(12, 5))

for i in range(n_afisari):
    ax = plt.subplot(2, n_afisari, i + 1)
    plt.imshow(x_train_norm[i])
    plt.title(f"Clean (Class {y_train[i][0]})")
    plt.axis('off')

    ax = plt.subplot(2, n_afisari, i + 1 + n_afisari)
    plt.imshow(x_train_noisy[i])
    plt.title("Noisy (Input)")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Flatten to pixels and compute PCA (principal components importance)
pca = PCA(n_components=3)
pca.fit(x_train_flat)

explained_variance = pca.explained_variance_ratio_
print("\nExplained variance ratio per component:")
print(explained_variance)

plt.figure(figsize=(5, 4))
plt.bar(['PC1', 'PC2', 'PC3'], explained_variance, color='teal')
plt.title("Feature Importance (PCA variance ratio)")
plt.ylabel("Explained variance")
plt.show()
