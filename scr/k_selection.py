import numpy as np
import matplotlib.pyplot as plt
from models import KMeansScratch
from metrics import silhouette_score_manual


def elbow_method(data, k_min=2, k_max=13):
    wcss = []
    ks = list(range(k_min, k_max + 1))

    for k in ks:
        kmeans = KMeansScratch(k=k, max_iters=50)
        kmeans.fit(data)

        distances = kmeans._compute_distances(data)
        min_distances = np.min(distances, axis=1)
        inertia = np.sum(min_distances ** 2)

        wcss.append(inertia)
        print(f"k={k}: WCSS={inertia:.2f}")

    return ks, wcss

def plot_elbow(ks, wcss):
    plt.figure(figsize=(8, 5))
    plt.plot(ks, wcss, marker='o', linestyle='--', color='b')
    plt.xlabel("nr of clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.show()

def silhouette_method(data, k_min=2, k_max=11):
    scores = []
    ks = list(range(k_min, k_max + 1))

    for k in ks:
        kmeans = KMeansScratch(k=k, max_iters=50)
        kmeans.fit(data)

        score = silhouette_score_manual(data, kmeans.labels_, sample_size=1000)
        scores.append(score)
        print(f"k={k}: Silhouette={score:.4f}")

    return ks, scores

def plot_silhouette(ks, scores):
    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, marker='o', color='green', linestyle='--')
    plt.xlabel("nr of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")
    plt.grid(True)
    plt.show()


