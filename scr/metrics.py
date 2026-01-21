import numpy as np

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def silhouette_score_manual(x, labels, sample_size=1000):

    if len(x) > sample_size:
        idx = np.random.choice(len(x), sample_size, replace=False)
        x = x[idx]
        labels = labels[idx]

    n = len(x)
    scores = []
    unique_labels = np.unique(labels)

    for i in range(n):
        point = x[i]
        label = labels[i]

        own_cluster = x[labels == label]
        if len(own_cluster) > 1:
            a = np.mean(np.linalg.norm(own_cluster - point, axis=1))
        else:
            a = 0

        b = np.inf
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_cluster = x[labels == other_label]
            dist = np.mean(np.linalg.norm(other_cluster - point, axis=1))
            if dist < b:
                b = dist

        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0)

    return np.mean(scores)
