import numpy as np

class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        # s * (1 - s)
        return grad_output * self.output * (1 - self.output)

class DenseLayer:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros((1, output_size))
        self.lr = learning_rate

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        self.weights -= self.lr * grad_weights
        self.bias -= self.lr * grad_bias
        return grad_input

class AutoencoderScratch:
    def __init__(self, input_dim=3072, latent_dim=64, lr=0.01):
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # encoder
        self.enc_layer = DenseLayer(input_dim, latent_dim, lr)
        self.enc_act = Sigmoid()

        # decoder
        self.dec_layer = DenseLayer(latent_dim, input_dim, lr)
        self.dec_act = Sigmoid()

    def encode(self, x):
        # latent vector
        return self.enc_act.forward(self.enc_layer.forward(x))

    def backward(self, x_input, x_recon):
        # loss func = 1/ N SUM (predicted - true)^2
        grad_output = 2 * (x_recon - x_input) / x_input.size

        grad = self.dec_act.backward(grad_output)
        grad = self.dec_layer.backward(grad)

        grad = self.enc_act.backward(grad)
        self.enc_layer.backward(grad)

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.dec_act.forward(self.dec_layer.forward(latent))
        return reconstructed

class KMeansScratch:
    def __init__(self, k=10, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels_ = None

    def fit(self, data):
        idx = np.random.choice(len(data), self.k, replace=False)
        self.centroids = data[idx]

        for i in range(self.max_iters):
            dists = self._compute_distances(data)
            self.labels_ = np.argmin(dists, axis=1)

            # centroids
            new_centroids = np.array([data[self.labels_ == j].mean(axis=0)
                                      for j in range(self.k)])

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids
        return self

    def _compute_distances(self, data):
        # vectorized euclidean dist
        data_sq = np.sum(data ** 2, axis=1, keepdims=True)
        cent_sq = np.sum(self.centroids ** 2, axis=1)

        distance_sq = data_sq + cent_sq - 2 * np.dot(data, self.centroids.T)
        return np.sqrt(np.maximum(distance_sq, 0))
