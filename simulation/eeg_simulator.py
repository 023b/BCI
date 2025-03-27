import numpy as np

class EEGSimulator:
    def __init__(self, buffer_size=256):
        self.buffer_size = buffer_size
        self.alpha = np.zeros(buffer_size)
        self.beta = np.zeros(buffer_size)
        self.mu = np.zeros(buffer_size)
        self.frontal_asymmetry = np.zeros(buffer_size)

    def simulate_from_features(self, facial_features, posture_features):
        alpha = np.mean(facial_features) * np.random.rand()
        beta = np.mean(posture_features) * np.random.rand()
        mu = np.mean(facial_features + posture_features) * np.random.rand()
        asymmetry = (alpha - beta) * np.random.rand()

        self.alpha = np.roll(self.alpha, -1)
        self.beta = np.roll(self.beta, -1)
        self.mu = np.roll(self.mu, -1)
        self.frontal_asymmetry = np.roll(self.frontal_asymmetry, -1)

        self.alpha[-1] = alpha
        self.beta[-1] = beta
        self.mu[-1] = mu
        self.frontal_asymmetry[-1] = asymmetry

        return np.array([alpha, beta, mu, asymmetry])
