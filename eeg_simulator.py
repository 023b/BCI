# eeg_simulator.py - Simulated EEG Signal Generator

import numpy as np
import pandas as pd
import random

class EEGSimulator:
    def __init__(self):
        self.time = 0
        self.history = []

    def generate(self):
        # Simulated EEG bands
        alpha = np.clip(np.random.normal(0.5, 0.1), 0, 1)
        beta = np.clip(np.random.normal(0.3, 0.1), 0, 1)
        mu = np.clip(np.random.normal(0.2, 0.1), 0, 1)
        attention = (alpha + beta) / 2

        signal = {
            "alpha": alpha,
            "beta": beta,
            "mu": mu,
            "attention": attention
        }

        self.history.append(signal)
        self.time += 1

        # Return as DataFrame for Streamlit
        df = pd.DataFrame(self.history[-30:])
        return df

    def infer_intent(self):
        # Dummy inference based on attention
        if not self.history:
            return "Idle"

        attention = self.history[-1]['attention']
        if attention > 0.7:
            return "Focused"
        elif attention > 0.4:
            return "Thinking"
        else:
            return "Distracted"

if __name__ == '__main__':
    sim = EEGSimulator()
    for _ in range(10):
        print(sim.generate().tail(1))
        print(sim.infer_intent())
