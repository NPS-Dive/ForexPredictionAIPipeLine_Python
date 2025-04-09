# Monte Carlo Simulation (Domain Layer)
# Single Responsibility: Assess risk across all timeframes

import numpy as np

class MonteCarloSimulator:
    def __init__(self, num_simulations: int = 1000):
        """Initialize Monte Carlo simulator."""
        self.num_simulations = num_simulations

    def simulate(self, prediction: float, volatility: float, is_classifier: bool = True) -> dict:
        """Simulate price movements and calculate risk metrics."""
        drift = (prediction - 0.5) * 2 if is_classifier else prediction
        simulations = np.random.normal(loc=drift * volatility, 
                                     scale=volatility, 
                                     size=self.num_simulations)
        
        up_prob = float(np.mean(simulations > 0))
        expected_move = float(np.mean(simulations))
        pos_returns = simulations[simulations > 0]
        neg_returns = simulations[simulations < 0]
        risk_reward = float(np.abs(np.mean(pos_returns) / np.mean(neg_returns or [-0.001])))

        return {
            'up_prob': up_prob,
            'expected_move': expected_move,
            'risk_reward': risk_reward
        }