# Monte Carlo simulation (Domain Layer)
# Single Responsibility: Simulate outcomes for risk assessment
import numpy as np

class MonteCarloSimulator:
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations

    def simulate(self, predicted_price: float, volatility: float, time_horizon: int = 5) -> dict:
        """Simulate possible price paths."""
        sims = np.random.normal(predicted_price, volatility, (self.n_simulations, time_horizon))
        percentiles = np.percentile(sims[:, -1], [25, 50, 75])
        return {
            'median': percentiles[1],
            'lower_bound': percentiles[0],
            'upper_bound': percentiles[2],
            'success_prob': np.mean(sims[:, -1] > predicted_price)
        }