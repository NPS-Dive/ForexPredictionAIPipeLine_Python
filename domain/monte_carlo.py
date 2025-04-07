# Monte Carlo Simulation (Domain Layer)
import numpy as np

class MonteCarloSimulator:
    def __init__(self, num_simulations: int = 1000):
        """Initialize Monte Carlo simulator for risk assessment."""
        self.num_simulations = num_simulations
    
    def simulate(self, prediction: float, volatility: float) -> dict:
        """
        Simulate possible price movements based on the prediction and volatility.
        
        Args:
            prediction: Probability of price increase (0-1)
            volatility: Historical price volatility in percentage
        
        Returns:
            Dictionary with simulation results
        """
        # Convert prediction to drift (-1 to 1 scale)
        drift = (prediction - 0.5) * 2
        
        # Simulate price movements
        simulations = np.random.normal(
            loc=drift * volatility,  # Mean movement based on prediction
            scale=volatility,        # Standard deviation based on volatility
            size=self.num_simulations
        )
        
        # Calculate statistics
        return {
            'up_probability': float(np.mean(simulations > 0)),
            'down_probability': float(np.mean(simulations < 0)),
            'expected_move': float(np.mean(simulations)),
            'risk_reward_ratio': float(np.abs(np.mean(simulations[simulations > 0]) / np.mean(simulations[simulations < 0] or -0.001))),
            'confidence': float(abs(drift))
        }