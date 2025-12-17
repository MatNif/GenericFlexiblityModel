"""
Data loading utilities for the battery vs. market scenario.

Handles loading and preprocessing of:
- Imbalance prices (Swissgrid data)
- Imbalance profile (energy deviation time series)
"""

from pathlib import Path
from typing import Dict, Tuple
import csv


def load_imbalance_prices(filepath: str) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Load imbalance prices from Swissgrid CSV file.

    Expected CSV format (from Swissgrid):
        ,BG long (ct/kWh),BG short (ct/kWh),
        01.11.2025 00:00:00,7.65,35.62,
        01.11.2025 00:15:00,7.65,30.67,
        ...

    Where:
        - Column 1: Timestamp (DD.MM.YYYY HH:MM:SS)
        - BG long: Price when you have excess energy (can sell) [ct/kWh]
        - BG short: Price when you need energy (must buy) [ct/kWh]

    Notes:
        - Prices are converted from ct/kWh to CHF/kWh (divided by 100)
        - Timestamps are converted to sequential integer timesteps (0, 1, 2, ...)
        - BG short → p_buy (price for positive imbalance)
        - BG long → p_sell (price for negative imbalance)

    Args:
        filepath: Path to Swissgrid CSV file with imbalance prices.

    Returns:
        (p_buy, p_sell) where:
            - p_buy: Dict mapping timestep -> price for positive imbalance [CHF/kWh]
            - p_sell: Dict mapping timestep -> price for negative imbalance [CHF/kWh]
    """
    p_buy = {}
    p_sell = {}

    with open(filepath, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
        reader = csv.reader(f)

        # Skip header row
        next(reader)

        # Read data rows
        timestep = 0
        for row in reader:
            if len(row) < 3:
                continue  # Skip empty rows

            # Skip rows with empty price values
            if not row[1] or not row[2] or row[1].strip() == '' or row[2].strip() == '':
                continue

            try:
                # timestamp = row[0]  # Keep for future use if needed
                bg_long_ct = float(row[1])   # Price when long (can sell) [ct/kWh]
                bg_short_ct = float(row[2])  # Price when short (must buy) [ct/kWh]

                # Convert from ct/kWh to CHF/kWh
                p_buy[timestep] = bg_short_ct / 100.0  # BG short = price to buy
                p_sell[timestep] = bg_long_ct / 100.0  # BG long = price to sell

                timestep += 1

            except (ValueError, IndexError):
                # Skip rows that can't be parsed
                continue

    return p_buy, p_sell


def load_imbalance_profile(filepath: str) -> Dict[int, float]:
    """
    Load imbalance profile from CSV file.

    Expected CSV format:
        timestep,imbalance_kw
        0,15.5
        1,-8.2
        2,22.3
        ...

    Where:
        - timestep: Integer time index (0, 1, 2, ...)
        - imbalance_kw: Power imbalance at this timestep [kW]
          Positive = consumed more than scheduled (need to buy)
          Negative = consumed less than scheduled (can sell)

    Args:
        filepath: Path to CSV file with imbalance profile.

    Returns:
        Dict mapping timestep -> imbalance power [kW]
    """
    imbalance = {}

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row['timestep'])
            imbalance[t] = float(row['imbalance_kw'])

    return imbalance


def generate_dummy_imbalance_profile(
    n_timesteps: int,
    mean_power: float = 0.0,
    amplitude: float = 30.0,
    seed: int = 42,
) -> Dict[int, float]:
    """
    Generate dummy imbalance profile for testing.

    Creates a synthetic imbalance profile with:
    - Daily pattern (24-hour cycle)
    - Random noise
    - Occasional large imbalances

    Args:
        n_timesteps: Number of timesteps to generate.
        mean_power: Mean imbalance power [kW].
        amplitude: Amplitude of daily variation [kW].
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping timestep -> imbalance power [kW]
    """
    import random
    import math

    random.seed(seed)
    imbalance = {}

    # Assuming 15-minute timesteps: 96 per day
    timesteps_per_day = 96

    for t in range(n_timesteps):
        # Daily sinusoidal pattern
        hour_of_day = (t % timesteps_per_day) / timesteps_per_day  # [0, 1)
        daily_pattern = amplitude * math.sin(2 * math.pi * hour_of_day)

        # Random noise
        noise = random.gauss(0, amplitude / 3)

        # Occasional large deviations (10% chance)
        if random.random() < 0.1:
            spike = random.choice([-1, 1]) * amplitude * 2
        else:
            spike = 0

        imbalance[t] = mean_power + daily_pattern + noise + spike

    return imbalance


def generate_dummy_imbalance_prices(
    n_timesteps: int,
    mean_buy: float = 0.25,
    mean_sell: float = 0.15,
    volatility: float = 0.10,
    seed: int = 42,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Generate dummy imbalance prices for testing.

    Creates synthetic prices with:
    - Time-varying behavior
    - Asymmetry between buy and sell prices
    - Volatility spikes

    Args:
        n_timesteps: Number of timesteps to generate.
        mean_buy: Mean buy price [CHF/kWh].
        mean_sell: Mean sell price [CHF/kWh].
        volatility: Price volatility factor.
        seed: Random seed for reproducibility.

    Returns:
        (p_buy, p_sell) dictionaries mapping timestep -> price [CHF/kWh]
    """
    import random
    import math

    random.seed(seed)
    p_buy = {}
    p_sell = {}

    # Assuming 15-minute timesteps: 96 per day
    timesteps_per_day = 96

    for t in range(n_timesteps):
        # Daily pattern: higher prices during peak hours
        hour_of_day = (t % timesteps_per_day) / timesteps_per_day
        peak_factor = 1.0 + 0.3 * math.sin(2 * math.pi * (hour_of_day - 0.25))  # Peak around noon

        # Random walk component
        noise_buy = random.gauss(0, volatility)
        noise_sell = random.gauss(0, volatility * 0.8)  # Slightly less volatile

        # Ensure buy > sell (market spread)
        p_buy[t] = max(0.05, mean_buy * peak_factor + noise_buy)
        p_sell[t] = max(0.01, min(mean_sell * peak_factor + noise_sell, p_buy[t] * 0.8))

    return p_buy, p_sell


def get_data_path() -> Path:
    """Return path to data directory."""
    return Path(__file__).parent.parent / 'data'
