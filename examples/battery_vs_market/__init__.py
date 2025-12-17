"""
Battery vs. Imbalance Market Comparison.

This example answers the question: Is it smarter to invest in a battery
or should balancing energy be procured through the imbalance market?

Scenario:
    - Time horizon: 1 week at 15-minute resolution (672 timesteps)
    - Imbalance prices: Real data from Swissgrid (Preise für Ausgleichsenergie)
    - Imbalance profile: Energy deviations that need to be settled
    - Battery option: 100 kWh / 50 kW BESS
    - Market option: Direct settlement with TSO

Comparison:
    1. Scenario A: Battery + market backup
    2. Scenario B: Pure market settlement
"""