"""
Generate dummy imbalance profile for testing.

Creates a realistic imbalance profile matching the time horizon of the
Swissgrid price data.
"""

import csv
from .data_loader import generate_dummy_imbalance_profile, get_data_path, load_imbalance_prices

# Load price data to get the time horizon
data_path = get_data_path()
prices_file = data_path / 'imbalance_prices.csv'

if prices_file.exists():
    p_buy, p_sell = load_imbalance_prices(str(prices_file))
    n_timesteps = len(p_buy)
    print(f"Loaded {n_timesteps} timesteps from price data")
    print(f"Time horizon: {n_timesteps / 96:.1f} days (15-min resolution)")
else:
    print("Price data not found, using default horizon of 1 week")
    n_timesteps = 672  # 1 week at 15-min resolution

# Generate dummy imbalance profile
print("\nGenerating dummy imbalance profile...")
print("  Parameters:")
print("    - Mean imbalance: 0 kW (balanced on average)")
print("    - Daily variation: +/-30 kW")
print("    - Random noise: Gaussian with sigma=10 kW")
print("    - Occasional spikes: +/-60 kW (10% probability)")

imbalance = generate_dummy_imbalance_profile(
    n_timesteps=n_timesteps,
    mean_power=0.0,      # Balanced on average
    amplitude=30.0,      # ±30 kW daily variation
    seed=42,             # Reproducible
)

# Save to CSV
output_file = data_path / 'imbalance_profile.csv'
print(f"\nSaving to {output_file}...")

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestep', 'imbalance_kw'])

    for t in range(n_timesteps):
        writer.writerow([t, f"{imbalance[t]:.2f}"])

print(f"Saved {n_timesteps} timesteps")

# Calculate statistics
imb_values = list(imbalance.values())
mean_imb = sum(imb_values) / len(imb_values)
min_imb = min(imb_values)
max_imb = max(imb_values)
positive_count = sum(1 for v in imb_values if v > 0)
negative_count = sum(1 for v in imb_values if v < 0)

print("\nImbalance profile statistics:")
print(f"  Mean:     {mean_imb:6.2f} kW")
print(f"  Min:      {min_imb:6.2f} kW (excess generation)")
print(f"  Max:      {max_imb:6.2f} kW (excess consumption)")
print(f"  Positive: {positive_count:4d} timesteps ({100*positive_count/len(imb_values):.1f}%)")
print(f"  Negative: {negative_count:4d} timesteps ({100*negative_count/len(imb_values):.1f}%)")

# Calculate total energy
total_energy_kwh = sum(imb_values) * 0.25  # 15-min timesteps
print(f"\nTotal net imbalance: {total_energy_kwh:.1f} kWh over {n_timesteps/96:.1f} days")
print(f"Average daily imbalance: {total_energy_kwh / (n_timesteps/96):.1f} kWh/day")

print("\nDone! You can now run the optimization scenario.")