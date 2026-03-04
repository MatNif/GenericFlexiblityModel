"""
This file transforms the available raw data into valid imbalance_prices.csv and imbalance_profile.csv files.

The core transformation steps include:
- fetching raw adata from data/raw subfolder
- conversion of any .xlsx files into .csv files
- retrieval of sell and buy prices for balance energy from any "AEPreise_*.csv" files
- consolidation of all pricing data into single data object
- elimination of duplicate and erroneous data:
    - generally values from newer file > values from older file, files' (generation date is part of file naming)
    - filter out extreme spikes
    - clear missing data values (e.g. price=0 or =NA)
- retrieval of actual and predicted load curves from "Prognosekontrolle_*.csv" files
- computation of imbalance profile
- unit conversion
- cropping of the time series so pricing and profile match up

This file also contains a few debugging and visualization methods to ensure the raw data is converted correctly.
"""

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ConversionConfig:
    """Configuration parameters for data conversion pipeline."""

    raw_dir: Path
    output_dir: Path
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    spike_threshold_sigma: float = 5.0
    filter_zero_prices: bool = False  # Replace prices <= min_valid_price with forward-fill (negative prices are valid!)
    min_valid_price: float = 0.01  # CHF/MWh threshold for filtering
    price_unit_factor: float = 0.1  # CHF/MWh → ct/kWh
    imbalance_unit_factor: float = 1.0  # kW scaling factor
    require_continuous_prices: bool = False
    require_continuous_imbalance: bool = False
    remove_spikes: bool = False  # Enable/disable spike removal
    debug_limit_rows: Optional[int] = None  # Limit rows for debugging (e.g., 10000)


def get_output_paths(config: ConversionConfig) -> Tuple[Path, Path, Path]:
    """
    Return output paths for raw data conversion.
    :param config: configuration parameters for conversion.
    :return: output paths for raw data conversion.
    """
    prices_path = config.output_dir / "imbalance_prices.csv"
    profile_path = config.output_dir / "imbalance_profile.csv"
    config_cache_path = config.output_dir / ".conversion_config_cache.json"
    return prices_path, profile_path, config_cache_path


def is_conversion_necessary(config: ConversionConfig) -> bool:
    """
    Check if the conversion is necessary, judging by whether the config has changed or new raw data has been added
    since the last conversion.

    :param config: new conversion configuration
    :return: True if conversion was necessary, False otherwise.
    """
    prices_path, profile_path, config_cache_path = get_output_paths(config)
    if prices_path.exists() and profile_path.exists():
        # Check if config has changed
        config_changed = True
        if config_cache_path.exists():
            try:
                with open(config_cache_path, 'r') as f:
                    cached_config = json.load(f)

                # Convert current config to dict for comparison
                current_config = asdict(config)
                # Convert Path objects to strings for JSON comparison
                current_config['raw_dir'] = str(current_config['raw_dir'])
                current_config['output_dir'] = str(current_config['output_dir'])
                current_config['start_date'] = current_config['start_date'].isoformat() if current_config[
                    'start_date'] else None
                current_config['end_date'] = current_config['end_date'].isoformat() if current_config[
                    'end_date'] else None

                config_changed = (current_config != cached_config)
            except Exception as e:
                print(f"Warning: Could not load cached config: {e}")
                config_changed = True

        if not config_changed:
            # Check if outputs are newer than inputs
            output_mtime = min(prices_path.stat().st_mtime, profile_path.stat().st_mtime)

            # Get all raw input files
            raw_files = list(config.raw_dir.glob("AEPreise-HPFC_BabyCore_*.xlsx"))
            raw_files.extend(config.raw_dir.glob("Prognosekontrolle*.csv"))

            if raw_files:
                newest_input = max(f.stat().st_mtime for f in raw_files)

                if output_mtime > newest_input:
                    print("\n" + "=" * 60)
                    print("✓ Output files are up to date, skipping conversion")
                    print("=" * 60)
                    return False
            else:
                print("Warning: No raw input files found")
        else:
            print("\nConfig has changed since last conversion, rerunning...")

    return True


def combine_xlsx_to_csv(raw_dir: Path) -> Path:
    """
    Combine all AEPreise XLSX files into a single CSV.

    Args:
        raw_dir: Directory containing raw XLSX files

    Returns:
        Path to combined CSV file
    """
    combined_path = raw_dir / "aepreise_hpfc_combined.csv"
    xlsx_files = list(raw_dir.glob("AEPreise-HPFC_BabyCore_*.xlsx"))

    if not xlsx_files:
        raise FileNotFoundError(f"No AEPreise XLSX files found in {raw_dir}")

    # Check if combined CSV is up to date
    if combined_path.exists():
        combined_mtime = combined_path.stat().st_mtime
        xlsx_mtimes = [f.stat().st_mtime for f in xlsx_files]
        if combined_mtime > max(xlsx_mtimes):
            print(f"Combined CSV already up to date: {combined_path}")
            return combined_path

    print(f"Combining {len(xlsx_files)} XLSX files...")
    dfs = []

    for xlsx_file in sorted(xlsx_files):
        print(f"  Reading {xlsx_file.name}")
        df = pd.read_excel(xlsx_file)
        df['sourcefile'] = xlsx_file.name
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(combined_path, index=False)
    print(f"Wrote combined CSV with {len(combined_df)} rows to {combined_path}")

    return combined_path


def load_raw_prices(config: ConversionConfig) -> pd.DataFrame:
    """
    Load raw price data from combined CSV.

    Args:
        config: Conversion configuration

    Returns:
        DataFrame with columns: timestamp, AE_Kauf, AE_Verkauf, source_date
    """
    # Ensure combined CSV exists
    combine_xlsx_to_csv(config.raw_dir)

    csv_path = config.raw_dir / "aepreise_hpfc_combined.csv"
    print(f"Loading raw prices from {csv_path}")

    df = pd.read_csv(
        csv_path,
        usecols=['TimeStamp', 'AE_Kauf', 'AE_Verkauf', 'sourcefile']
    )

    # Parse timestamps (format: DD.MM.YYYY HH:MM:SS)
    df['timestamp'] = pd.to_datetime(df['TimeStamp'], format='%d.%m.%Y %H:%M:%S')

    # Extract source date from filename using regex
    def extract_source_date(filename):
        match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if match:
            return pd.to_datetime(match.group(1))
        return pd.NaT

    df['source_date'] = df['sourcefile'].apply(extract_source_date)

    # Drop rows where both prices are 0 or NaN
    initial_rows = len(df)
    df = df[~((df['AE_Kauf'].isna() | (df['AE_Kauf'] == 0)) &
              (df['AE_Verkauf'].isna() | (df['AE_Verkauf'] == 0)))]

    print(f"Loaded {len(df)} rows ({initial_rows - len(df)} rows with invalid prices dropped)")

    return df[['timestamp', 'AE_Kauf', 'AE_Verkauf', 'source_date']]


def clean_prices(df: pd.DataFrame, config: ConversionConfig) -> pd.DataFrame:
    """
    Clean price data: deduplicate, filter invalid values, remove spikes.

    Args:
        df: Raw price DataFrame
        config: Conversion configuration

    Returns:
        Cleaned DataFrame with columns: timestamp, AE_Kauf, AE_Verkauf
    """
    initial_rows = len(df)
    print(f"Cleaning prices (initial: {initial_rows} rows)...")

    # Step 1: Deduplication - sort by timestamp and source_date, keep last (newer file wins)
    df = df.sort_values(['timestamp', 'source_date'])
    df = df.drop_duplicates(subset='timestamp', keep='last')
    print(f"  After deduplication: {len(df)} rows ({initial_rows - len(df)} duplicates removed)")

    # Step 2: Replace invalid prices with forward-fill (preserves all timestamps)
    # Negative prices are valid in energy markets!
    if config.filter_zero_prices:
        for col in ['AE_Kauf', 'AE_Verkauf']:
            # Replace values at or below threshold with NaN, then forward-fill
            invalid_mask = df[col] <= config.min_valid_price
            n_invalid = invalid_mask.sum()
            if n_invalid > 0:
                df.loc[invalid_mask, col] = np.nan
                df[col] = df[col].ffill()
                print(f"  Replaced {n_invalid} invalid values in {col} with forward-fill")
    else:
        print("  Skipping price threshold filtering (disabled - all prices including negative allowed)")

    # Step 3: Spike removal using rolling z-score (optional)
    if config.remove_spikes:
        window = 96  # 1 day of 15-min intervals

        for col in ['AE_Kauf', 'AE_Verkauf']:
            # Calculate rolling mean and std
            rolling_mean = df[col].rolling(window=window, center=True, min_periods=10).mean()
            rolling_std = df[col].rolling(window=window, center=True, min_periods=10).std()

            # Calculate z-scores
            z_scores = (df[col] - rolling_mean) / rolling_std

            # Replace spikes with NaN
            spike_mask = z_scores.abs() > config.spike_threshold_sigma
            spikes_found = spike_mask.sum()

            if spikes_found > 0:
                df.loc[spike_mask, col] = np.nan
                print(f"  Removed {spikes_found} spikes from {col}")

                # Forward fill NaN values
                df[col] = df[col].ffill()

        # Drop any remaining rows with NaN
        before_dropna = len(df)
        df = df.dropna(subset=['AE_Kauf', 'AE_Verkauf'])
        if before_dropna > len(df):
            print(f"  Dropped {before_dropna - len(df)} rows with remaining NaN values")
    else:
        print("  Skipping spike removal (disabled)")

    # Debug: limit rows if specified
    if config.debug_limit_rows:
        df = df.head(config.debug_limit_rows)
        print(f"  Debug mode: limited to {len(df)} rows")

    print(f"Cleaning complete: {len(df)} rows remaining")

    return df[['timestamp', 'AE_Kauf', 'AE_Verkauf']].reset_index(drop=True)


def load_raw_load(config: ConversionConfig) -> pd.DataFrame:
    """
    Load raw load data from Prognosekontrolle CSV.

    Args:
        config: Conversion configuration

    Returns:
        DataFrame with columns: timestamp, lgs_plus, prognose_lgs_plus
    """
    # Find Prognosekontrolle CSV file
    prognose_files = list(config.raw_dir.glob("Prognosekontrolle*.csv"))

    if not prognose_files:
        raise FileNotFoundError(f"No Prognosekontrolle CSV found in {config.raw_dir}")

    if len(prognose_files) > 1:
        print(f"Warning: Multiple Prognosekontrolle files found, using {prognose_files[0].name}")

    csv_path = prognose_files[0]
    print(f"Loading raw load data from {csv_path}")

    df = pd.read_csv(
        csv_path,
        usecols=['zeit', 'lgs_plus', 'prognose_lgs_plus']
    )

    # Parse timestamps (ISO 8601 with timezone)
    df['timestamp'] = pd.to_datetime(df['zeit'])

    # Convert from UTC to Europe/Zurich, then strip timezone
    df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Zurich').dt.tz_localize(None)

    # Drop rows where actual=0 and forecast>0 (future forecast-only data)
    initial_rows = len(df)
    df = df[~((df['lgs_plus'] == 0) & (df['prognose_lgs_plus'] > 0))]

    print(f"Loaded {len(df)} rows ({initial_rows - len(df)} forecast-only rows dropped)")

    return df[['timestamp', 'lgs_plus', 'prognose_lgs_plus']]


def compute_imbalance(df: pd.DataFrame, config: ConversionConfig) -> pd.DataFrame:
    """
    Compute imbalance from actual and predicted load.

    Args:
        df: DataFrame with lgs_plus and prognose_lgs_plus columns
        config: Conversion configuration

    Returns:
        DataFrame with columns: timestamp, imbalance_kw
    """
    result = df.copy()
    result['imbalance_kw'] = result['lgs_plus'] - result['prognose_lgs_plus']

    # Remove duplicate timestamps (keep last)
    initial_rows = len(result)
    result = result.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='last')
    if initial_rows > len(result):
        print(f"  Removed {initial_rows - len(result)} duplicate timestamps")

    # Debug: limit rows if specified
    if config.debug_limit_rows:
        result = result.head(config.debug_limit_rows)
        print(f"  Debug mode: limited to {len(result)} rows")

    print(f"Computed imbalance: mean={result['imbalance_kw'].mean():.2f} kW, "
          f"std={result['imbalance_kw'].std():.2f} kW")

    return result[['timestamp', 'imbalance_kw']]


def align_and_crop(
    prices: pd.DataFrame,
    imbalance: pd.DataFrame,
    config: ConversionConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align and crop price and imbalance time series.

    Args:
        prices: Price DataFrame
        imbalance: Imbalance DataFrame
        config: Conversion configuration

    Returns:
        Tuple of (prices_cropped, imbalance_cropped)
    """
    print("Aligning and cropping time series...")

    # Determine time range
    if config.start_date and config.end_date:
        start = config.start_date
        end = config.end_date
        print(f"  Using specified date range: {start} to {end}")
    else:
        # Find overlapping range
        start = max(prices['timestamp'].min(), imbalance['timestamp'].min())
        end = min(prices['timestamp'].max(), imbalance['timestamp'].max())
        print(f"  Using overlapping date range: {start} to {end}")

    # Create complete 15-minute date range
    complete_range = pd.date_range(start=start, end=end, freq='15min')
    print(f"  Expected {len(complete_range)} timesteps")

    # Reindex prices
    prices_indexed = prices.set_index('timestamp').reindex(complete_range)

    # Check for gaps in prices
    price_gaps = prices_indexed[['AE_Kauf', 'AE_Verkauf']].isna().any(axis=1)
    if price_gaps.sum() > 0:
        gap_indices = complete_range[price_gaps]
        if config.require_continuous_prices:
            raise ValueError(
                f"Price time series has {price_gaps.sum()} gaps. "
                f"First gap at: {gap_indices[0]}, last gap at: {gap_indices[-1]}. "
                f"Set require_continuous_prices=False to forward-fill gaps."
            )
        else:
            print(f"  Warning: {price_gaps.sum()} gaps in price series, forward-filling (up to 4 steps)")
            prices_indexed = prices_indexed.ffill(limit=4)
            # Check if any gaps remain
            remaining_gaps = prices_indexed[['AE_Kauf', 'AE_Verkauf']].isna().any(axis=1).sum()
            if remaining_gaps > 0:
                raise ValueError(f"Could not fill all price gaps: {remaining_gaps} gaps remain")

    # Reindex imbalance
    imbalance_indexed = imbalance.set_index('timestamp').reindex(complete_range)

    # Check for gaps in imbalance
    imbalance_gaps = imbalance_indexed['imbalance_kw'].isna()
    if imbalance_gaps.sum() > 0:
        gap_indices = complete_range[imbalance_gaps]
        if config.require_continuous_imbalance:
            raise ValueError(
                f"Imbalance time series has {imbalance_gaps.sum()} gaps. "
                f"First gap at: {gap_indices[0]}, last gap at: {gap_indices[-1]}. "
                f"Set require_continuous_imbalance=False to forward-fill gaps."
            )
        else:
            print(f"  Warning: {imbalance_gaps.sum()} gaps in imbalance series, forward-filling (up to 4 steps)")
            imbalance_indexed = imbalance_indexed.ffill(limit=4)
            # Check if any gaps remain
            remaining_gaps = imbalance_indexed['imbalance_kw'].isna().sum()
            if remaining_gaps > 0:
                raise ValueError(f"Could not fill all imbalance gaps: {remaining_gaps} gaps remain")

    # Apply imbalance unit factor
    imbalance_indexed['imbalance_kw'] = imbalance_indexed['imbalance_kw'] * config.imbalance_unit_factor

    # Reset index to get timestamp column back
    prices_cropped = prices_indexed.reset_index().rename(columns={'index': 'timestamp'})
    imbalance_cropped = imbalance_indexed.reset_index().rename(columns={'index': 'timestamp'})

    # Verify identical length
    assert len(prices_cropped) == len(imbalance_cropped), "Length mismatch after alignment"

    print(f"  Aligned series with {len(prices_cropped)} timesteps")

    return prices_cropped, imbalance_cropped


def save_imbalance_prices(df: pd.DataFrame, config: ConversionConfig) -> Path:
    """
    Save imbalance prices in Swissgrid CSV format.

    Args:
        df: Price DataFrame with timestamp, AE_Kauf, AE_Verkauf
        config: Conversion configuration

    Returns:
        Path to saved CSV file
    """
    output_path, _, _ = get_output_paths(config)

    # Apply unit conversion (CHF/MWh → ct/kWh)
    df = df.copy()
    df['bg_long'] = (df['AE_Verkauf'] * config.price_unit_factor).round(2)
    df['bg_short'] = (df['AE_Kauf'] * config.price_unit_factor).round(2)

    # Format timestamps
    df['timestamp_str'] = df['timestamp'].dt.strftime('%d.%m.%Y %H:%M:%S')

    # Write CSV with UTF-8-BOM encoding
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        # Write header
        f.write(',BG long (ct/kWh),BG short (ct/kWh),\n')

        # Write data rows
        for _, row in df.iterrows():
            f.write(f"{row['timestamp_str']},{row['bg_long']:.2f},{row['bg_short']:.2f},\n")

    print(f"Saved imbalance prices to {output_path} ({len(df)} rows)")

    return output_path


def save_imbalance_profile(df: pd.DataFrame, config: ConversionConfig) -> Path:
    """
    Save imbalance profile CSV.

    Args:
        df: Imbalance DataFrame with timestamp and imbalance_kw
        config: Conversion configuration

    Returns:
        Path to saved CSV file
    """
    _, output_path, _ = get_output_paths(config)

    # Create 0-indexed timestep column
    df = df.copy()
    df['timestep'] = range(len(df))
    df['imbalance_kw'] = df['imbalance_kw'].round(2)

    # Write CSV
    df[['timestep', 'imbalance_kw']].to_csv(output_path, index=False)

    print(f"Saved imbalance profile to {output_path} ({len(df)} rows)")

    return output_path


def load_output_data(config: ConversionConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the saved output CSV files for analysis/plotting.

    Args:
        config: Conversion configuration

    Returns:
        Tuple of (prices_df, imbalance_df) with original column names
    """
    prices_path, profile_path, _ = get_output_paths(config)

    # Load prices - need to reverse the unit conversion to get CHF/MWh
    prices_df = pd.read_csv(prices_path, encoding='utf-8-sig')
    prices_df['timestamp'] = pd.to_datetime(prices_df.iloc[:, 0], format='%d.%m.%Y %H:%M:%S')
    prices_df['AE_Verkauf'] = prices_df['BG long (ct/kWh)'] / config.price_unit_factor
    prices_df['AE_Kauf'] = prices_df['BG short (ct/kWh)'] / config.price_unit_factor

    # Load imbalance profile
    imbalance_df = pd.read_csv(profile_path)
    imbalance_df = imbalance_df.rename(columns={'imbalance_kw': 'imbalance_kw'})

    # Add timestamps to imbalance (approximate based on prices start time)
    if len(prices_df) == len(imbalance_df):
        imbalance_df['timestamp'] = prices_df['timestamp'].values

    return prices_df[['timestamp', 'AE_Kauf', 'AE_Verkauf']], imbalance_df[['timestamp', 'imbalance_kw']]


def plot_diagnostics(prices: pd.DataFrame, imbalance: pd.DataFrame):
    """
    Plot diagnostic figures for converted data.

    Args:
        prices: Price DataFrame
        imbalance: Imbalance DataFrame
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("Matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Prices over time
    ax = axes[0]
    ax.plot(prices['timestamp'], prices['AE_Verkauf'], label='AE Verkauf (BG long)', alpha=0.7)
    ax.plot(prices['timestamp'], prices['AE_Kauf'], label='AE Kauf (BG short)', alpha=0.7)
    ax.set_ylabel('Price (CHF/MWh)')
    ax.set_title('Imbalance Prices')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Plot 2: Imbalance over time
    ax = axes[1]
    ax.plot(imbalance['timestamp'], imbalance['imbalance_kw'], alpha=0.7)
    ax.set_ylabel('Imbalance (kW)')
    ax.set_title('Imbalance Profile')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Plot 3: Price spread histogram
    ax = axes[2]
    spread = prices['AE_Kauf'] - prices['AE_Verkauf']
    ax.hist(spread, bins=50, alpha=0.7)
    ax.set_xlabel('Price Spread (CHF/MWh)')
    ax.set_ylabel('Frequency')
    ax.set_title('Price Spread Distribution (Buy - Sell)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"\nPrices (CHF/MWh):")
    print(f"  AE Verkauf (sell): mean={prices['AE_Verkauf'].mean():.2f}, "
          f"std={prices['AE_Verkauf'].std():.2f}, "
          f"min={prices['AE_Verkauf'].min():.2f}, "
          f"max={prices['AE_Verkauf'].max():.2f}")
    print(f"  AE Kauf (buy):     mean={prices['AE_Kauf'].mean():.2f}, "
          f"std={prices['AE_Kauf'].std():.2f}, "
          f"min={prices['AE_Kauf'].min():.2f}, "
          f"max={prices['AE_Kauf'].max():.2f}")
    print(f"  Spread:            mean={spread.mean():.2f}, "
          f"std={spread.std():.2f}")

    print(f"\nImbalance (kW):")
    print(f"  mean={imbalance['imbalance_kw'].mean():.2f}, "
          f"std={imbalance['imbalance_kw'].std():.2f}, "
          f"min={imbalance['imbalance_kw'].min():.2f}, "
          f"max={imbalance['imbalance_kw'].max():.2f}")


def convert(config: Optional[ConversionConfig] = None, no_plot: bool = False) -> Tuple[Path, Path]:
    """
    Main conversion function orchestrating the entire pipeline.

    Args:
        config: Conversion configuration (if None, uses defaults)
        no_plot: If True, skip diagnostic plots (useful when called programmatically)

    Returns:
        Tuple of (imbalance_prices_path, imbalance_profile_path)
    """
    # Use default config if not provided
    if config is None:
        from .data_loader import get_data_path
        data_path = get_data_path()
        config = ConversionConfig(
            raw_dir=data_path / 'raw',
            output_dir=data_path
        )

    print("=" * 60)
    print("Starting raw data conversion pipeline")
    print("=" * 60)
    print(f"Raw data directory: {config.raw_dir}")
    print(f"Output directory: {config.output_dir}")

    # Check if conversion is necessary
    prices_path, profile_path, config_cache_path = get_output_paths(config)
    conversion_needed = is_conversion_necessary(config)

    if conversion_needed:
        # Step 1: Load and clean prices
        print("\n--- Step 1: Load and clean prices ---")
        raw_prices = load_raw_prices(config)
        clean_prices_df = clean_prices(raw_prices, config)

        # Step 2: Load and process load data
        print("\n--- Step 2: Load and process load data ---")
        raw_load = load_raw_load(config)
        imbalance_df = compute_imbalance(raw_load, config)

        # Step 3: Align and crop time series
        print("\n--- Step 3: Align and crop time series ---")
        prices_combined, imbalance_combined = align_and_crop(clean_prices_df, imbalance_df, config)

        # Step 4: Save outputs
        print("\n--- Step 4: Save outputs ---")
        prices_path = save_imbalance_prices(prices_combined, config)
        profile_path = save_imbalance_profile(imbalance_combined, config)

        print("\n" + "=" * 60)
        print("Conversion complete!")
        print("=" * 60)

        # Save config cache for future runs
        try:
            config_dict = asdict(config)
            config_dict['raw_dir'] = str(config_dict['raw_dir'])
            config_dict['output_dir'] = str(config_dict['output_dir'])
            config_dict['start_date'] = config_dict['start_date'].isoformat() if config_dict['start_date'] else None
            config_dict['end_date'] = config_dict['end_date'].isoformat() if config_dict['end_date'] else None

            with open(config_cache_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"Saved config cache to {config_cache_path}")
        except Exception as e:
            print(f"Warning: Could not save config cache: {e}")

    # Step 5: Plot diagnostics (skip if no_plot=True)
    if not no_plot:
        print("\n--- Step 5: Diagnostics ---")
        try:
            if not conversion_needed:
                # Load from saved files
                prices_combined, imbalance_combined = load_output_data(config)
            # Plot converted and aligned prices and imbalances
            plot_diagnostics(prices_combined, imbalance_combined)
        except Exception as e:
            print(f"Could not plot diagnostics: {e}")

    return prices_path, profile_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw energy market data to imbalance prices and profile CSVs"
    )
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--spike-sigma',
        type=float,
        default=5.0,
        help='Spike detection threshold (standard deviations, default: 5.0)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip diagnostic plots'
    )
    parser.add_argument(
        '--allow-gaps',
        action='store_true',
        help='Allow gaps in data and forward-fill them (sets require_continuous_prices/imbalance to False)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode: limit to 10,000 rows and skip spike removal for faster testing'
    )

    args = parser.parse_args()

    # Get data path - go up 2 levels from this file to battery_vs_market/data
    data_path = Path(__file__).parent.parent.parent / 'data'

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else None
    end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else None

    # Create config
    config = ConversionConfig(
        raw_dir=data_path / 'raw',
        output_dir=data_path,
        start_date=start_date,
        end_date=end_date,
        spike_threshold_sigma=args.spike_sigma,
#        require_continuous_prices=not args.allow_gaps,
#        require_continuous_imbalance=not args.allow_gaps,
#        remove_spikes=not args.debug,
        debug_limit_rows=10000 if args.debug else None
    )

    # Run conversion
    if args.no_plot:
        # Monkey-patch plot_diagnostics to skip plotting
        import sys
        module = sys.modules[__name__]
        original_plot = module.plot_diagnostics
        module.plot_diagnostics = lambda *args, **kwargs: None

    convert(config)
