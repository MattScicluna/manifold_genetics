#!/usr/bin/env python3
"""
Add Genetic_region_merged column to fit_labels.csv based on Population mapping.

Usage:
    python add_genetic_region.py <input_csv> [output_csv]

If output_csv is not specified, the input file is modified in place.

Example:
    python add_genetic_region.py ../hgdp_1kgp_proj/data/fit_labels.csv
"""

import pandas as pd
import sys
from pathlib import Path

# Population to Genetic_region_merged mapping
POPULATION_TO_REGION = {
    # Africa
    'ACB': 'Africa',
    'ASW': 'Africa',
    'BantuKenya': 'Africa',
    'BantuSouthAfrica': 'Africa',
    'Biaka': 'Africa',
    'BiakaPygmy': 'Africa',
    'ESN': 'Africa',
    'GWD': 'Africa',
    'LWK': 'Africa',
    'MSL': 'Africa',
    'Mandenka': 'Africa',
    'Mbuti': 'Africa',
    'MbutiPygmy': 'Africa',
    'San': 'Africa',
    'YRI': 'Africa',
    'Yoruba': 'Africa',

    # America
    'CLM': 'America',
    'Colombian': 'America',
    'Karitiana': 'America',
    'MXL': 'America',
    'Maya': 'America',
    'PEL': 'America',
    'PUR': 'America',
    'Pima': 'America',
    'Surui': 'America',

    # Central_South_Asia
    'BEB': 'Central_South_Asia',
    'Balochi': 'Central_South_Asia',
    'Brahui': 'Central_South_Asia',
    'Burusho': 'Central_South_Asia',
    'GIH': 'Central_South_Asia',
    'Hazara': 'Central_South_Asia',
    'ITU': 'Central_South_Asia',
    'Kalash': 'Central_South_Asia',
    'Makrani': 'Central_South_Asia',
    'PJL': 'Central_South_Asia',
    'Pathan': 'Central_South_Asia',
    'STU': 'Central_South_Asia',
    'Sindhi': 'Central_South_Asia',

    # East_Asia
    'CDX': 'East_Asia',
    'CHB': 'East_Asia',
    'CHS': 'East_Asia',
    'Cambodian': 'East_Asia',
    'Dai': 'East_Asia',
    'Daur': 'East_Asia',
    'Han': 'East_Asia',
    'NorthernHan': 'East_Asia',
    'Hezhen': 'East_Asia',
    'JPT': 'East_Asia',
    'Japanese': 'East_Asia',
    'KHV': 'East_Asia',
    'Lahu': 'East_Asia',
    'Miao': 'East_Asia',
    'Mongola': 'East_Asia',
    'Mongolian': 'East_Asia',
    'Naxi': 'East_Asia',
    'Oroqen': 'East_Asia',
    'She': 'East_Asia',
    'Tu': 'East_Asia',
    'Tujia': 'East_Asia',
    'Uygur': 'East_Asia',
    'Xibo': 'East_Asia',
    'Yakut': 'East_Asia',
    'Yi': 'East_Asia',

    # Europe
    'Adygei': 'Europe',
    'Basque': 'Europe',
    'CEU': 'Europe',
    'FIN': 'Europe',
    'French': 'Europe',
    'GBR': 'Europe',
    'IBS': 'Europe',
    'Italian': 'Europe',
    'BergamoItalian': 'Europe',
    'Orcadian': 'Europe',
    'Russian': 'Europe',
    'Sardinian': 'Europe',
    'TSI': 'Europe',
    'Tuscan': 'Europe',

    # Middle_East
    'Bedouin': 'Middle_East',
    'Druze': 'Middle_East',
    'Mozabite': 'Middle_East',
    'Palestinian': 'Middle_East',

    # Oceania
    'Bougainville': 'Oceania',
    'Melanesian': 'Oceania',
    'Papuan': 'Oceania',
    'PapuanHighlands': 'Oceania',
    'PapuanSepik': 'Oceania',
}


def add_genetic_region(input_path, output_path=None):
    """Add Genetic_region_merged column based on Population mapping."""

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    # Load CSV
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Check if Population column exists
    if 'Population' not in df.columns:
        print(f"Error: 'Population' column not found in {input_path}")
        print(f"Available columns: {list(df.columns)}")
        return False

    # Show unique populations
    populations = df['Population'].unique()
    print(f"Found {len(populations)} unique populations")

    # Check for unmapped populations
    unmapped = [p for p in populations if p not in POPULATION_TO_REGION]
    if unmapped:
        print(f"Warning: {len(unmapped)} populations not in mapping:")
        for p in unmapped:
            print(f"  - {p}")

    # Add Genetic_region_merged column
    df['Genetic_region_merged'] = df['Population'].map(POPULATION_TO_REGION)

    # Report mapping results
    mapped_count = df['Genetic_region_merged'].notna().sum()
    print(f"Mapped {mapped_count}/{len(df)} samples to regions")

    # Show distribution
    print("\nGenetic_region_merged distribution:")
    print(df['Genetic_region_merged'].value_counts())

    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    success = add_genetic_region(input_file, output_file)
    sys.exit(0 if success else 1)
