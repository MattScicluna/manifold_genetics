#!/usr/bin/env python3
"""
Select samples for AoU subsample experiments.

Supports flexible group-based subsampling with pattern matching on
race/ethnicity columns.

Examples:
    # 60K White + all remaining samples
    python select_samples.py \
        --metadata DemographicData.tsv \
        --fam project_subset.fam \
        --output fit_samples.txt \
        --group "White|European:60000" \
        --include-rest

    # 10K each of White, Black, Hispanic
    python select_samples.py \
        --metadata DemographicData.tsv \
        --fam project_subset.fam \
        --output fit_samples.txt \
        --group "White|European:10000" \
        --group "Black or African American:10000" \
        --group "Hispanic or Latino:10000"
"""

import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Select samples for AoU subsample experiments"
    )
    parser.add_argument("--metadata", required=True,
                        help="Path to DemographicData.tsv")
    parser.add_argument("--fam", required=True,
                        help="Path to intersected .fam file")
    parser.add_argument("--output", required=True,
                        help="Output sample list file (FID TAB IID)")
    parser.add_argument("--group", action="append", required=True,
                        help="Group spec as 'pattern:count' (e.g. 'White|European:60000'). "
                             "Pattern is matched case-insensitively via str.contains. "
                             "Can be specified multiple times.")
    parser.add_argument("--include-rest", action="store_true",
                        help="Include all samples not matched by any group")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling (default: 42)")
    args = parser.parse_args()

    # Parse group specifications
    groups = []
    for g in args.group:
        pattern, count = g.rsplit(":", 1)
        groups.append((pattern, int(count)))

    # Read metadata
    metadata = pd.read_csv(args.metadata, sep='\t', low_memory=False)
    metadata['sample_id'] = metadata['person_id'].astype(str)

    # Read .fam file to get available samples and FID mapping
    fam = pd.read_csv(args.fam, sep=r'\s+', header=None,
                      names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype'])
    iid_to_fid = dict(zip(fam['IID'].astype(str), fam['FID'].astype(str)))
    available_samples = set(fam['IID'].astype(str))

    # Filter metadata to available samples
    metadata = metadata[metadata['sample_id'].isin(available_samples)]

    # Auto-detect race/ethnicity column
    if 'race_ethnicity' in metadata.columns:
        race_col = 'race_ethnicity'
    elif 'race' in metadata.columns:
        race_col = 'race'
    else:
        print("  Error: Could not find race/ethnicity column in metadata!")
        exit(1)

    print(f"    Using column: {race_col}")
    print(f"    Total available samples: {len(metadata)}")

    # Select samples from each group
    selected = []
    selected_ids = set()

    for pattern, target_count in groups:
        mask = metadata[race_col].str.contains(pattern, case=False, na=False)
        # Exclude already-selected samples to prevent overlap between groups
        mask = mask & ~metadata['sample_id'].isin(selected_ids)
        group_samples = metadata[mask]

        if len(group_samples) > target_count:
            print(f"    {pattern}: {len(group_samples)} available, subsampling to {target_count}")
            group_selected = group_samples.sample(n=target_count, random_state=args.seed)
        else:
            print(f"    {pattern}: {len(group_samples)} available, using all")
            group_selected = group_samples

        selected.append(group_selected)
        selected_ids.update(group_selected['sample_id'])

    # Optionally include all remaining samples
    if args.include_rest:
        rest = metadata[~metadata['sample_id'].isin(selected_ids)]
        print(f"    Other (all remaining): {len(rest)}")
        selected.append(rest)

    fit_samples = pd.concat(selected)
    print(f"    Final fit subset size: {len(fit_samples)}")

    # Write FID/IID pairs for PLINK extraction
    with open(args.output, 'w') as f:
        for sample_id in fit_samples['sample_id']:
            if sample_id in iid_to_fid:
                f.write(f"{iid_to_fid[sample_id]}\t{sample_id}\n")

    print(f"  Saved fit sample list: {args.output}")


if __name__ == "__main__":
    main()
