#!/usr/bin/env python3
"""
Filter duplicate and multi-allelic SNPs from PLINK files.

This script handles two types of filtering:
1. Duplicate positions (same chr:pos, possibly different alleles): keep lowest missingness
2. Multi-allelic positions (same chr:pos, different alleles): keep highest MAF

Usage:
    python -m manifold_genetics.utils.filter_duplicates --bim FILE.bim --lmiss FILE.lmiss --frq FILE.frq --out PREFIX

Outputs:
    PREFIX.keep_snps.txt - SNP IDs to keep after filtering
    PREFIX.stats.txt - Summary statistics
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Filter duplicate/multi-allelic SNPs")
    parser.add_argument("--bim", required=True, help="PLINK .bim file")
    parser.add_argument("--lmiss", required=True, help="PLINK .lmiss file (from --missing)")
    parser.add_argument("--frq", required=True, help="PLINK .frq file (from --freq)")
    parser.add_argument("--out", required=True, help="Output prefix")
    return parser.parse_args()


def read_bim(bim_path: Path) -> dict:
    """Read BIM file and return dict of SNP_ID -> (chr, pos, a1, a2)."""
    snp_info = {}
    with open(bim_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            chrom, snp_id, cm, pos, a1, a2 = parts
            snp_info[snp_id] = (chrom, int(pos), a1, a2)
    return snp_info


def read_lmiss(lmiss_path: Path) -> dict:
    """Read .lmiss file and return dict of SNP_ID -> missingness rate."""
    missingness = {}
    with open(lmiss_path, "r") as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                snp_id = parts[1]  # SNP column
                miss_rate = float(parts[4])  # F_MISS column
                missingness[snp_id] = miss_rate
    return missingness


def read_frq(frq_path: Path) -> dict:
    """Read .frq file and return dict of SNP_ID -> MAF."""
    maf = {}
    with open(frq_path, "r") as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                snp_id = parts[1]  # SNP column
                maf_val = float(parts[4]) if parts[4] != "NA" else 0.0  # MAF column
                maf[snp_id] = maf_val
    return maf


def filter_snps(snp_info: dict, missingness: dict, maf: dict) -> tuple:
    """
    Filter SNPs to keep only one per position.

    Strategy:
    1. Group SNPs by chr:pos
    2. For positions with multiple SNPs:
       - First, prefer SNPs with lowest missingness
       - If tied, prefer SNPs with highest MAF

    Returns:
        (keep_snps, stats_dict)
    """
    # Group SNPs by position (chr:pos)
    pos_to_snps = {}
    for snp_id, (chrom, pos, a1, a2) in snp_info.items():
        key = f"{chrom}:{pos}"
        if key not in pos_to_snps:
            pos_to_snps[key] = []
        pos_to_snps[key].append(snp_id)

    # Stats
    total_snps = len(snp_info)
    unique_positions = len(pos_to_snps)
    duplicate_positions = sum(1 for snps in pos_to_snps.values() if len(snps) > 1)
    snps_at_duplicates = sum(len(snps) for snps in pos_to_snps.values() if len(snps) > 1)

    # Select best SNP per position
    keep_snps = []
    for pos_key, snp_ids in pos_to_snps.items():
        if len(snp_ids) == 1:
            # Only one SNP at this position - keep it
            keep_snps.append(snp_ids[0])
        else:
            # Multiple SNPs - select best one
            # Sort by: missingness (ascending), then MAF (descending)
            scored = []
            for snp_id in snp_ids:
                miss = missingness.get(snp_id, 1.0)  # Default to high missingness if missing
                m = maf.get(snp_id, 0.0)  # Default to 0 MAF if missing
                scored.append((miss, -m, snp_id))  # Negative MAF for descending sort

            scored.sort()
            best_snp = scored[0][2]
            keep_snps.append(best_snp)

    stats = {
        "total_snps": total_snps,
        "unique_positions": unique_positions,
        "duplicate_positions": duplicate_positions,
        "snps_at_duplicate_positions": snps_at_duplicates,
        "snps_removed": total_snps - len(keep_snps),
        "snps_kept": len(keep_snps),
    }

    return keep_snps, stats


def main():
    args = parse_args()

    bim_path = Path(args.bim)
    lmiss_path = Path(args.lmiss)
    frq_path = Path(args.frq)
    out_prefix = args.out

    # Validate inputs
    for p in [bim_path, lmiss_path, frq_path]:
        if not p.exists():
            print(f"Error: File not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Reading BIM file: {bim_path}")
    snp_info = read_bim(bim_path)
    print(f"  Total SNPs: {len(snp_info)}")

    print(f"Reading missingness file: {lmiss_path}")
    missingness = read_lmiss(lmiss_path)
    print(f"  SNPs with missingness data: {len(missingness)}")

    print(f"Reading frequency file: {frq_path}")
    maf = read_frq(frq_path)
    print(f"  SNPs with MAF data: {len(maf)}")

    print("Filtering duplicate/multi-allelic SNPs...")
    keep_snps, stats = filter_snps(snp_info, missingness, maf)

    # Write output files
    keep_file = f"{out_prefix}.keep_snps.txt"
    stats_file = f"{out_prefix}.stats.txt"

    with open(keep_file, "w") as f:
        for snp_id in keep_snps:
            f.write(f"{snp_id}\n")
    print(f"  Wrote {len(keep_snps)} SNP IDs to: {keep_file}")

    with open(stats_file, "w") as f:
        for key, val in stats.items():
            f.write(f"{key}\t{val}\n")
    print(f"  Wrote stats to: {stats_file}")

    # Print summary
    print("\nSummary:")
    print(f"  Total SNPs: {stats['total_snps']}")
    print(f"  Unique positions: {stats['unique_positions']}")
    print(f"  Positions with duplicates: {stats['duplicate_positions']}")
    print(f"  SNPs at duplicate positions: {stats['snps_at_duplicate_positions']}")
    print(f"  SNPs removed: {stats['snps_removed']}")
    print(f"  SNPs kept: {stats['snps_kept']}")


if __name__ == "__main__":
    main()
