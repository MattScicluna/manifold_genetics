#!/usr/bin/env python
"""One row per run: status, time and peak per step, widest kneighbors call, theory."""

import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")]
hdr = f"{'label':22} {'status':14} {'n_fit':>7} {'fit s':>7} {'fit GB':>7} {'tr s':>7} {'tr GB':>7} {'max knn call':>16} {'pred GB':>8}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    s = r["steps"]
    widest = max(r["kneighbors_calls"], key=lambda c: c["rows"] * c["n_neighbors"], default=None)
    call = f"{widest['rows']}x{widest['n_neighbors']}" if widest else "-"

    def g(step, field, s=s):
        return s.get(step, {}).get(field, float("nan"))

    print(
        f"{r['config']['label']:22} {r['status'][:14]:14} {r.get('n_fit', 0):>7} "
        f"{g('fit', 'seconds'):>7.1f} {g('fit', 'peak_gb'):>7.2f} {g('transform', 'seconds'):>7.1f} "
        f"{g('transform', 'peak_gb'):>7.2f} {call:>16} {r['max_kneighbors_expected_gb']:>8.2f}"
    )
