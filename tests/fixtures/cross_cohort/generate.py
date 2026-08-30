"""Regenerate the cross-cohort fixture files. Seeded — reproduces the checked-in CSVs exactly."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

d = Path(__file__).parent
d.mkdir(parents=True, exist_ok=True)
ids = [f"SAMPLE_{i:03d}" for i in range(50)]
rng = np.random.default_rng(0)

fit_pops = ["FrenchBasque", "Yoruba", "Han", "Karitiana", "Papuan"]
pd.DataFrame(
    {"sample_id": ids, "Population": [fit_pops[i % len(fit_pops)] for i in range(50)]}
).to_csv(d / "fit_labels.csv", index=False)

proj_anc = ["European", "African", "East Asian", "South Asian", "Admixed"]
pd.DataFrame(
    {
        "sample_id": ids,
        "self_described_ancestry": [proj_anc[i % len(proj_anc)] for i in range(50)],
    }
).to_csv(d / "project_labels.csv", index=False)

(d / "fit_colormap.json").write_text(
    json.dumps(
        {
            "Population": {
                "FrenchBasque": "#1f77b4",
                "Yoruba": "#ff7f0e",
                "Han": "#2ca02c",
                "Karitiana": "#d62728",
                "Papuan": "#9467bd",
            }
        },
        indent=2,
    )
    + "\n"
)
(d / "project_colormap.json").write_text(
    json.dumps(
        {
            "self_described_ancestry": {
                "European": "#1f77b4",
                "African": "#ff7f0e",
                "East Asian": "#2ca02c",
                "South Asian": "#d62728",
                "Admixed": "#7f7f7f",
            }
        },
        indent=2,
    )
    + "\n"
)

pd.DataFrame(
    {
        "sample_id": ids,
        "latitude": rng.uniform(-55, 70, 50).round(4),
        "longitude": rng.uniform(-160, 175, 50).round(4),
    }
).to_csv(d / "geographic.csv", index=False)
print("wrote", *sorted(p.name for p in d.iterdir()))
