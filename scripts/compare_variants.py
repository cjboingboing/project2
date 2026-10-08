"""Compare the shipped ranking against the alternatives we tried.

The shipped pipeline (notebooks/) now runs on the hybrid weights and writes to
data/tables/processed. The pure-learned-weight variant lives in notebooks_weve_tried/
and writes to data/variants/machine/processed.

Run after both have been executed:

    python scripts/compare_variants.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    "shipped (hybrid)": (PROJECT_ROOT / "data" / "tables" / "processed", "hybrid"),
    "machine weights":  (PROJECT_ROOT / "data" / "variants" / "machine" / "processed", "machine"),
}


def load(folder: Path, name: str):
    path = folder / name
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    tops, rows = {}, []
    for label, (folder, tag) in VARIANTS.items():
        top = load(folder, f"top100_{tag}.csv")
        if top is None:
            print(f"[belum dijalankan] {label}: {folder}/top100_{tag}.csv tidak ada")
            continue
        tops[label] = set(top["merchant_abn"])
        rows.append({
            "variant": label,
            "merchants in top 100": len(top),
            "BNPL revenue in top 100 ($M)": round(top["estimated_bnpl_revenue"].sum() / 1e6, 2),
            "median transactions": int(top["transaction_count"].median()),
            "fraud-adjusted list": "yes" if load(folder, f"top100_{tag}_fraud_adjusted.csv") is not None else "missing",
            "top 10 per segment": "yes" if load(folder, "top10_per_segment.csv") is not None else "missing",
        })

    if not rows:
        print("tidak ada yang sudah dijalankan")
        return

    print("\n=== PROFIL ===")
    print(pd.DataFrame(rows).set_index("variant").to_string())

    if len(tops) == 2:
        a, b = list(tops)
        shared = len(tops[a] & tops[b])
        print(f"\n=== TUMPANG TINDIH ===")
        print(f"{a} vs {b}: {shared}/100 toko sama")
        print(f"hanya di '{a}': {len(tops[a] - tops[b])}")
        print(f"hanya di '{b}': {len(tops[b] - tops[a])}")


if __name__ == "__main__":
    main()
