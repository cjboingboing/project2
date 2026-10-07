"""Compare the three weightings of the ranking side by side.

Reads each variant's own outputs and reports how much the shortlists actually differ.
Run after all three have been executed:

    python scripts/compare_variants.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    "hand (notebooks/)": PROJECT_ROOT / "data" / "tables" / "processed",
    "hybrid":            PROJECT_ROOT / "data" / "variants" / "hybrid" / "processed",
    "machine":           PROJECT_ROOT / "data" / "variants" / "machine" / "processed",
}


def load(folder: Path, name: str) -> pd.DataFrame | None:
    path = folder / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    tops, rows = {}, []
    for label, folder in VARIANTS.items():
        top = load(folder, "top100_v2.csv")
        fraud = load(folder, "top100_v2_fraud_adjusted.csv")
        seg = load(folder, "top10_per_segment.csv")
        if top is None:
            print(f"[belum dijalankan] {label}: {folder/'top100_v2.csv'} tidak ada")
            continue
        tops[label] = set(top["merchant_abn"])
        rows.append({
            "variant": label,
            "merchants in top 100": len(top),
            "BNPL revenue in top 100 ($M)": round(top["estimated_bnpl_revenue"].sum() / 1e6, 2),
            "median transactions": int(top["transaction_count"].median()),
            "fraud-adjusted list": "yes" if fraud is not None else "missing",
            "top 10 per segment": "yes" if seg is not None else "missing",
        })

    if not rows:
        print("tidak ada varian yang sudah dijalankan")
        return

    print("\n=== PROFIL TIAP VARIAN ===")
    print(pd.DataFrame(rows).set_index("variant").to_string())

    print("\n=== BERAPA TOKO YANG SAMA DI TIAP PASANGAN (dari 100) ===")
    labels = list(tops)
    overlap = pd.DataFrame(index=labels, columns=labels, dtype="object")
    for a in labels:
        for b in labels:
            overlap.loc[a, b] = len(tops[a] & tops[b])
    print(overlap.to_string())

    if len(labels) == 3:
        everywhere = set.intersection(*tops.values())
        print(f"\nToko yang terpilih oleh ketiganya : {len(everywhere)}/100")
        for a in labels:
            only = tops[a] - set.union(*(tops[b] for b in labels if b != a))
            print(f"Hanya muncul di '{a}' : {len(only)}")


if __name__ == "__main__":
    main()
