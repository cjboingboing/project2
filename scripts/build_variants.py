import json, re, shutil
from pathlib import Path

ROOT = Path("/sessions/rcw-01jeqmsvqtdvcylqgpvpulhh/mnt/MAST30034_Python/project2")
SHARED = ROOT/"data"/"tables"/"processed"

VARIANTS = {
 "hybrid": {
   "folder": "notebooks_hybrid",
   "title": "Hybrid weights (learned shape + ABS floor)",
   "weights": {"estimated_bnpl_revenue":0.70,"log_value_growth":0.02,
               "transactions_per_customer":0.0,"repeat_rate":0.0,"stability":0.0,
               "active_month_ratio":0.18,"customer_spread":0.05,
               "customer_income":0.03,"customer_irsad":0.02},
 },
 "machine": {
   "folder": "notebooks_machine",
   "title": "Pure learned weights (no floors)",
   "weights": {"estimated_bnpl_revenue":0.724,"log_value_growth":0.0,
               "transactions_per_customer":0.0,"repeat_rate":0.0,"stability":0.0,
               "active_month_ratio":0.218,"customer_spread":0.058,
               "customer_income":0.0,"customer_irsad":0.0},
 },
}

NOTEBOOKS = ["merchant_ranking_v2.ipynb","fraud_penalty.ipynb",
             "fraud_imputation_check.ipynb","segmentation.ipynb","final_findings.ipynb"]

# files each variant WRITES - these must never be symlinked
WRITTEN = {"merchant_ranking_v2_features.parquet","merchant_ranking_v2.parquet",
           "top100_v2.csv","weights_v2.csv","v2_holdout_results.csv",
           "merchant_risk","merchant_ranking_v2_fraud_adjusted.parquet",
           "top100_v2_fraud_adjusted.csv","top10_per_segment.csv"}

def wdict(w):
    width = max(len(k) for k in w) + 2
    body = "\n".join(f'    "{k}":{" "*(width-len(k))}{v},' for k, v in w.items())
    return "WEIGHTS_V2 = {\n" + body + "\n}"

for key, cfg in VARIANTS.items():
    vdir = ROOT/cfg["folder"]
    (vdir/"scripts").mkdir(parents=True, exist_ok=True)
    out = ROOT/"data"/"variants"/key/"processed"
    out.mkdir(parents=True, exist_ok=True)

    # symlink every shared input the variant reads but never writes
    n_link = 0
    for item in sorted(SHARED.iterdir()):
        if item.name in WRITTEN or item.name.startswith("."):
            continue
        link = out/item.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(Path("../../../tables/processed")/item.name)
            n_link += 1

    for nb_name in NOTEBOOKS:
        nb = json.loads((ROOT/"notebooks"/nb_name).read_text())
        n_patch = 0
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            s = "".join(cell["source"]); before = s
            # 1. send every read and write to this variant's own data folder
            s = re.sub(r'(PROCESSED(?:_PATH)?\s*=\s*PROJECT_ROOT\s*/\s*)"data"\s*/\s*"tables"\s*/\s*"processed"',
                       rf'\1"data" / "variants" / "{key}" / "processed"', s)
            # 2. use this variant's own copy of the helper script
            s = s.replace('SCRIPTS = PROJECT_ROOT / "scripts"',
                          f'SCRIPTS = PROJECT_ROOT / "{cfg["folder"]}" / "scripts"')
            # 3. the weights themselves
            if re.search(r'^WEIGHTS_V2\s*=\s*\{', s, re.M):
                s = re.sub(r'^WEIGHTS_V2\s*=\s*\{[^}]*\}', wdict(cfg["weights"]), s, flags=re.M)
            if s != before:
                cell["source"] = s.splitlines(keepends=True); n_patch += 1

        banner = (f"# VARIANT: {cfg['title']}\n\n"
                  f"This is a copy of `notebooks/{nb_name}`. **Three things differ and nothing else:**\n\n"
                  f"1. `WEIGHTS_V2` holds this variant's weights (set in `merchant_ranking_v2.ipynb`).\n"
                  f"2. Reads and writes go to `data/variants/{key}/processed/`, so the three\n"
                  f"   variants never overwrite each other. Shared inputs there are symlinks to\n"
                  f"   `data/tables/processed/`.\n"
                  f"3. `segmentation.ipynb` loads its helper from this folder's own `scripts/`.\n\n"
                  f"Every other line is identical to the original. Compare with `diff` if in doubt.\n")
        nb["cells"].insert(0, {"cell_type":"markdown","metadata":{},
                               "source":banner.splitlines(keepends=True)})
        (vdir/nb_name).write_text(json.dumps(nb, indent=1))
        print(f"  {key}/{nb_name}: {n_patch} sel dipatch")

    # the helper script, with the same three changes
    sc = (ROOT/"scripts"/"top10_per_segment.py").read_text()
    sc = sc.replace("PROJECT_ROOT = SCRIPT_DIR.parent",
                    "PROJECT_ROOT = SCRIPT_DIR.parents[1]   # variant copy: scripts/ sits one level deeper")
    sc = sc.replace('PROCESSED = PROJECT_ROOT / "data" / "tables" / "processed"',
                    f'PROCESSED = PROJECT_ROOT / "data" / "variants" / "{key}" / "processed"')
    sc = re.sub(r'^WEIGHTS_V2\s*=\s*\{[^}]*\}', wdict(cfg["weights"]), sc, flags=re.M)
    (vdir/"scripts"/"top10_per_segment.py").write_text(sc)

    print(f"{key}: {n_link} symlink input, folder {cfg['folder']}/\n")
