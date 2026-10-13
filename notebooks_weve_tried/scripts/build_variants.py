"""Regenerate the optional fixed-Machine notebooks from the current main pipeline.

Run explicitly after editing main notebooks. Shared data links are prepared here; generated outputs are never symlinked.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
PAIRS = {'merchant_ranking_hybrid': 'merchant_ranking_machine',
         'fraud_penalty': 'fraud_penalty_machine',
         'fraud_imputation_check': 'fraud_imputation_check_machine',
         'segmentation': 'segmentation_machine', 'final_findings': 'final_findings_machine'}


INPUTS = ["transactions_clean", "merchants_clean", "merchant_features", "consumers",
          "consumers_sa2", "consumer_fraud_probability", "merchant_fraud_probability",
          "duplicate_order_ids", "merchant_behaviour_segments", "merchant_segments"]


def prepare_inputs():
    target = ROOT / "data" / "variants" / "machine" / "processed"
    target.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        source, link = ROOT / "data" / "tables" / "processed" / name, target / name
        if not source.exists():
            raise FileNotFoundError(source)
        if link.exists() or link.is_symlink():
            if link.resolve() != source.resolve():
                raise RuntimeError(f"Refusing to replace existing variant input: {link}")
        else:
            link.symlink_to(source, target_is_directory=source.is_dir())
    print(f"Prepared shared inputs in {target}")



def main():
    prepare_inputs()
    for source, name in PAIRS.items():
        nb = json.loads((ROOT / 'notebooks' / f'{source}.ipynb').read_text())
        for cell in nb['cells']:
            text = ''.join(cell['source'])
            if cell['cell_type'] == 'code':
                text = text.replace('/ "data" / "tables" / "processed"',
                                    '/ "data" / "variants" / "machine" / "processed"')
                text = text.replace('SCRIPTS = PROJECT_ROOT / "scripts"',
                                    'SCRIPTS = PROJECT_ROOT / "notebooks_weve_tried" / "scripts"')
                for a, b in [('merchant_ranking_hybrid', 'merchant_ranking_machine'),
                             ('top100_hybrid', 'top100_machine'), ('weights_hybrid', 'weights_machine'),
                             ('hybrid_holdout_results', 'machine_holdout_results'),
                             ('hybrid_fraud_sensitivity', 'machine_fraud_sensitivity'),
                             ('top{TOP_N}_hybrid', 'top{TOP_N}_machine')]:
                    text = text.replace(a, b)
                if source == 'merchant_ranking_hybrid' and 'def score_hybrid(' in text:
                    text = 'WEIGHTS_HYBRID = WEIGHTS_MACHINE\n' + text
                text = text.replace('K, REVENUE_WEIGHT = 300, 0.70',
                                    'K, REVENUE_WEIGHT = 300, 0.724')
                text = text.replace('hybrid (frozen before holdout)', 'machine (frozen before holdout)')
                text = text.replace('hybrid (shipped)', 'machine (comparison)')
                cell['outputs'] = []
                cell['execution_count'] = None
            else:
                text = text.replace('Hybrid', 'Machine').replace('hybrid', 'machine')
                if source == 'merchant_ranking_hybrid' and text.startswith('## 4.'):
                    text = ('## 4. Fixed Machine weights\n\nRevenue 72.4%, activity 21.8%, '
                            'customer spread 5.8%; all other terms zero. The fixed comparator '
                            'is not overwritten by current rolling fits.\n')
                if source == 'final_findings' and text.startswith('## Market segments'):
                    text = ('## Market segments\n\nBehavioural clusters are shared with the main pipeline. '
                            'Machine uses revenue 72.4%, activity 21.8% and customer spread 5.8%; '
                            'all other dimensions are diagnostic only.\n')
                if source == 'final_findings' and text.startswith('## Ranking model'):
                    text = ('## Ranking model\n\nThe fixed Machine comparison uses the shared '
                            'rank-space scoring rules and 72.4% revenue, 21.8% activity, '
                            '5.8% customer spread. September uses history-only predictors '
                            'but is inspected across experiments.\n')
            cell['source'] = text.splitlines(keepends=True)
        nb['cells'][0]['source'] = [
            '# Fixed Machine comparison\n\n',
            'Historical fixed weights: revenue 72.4%, activity 21.8%, spread 5.8%.\n',
            'These are a separate frozen comparator, not the current rolling-fit mean.\n',
            'All feature windows and scoring rules are shared with the main pipeline.\n',
            'Writes go to data/variants/machine/processed; diagnostic models are shared.\n',
            'September is an exploratory evaluation reused across comparisons.\n']
        (ROOT / 'notebooks_weve_tried' / f'{name}.ipynb').write_text(
            json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
    text = (ROOT / 'scripts/top10_per_segment.py').read_text()
    text = text.replace('PROJECT_ROOT = SCRIPT_DIR.parent',
                        'PROJECT_ROOT = SCRIPT_DIR.parents[1]\nimport sys\nsys.path.insert(0, str(PROJECT_ROOT / "scripts"))')
    text = text.replace('/ "data" / "tables" / "processed"',
                        '/ "data" / "variants" / "machine" / "processed"')
    text = text.replace('merchant_ranking_hybrid', 'merchant_ranking_machine')
    text = text.replace('SEGMENT_WEIGHTS =', 'WEIGHTS_HYBRID = WEIGHTS_MACHINE\nSEGMENT_WEIGHTS =')
    (ROOT / 'notebooks_weve_tried/scripts/top10_per_segment.py').write_text(text)
    print('Updated five Machine notebooks and the variant segment helper')


if __name__ == '__main__':
    main()
