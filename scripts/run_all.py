"""One-command entry point for the primary and secondary manuscript analyses."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent if ROOT.name == 'scripts' else ROOT


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Run all analyses used in the Ghana household-waste manuscript.')
    parser.add_argument('--data-zip', required=True, type=Path)
    parser.add_argument('--output-root', default=PACKAGE_ROOT / 'outputs', type=Path)
    parser.add_argument('--skip-secondary', action='store_true', help='Run only the primary binary analysis.')
    args = parser.parse_args()
    cfg = json.loads((PACKAGE_ROOT / 'analysis_config.json').read_text(encoding='utf-8'))
    data_zip = args.data_zip.expanduser().resolve()
    if not data_zip.exists():
        raise SystemExit(f'Data file not found: {data_zip}')
    observed = sha256(data_zip)
    if observed != cfg['expected_zip_sha256']:
        raise SystemExit(f'Data checksum mismatch. Expected {cfg["expected_zip_sha256"]}; observed {observed}.')
    args.output_root.mkdir(parents=True, exist_ok=True)
    primary = args.output_root / cfg['primary_output_dir']
    secondary = args.output_root / cfg['secondary_output_dir']
    subprocess.run([sys.executable, str(ROOT / 'robust_waste_analysis.py'),
                    '--data-zip', str(data_zip), '--csv-name', cfg['csv_member'],
                    '--output-dir', str(primary)], check=True)
    subprocess.run([sys.executable, str(ROOT / 'region_transfer_sensitivity.py'),
                    '--data-zip', str(data_zip), '--csv-name', cfg['csv_member'],
                    '--output', str(primary / 'leave_one_region_out_no_region.csv')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'strengthening_analysis.py'),
                    '--data-zip', str(data_zip), '--csv-name', cfg['csv_member'],
                    '--oof-path', str(primary / 'primary_oof_predictions.csv'),
                    '--existing-metrics', str(primary / 'repeated_model_metrics.csv'),
                    '--output-dir', str(primary)], check=True)
    if not args.skip_secondary:
        subprocess.run([sys.executable, str(ROOT / 'waste_ml_analysis.py'),
                        '--data-zip', str(data_zip), '--csv-name', cfg['csv_member'],
                        '--output-dir', str(secondary)], check=True)
    subprocess.run([sys.executable, str(ROOT / 'verify_results.py'),
                    '--primary-dir', str(primary),
                    '--secondary-dir', str(secondary) if not args.skip_secondary else ''], check=True)


if __name__ == '__main__':
    main()
