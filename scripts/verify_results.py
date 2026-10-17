"""Verify that regenerated aggregate results match the manuscript within tolerances."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


EXPECTED = {
    'hgb_auc': (0.8737476384, 5e-6),
    'hgb_ap': (0.7727340269, 5e-6),
    'hgb_f1': (0.6844316267, 5e-6),
    'hgb_auc_sd': (0.010, 5e-4),
    'hgb_ap_sd': (0.021, 5e-4),
    'hgb_f1_sd': (0.022, 5e-4),
    'hgb_brier': (0.135, 5e-4),
    'hgb_brier_sd': (0.006, 5e-4),
    'hgb_mcc': (0.542, 5e-4),
    'hgb_mcc_sd': (0.026, 5e-4),
    'cal_intercept': (0.0357199, 5e-5),
    'cal_slope': (1.035579, 5e-5),
    'cal_ece': (0.006364, 5e-5),
    'locked_recall': (0.802455, 5e-5),
    'locked_f1': (0.710404, 5e-5),
    'no_region_auc': (0.8188785, 5e-6),
    'rf_auc_12fold': (0.8703815, 5e-6),
    'logit_auc_12fold': (0.8690000, 5e-6),
    'budget10_precision': (0.9880702, 5e-6),
    'budget10_recall': (0.1484127, 5e-6),
    'rural_auc': (0.7882961, 5e-6),
    'urban_auc': (0.7806851, 5e-6),
    'hgb_minus_rf_auc': (0.0033661, 5e-6),
    'hgb_minus_logit_auc': (0.0047477, 5e-6),
    'logit_minus_hgb_f1': (0.0074186, 5e-6),
    'bootstrap_auc_low': (0.8550442, 5e-6),
    'bootstrap_auc_high': (0.8884949, 5e-6),
    'region_auc_min': (0.7434284, 5e-6),
    'region_auc_max': (0.8981487, 5e-6),
    'region_ece_min': (0.0127, 5e-4),
    'region_ece_max': (0.3138742, 5e-6),
    'budget10_lift': (1.4841143, 5e-6),
    'rural_recall': (0.0425516, 5e-6),
    'multiclass_macro_f1': (0.2295595, 5e-6),
}


def check(name, value):
    target, tol = EXPECTED[name]
    if abs(value - target) > tol:
        raise AssertionError(f'{name}: expected {target} ± {tol}, observed {value}')
    print(f'PASS {name}: {value:.6f}')


def check_finite_table(path):
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.size and not np.isfinite(numeric.to_numpy()).all():
        raise AssertionError(f'Non-finite numeric value found in {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary-dir', required=True, type=Path)
    parser.add_argument('--secondary-dir', default='', type=str)
    args = parser.parse_args()
    m = pd.read_csv(args.primary_dir / 'repeated_model_metrics.csv')
    h = m[m['model'].eq('Histogram gradient boosting')]
    if len(h) != 12:
        raise AssertionError(f'Expected 12 HGB outer folds; observed {len(h)}')
    check('hgb_auc', h.roc_auc.mean())
    check('hgb_ap', h.average_precision.mean())
    check('hgb_f1', h.f1.mean())
    check('hgb_auc_sd', h.roc_auc.std(ddof=1))
    check('hgb_ap_sd', h.average_precision.std(ddof=1))
    check('hgb_f1_sd', h.f1.std(ddof=1))
    check('hgb_brier', h.brier.mean())
    check('hgb_brier_sd', h.brier.std(ddof=1))
    check('hgb_mcc', h.mcc.mean())
    check('hgb_mcc_sd', h.mcc.std(ddof=1))
    first_repeat = h[h['repeat'].eq(1)]
    if int(first_repeat.n_test.sum()) != 835697:
        raise AssertionError(f'Expected 835697 eligible households; observed {int(first_repeat.n_test.sum())}')
    if int(first_repeat.n_test_districts.sum()) != 261:
        raise AssertionError(f'Expected 261 districts; observed {int(first_repeat.n_test_districts.sum())}')
    c = pd.read_csv(args.primary_dir / 'calibration_summary.csv').iloc[0]
    check('cal_intercept', c.calibration_intercept_mean)
    check('cal_slope', c.calibration_slope_mean)
    check('cal_ece', c.ece_10bin_mean)
    t = pd.read_csv(args.primary_dir / 'locked_threshold_metrics.csv')
    if len(t) != 12:
        raise AssertionError(f'Expected 12 locked-threshold outer folds; observed {len(t)}')
    check('locked_recall', t.recall.mean())
    check('locked_f1', t.f1.mean())
    nr = pd.read_csv(args.primary_dir / 'leave_one_region_out_no_region.csv')
    if len(nr) != 16:
        raise AssertionError(f'Expected 16 no-region transfer tests; observed {len(nr)}')
    check('no_region_auc', nr.roc_auc_no_region.mean())
    regional = pd.read_csv(args.primary_dir / 'leave_one_region_out.csv')
    if len(regional) != 16:
        raise AssertionError(f'Expected 16 regional transfer tests; observed {len(regional)}')
    check('region_auc_min', regional.roc_auc.min())
    check('region_auc_max', regional.roc_auc.max())
    extended = pd.read_csv(args.primary_dir / 'extended_model_metrics.csv')
    counts = extended.groupby('model').size()
    for model in ['Histogram gradient boosting', 'Random forest', 'Regularized logistic SGD',
                  'Weighted prevalence baseline']:
        if counts.get(model, 0) != 12:
            raise AssertionError(f'Expected 12 folds for {model}; observed {counts.get(model, 0)}')
    check('rf_auc_12fold', extended[extended.model.eq('Random forest')].roc_auc.mean())
    check('logit_auc_12fold', extended[extended.model.eq('Regularized logistic SGD')].roc_auc.mean())
    paired = pd.read_csv(args.primary_dir / 'paired_model_differences.csv')
    paired_key = paired.set_index(['comparison', 'metric']).mean_difference
    check('hgb_minus_rf_auc', paired_key.loc[('HGB minus Random forest', 'roc_auc')])
    check('hgb_minus_logit_auc', paired_key.loc[('HGB minus Regularized logistic SGD', 'roc_auc')])
    check('logit_minus_hgb_f1', -paired_key.loc[('HGB minus Regularized logistic SGD', 'f1')])
    bootstrap = pd.read_csv(args.primary_dir / 'district_bootstrap_ci.csv', index_col=0)
    check('bootstrap_auc_low', bootstrap.loc['roc_auc', 'ci_low'])
    check('bootstrap_auc_high', bootstrap.loc['roc_auc', 'ci_high'])
    budget = pd.read_csv(args.primary_dir / 'inspection_budget.csv')
    budget10 = budget.loc[budget.inspection_budget.eq(.10)].iloc[0]
    check('budget10_precision', budget10.precision_gap)
    check('budget10_recall', budget10.recall_gap)
    check('budget10_lift', budget10.lift_over_random)
    subgroup = pd.read_csv(args.primary_dir / 'urban_rural_performance.csv').set_index('subgroup')
    check('rural_auc', subgroup.loc['Rural', 'roc_auc'])
    check('urban_auc', subgroup.loc['Urban', 'roc_auc'])
    check('rural_recall', subgroup.loc['Rural', 'recall'])
    transfer_cal = pd.read_csv(args.primary_dir / 'regional_transfer_calibration.csv')
    check('region_ece_min', transfer_cal.ece_10bin.min())
    check('region_ece_max', transfer_cal.ece_10bin.max())
    if args.secondary_dir:
        s = Path(args.secondary_dir) / 'summary_metrics.csv'
        if not s.exists():
            raise AssertionError(f'Missing secondary summary: {s}')
        secondary = pd.read_csv(s, header=[0, 1], index_col=0)
        check('multiclass_macro_f1', secondary.loc['Weighted histogram gradient boosting', ('macro_f1_w', 'mean')])
    for path in sorted(args.primary_dir.glob('*.csv')):
        check_finite_table(path)
    if args.secondary_dir:
        for path in sorted(Path(args.secondary_dir).glob('*.csv')):
            check_finite_table(path)
    print('All manuscript headline checks passed.')


if __name__ == '__main__':
    main()
