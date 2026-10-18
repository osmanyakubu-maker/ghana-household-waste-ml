# Ghana household waste machine-learning reproducibility package

This package accompanies the revised manuscript. It contains analysis scripts, aggregate results and publication figures. It intentionally excludes the licensed/public-use household microdata and household-level predictions. Obtain catalogue 110 from the Ghana Statistical Service National Data Archive. The expected ZIP SHA-256 is recorded in analysis_config.json; the runner stops if the file differs. Household-level outputs are not included.

Primary design: 3 × 4 repeated district-group cross-validation for survey-weighted histogram gradient boosting; first-repeat district-cluster bootstrap; three-repeat calibration, nested threshold selection, learning curves and permutation importance; 16 leave-one-region-out tests; complete 12-fold comparator evaluation; audit-budget utility; urban-rural subgroup performance; regional calibration; and weighting sensitivity.

Environment: install the exact packages in requirements-lock.txt. The tested versions are pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0 and matplotlib 3.10.8.

One-command reproduction from the package root:

    python scripts/run_all.py --data-zip /path/to/housingandsanitation_10%_20221011d.zip

The runner validates the source checksum, executes both analyses and calls verify_results.py. The verifier checks the number of outer folds and all headline discrimination, calibration and locked-threshold estimates against manuscript values within explicit numerical tolerances. To verify the archived aggregate outputs without rerunning models, use:

    python scripts/verify_results.py --primary-dir aggregate_results/primary --secondary-dir aggregate_results/secondary

Publication figures are supplied as 600-dpi, LZW-compressed RGB TIFF files. Captions are in the main manuscript for Figures 1–3 and in the separate supplementary document for Supplementary Figures S1–S3; captions are not embedded in the images:

- `Figure_1_regional_collection_prevalence.tiff`
- `Figure_2_calibration_curve.tiff`
- `Figure_3_regional_transfer_and_learning_curve.tiff` (panels a-b)
- `Figure_4_permutation_importance.tiff` (Supplementary Figure S1)
- `Figure_5_multiclass_route_performance.tiff` (Supplementary Figure S2)
- `Figure_6_audit_budget_and_regional_calibration.tiff` (Supplementary Figure S3; panels a-b)

The verifier also checks the sample and district totals recoverable from the outer folds, reported uncertainty limits, comparator differences, regional performance ranges, audit-budget utility, subgroup results and the secondary ten-route macro-F1 result.
