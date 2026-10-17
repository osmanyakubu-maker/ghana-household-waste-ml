"""Leave-one-region-out sensitivity excluding the region predictor itself."""
import argparse
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


parser = argparse.ArgumentParser()
parser.add_argument('--data-zip', required=True)
parser.add_argument('--csv-name', default='housingandsanitation_10%_20221011d.csv')
parser.add_argument('--output', default='leave_one_region_out_no_region.csv')
args = parser.parse_args()

num = ['usual_members', 'h07a', 'h07b', 'h11b']
cat = ['urbrur','h01','h02','h03','h04','h05','h06','h08a','h08b','h09a','h09b','h10','h11a','h11c','h12',
       'h13a1','h13a2','h13b1','h13b2','h13c1','h13c2','h13d1','h13d2','h13e1','h13e2','h13f1','h13f2',
       'h13g1','h13g2','h13h1','h13h2','h13i1','h13i2','h13j1','h13j2','h13k1','h13k2','h13l1','h13l2',
       'h13m1','h13m2','h13n1','h13n2','h13o1','h13o2','h13p1','h13p2','h13q1','h13q2','h13r1','h13r2','h13s1','h13s2']
features = num + cat
use = set(['s02','restype','distcode','weight','region'] + features)
with zipfile.ZipFile(Path(args.data_zip).expanduser().resolve()) as z:
    with z.open(args.csv_name) as f:
        d = pd.read_csv(f, usecols=lambda c: c in use, dtype='string', low_memory=False)
d = d[d.restype.eq('Occupied housing unit') & d.s02.notna() & d.s02.str.strip().ne('') & d.distcode.notna()].copy()
d.s02 = d.s02.str.strip(); d.weight = pd.to_numeric(d.weight, errors='coerce'); d = d[d.weight.gt(0)].copy()
for c in num: d[c] = pd.to_numeric(d[c], errors='coerce')
for c in cat: d[c] = d[c].fillna('<Missing>').astype(str)
collection = {'Compaction truck','Other vehicles','Tricycle','Central container','Push carts/Walk-in attendant/Bicycle/Wheelbarrow'}
y = d.s02.isin(collection).astype(np.int8).to_numpy(); w = d.weight.to_numpy(float)
regions = d.region.astype(str).to_numpy(); X = d[features]
prep = ColumnTransformer([
    ('num', SimpleImputer(strategy='median', add_indicator=True), num),
    ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                      ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                                             encoded_missing_value=-1, dtype=np.float32))]), cat)
])
rows = []
for i, region in enumerate(sorted(pd.unique(regions))):
    test = np.flatnonzero(regions == region); train = np.flatnonzero(regions != region)
    fit = np.sort(np.random.default_rng(9000 + i).choice(train, size=min(100000, len(train)), replace=False))
    model = Pipeline([('prep', clone(prep)), ('model', HistGradientBoostingClassifier(
        max_iter=70, learning_rate=.07, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, early_stopping=True, validation_fraction=.1,
        n_iter_no_change=10, random_state=2026))])
    model.fit(X.iloc[fit], y[fit], model__sample_weight=w[fit])
    prob = model.predict_proba(X.iloc[test])[:, 1]
    rows.append({'held_out_region': region, 'n_test': len(test),
                 'prevalence': np.average(y[test], weights=w[test]),
                 'roc_auc_no_region': roc_auc_score(y[test], prob, sample_weight=w[test]),
                 'average_precision_no_region': average_precision_score(y[test], prob, sample_weight=w[test]),
                 'brier_no_region': brier_score_loss(y[test], prob, sample_weight=w[test])})
    print(region, rows[-1]['roc_auc_no_region'], flush=True)
pd.DataFrame(rows).to_csv(Path(args.output).expanduser().resolve(), index=False)
