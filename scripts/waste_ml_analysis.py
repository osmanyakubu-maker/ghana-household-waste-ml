import argparse, zipfile, pandas as pd, numpy as np, time, json, os
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, log_loss, confusion_matrix, average_precision_score
from sklearn.base import clone

parser=argparse.ArgumentParser(description='Reproduce the secondary ten-route district-held-out analysis.')
parser.add_argument('--data-zip',required=True,help='Path to the GSS household-and-sanitation ZIP file.')
parser.add_argument('--csv-name',default='housingandsanitation_10%_20221011d.csv',help='CSV member name inside the ZIP.')
parser.add_argument('--output-dir',default='waste_ml_results',help='Directory for aggregate outputs.')
args=parser.parse_args()
ZIP=str(Path(args.data_zip).expanduser().resolve())
INFILE=args.csv_name
OUT=str(Path(args.output_dir).expanduser().resolve())
os.makedirs(OUT,exist_ok=True)

# Primary features are household composition, location at region/locality scale,
# and housing conditions. All sanitation-module variables (S01-S08) and all
# record identifiers/fine spatial IDs are excluded from predictors.
num=['usual_members','visitors','absent_members','totpopn','h07a','h07b','h11b']
cat=['region','urbrur','h01','h02','h03','h04','h05','h06','h08a','h08b','h09a','h09b','h10','h11a','h11c','h12',
     'h13a1','h13a2','h13b1','h13b2','h13c1','h13c2','h13d1','h13d2','h13e1','h13e2','h13f1','h13f2',
     'h13g1','h13g2','h13h1','h13h2','h13i1','h13i2','h13j1','h13j2','h13k1','h13k2','h13l1','h13l2',
     'h13m1','h13m2','h13n1','h13n2','h13o1','h13o2','h13p1','h13p2','h13q1','h13q2','h13r1','h13r2','h13s1','h13s2']
allcols=['s02','restype','distcode','weight']+num+cat
with zipfile.ZipFile(ZIP) as z:
  with z.open(INFILE) as f:
    d=pd.read_csv(f,usecols=lambda c:c in allcols,low_memory=False,dtype='string')
print('read',d.shape,flush=True)
d=d[d['restype'].eq('Occupied housing unit')].copy()
d=d[d.s02.notna() & d.s02.str.strip().ne('')].copy()
d.s02=d.s02.str.strip()
d['weight']=pd.to_numeric(d.weight,errors='coerce')
d=d[d.weight.notna() & d.weight.gt(0) & d.distcode.notna()].copy()
for c in num: d[c]=pd.to_numeric(d[c],errors='coerce')
for c in cat: d[c]=d[c].fillna('<Missing>').astype(str)
# Do not duplicate household-size information: totpopn overlaps usual_members and visitors.
# Retain usual_members (core household composition), not totpopn/visitor counts.
num=['usual_members','h07a','h07b','h11b']
features=num+cat
# protect against columns absent in variant
features=[c for c in features if c in d]
y=d.s02.to_numpy()
groups=d.distcode.to_numpy()
w=d.weight.to_numpy(float)
labels=sorted(pd.unique(y))
print('rows',len(d),'districts',pd.Series(groups).nunique(),'classes',labels,flush=True)
print('class counts',d.s02.value_counts().to_dict(),flush=True)
print('weights',d.weight.describe().to_dict(),flush=True)

num_pipe=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler(with_mean=False))])
cat_pipe=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=30,dtype=np.float32,sparse_output=True))])
prep=ColumnTransformer([('num',num_pipe,num),('cat',cat_pipe,cat)],sparse_threshold=1.0)
# Weights applied in fitting and every reported metric; no synthetic balancing.
gbm_prep=ColumnTransformer([
    ('num',num_pipe,num),
    ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                     ('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=30,dtype=np.float32,sparse_output=False))]),cat)
],remainder='drop')
models={
 'Weighted logistic SGD': Pipeline([('prep',clone(prep)),('model',SGDClassifier(loss='log_loss',alpha=1e-5,max_iter=15,tol=1e-3,average=True,random_state=2026))]),
 'Weighted histogram gradient boosting': Pipeline([('prep',gbm_prep),('model',HistGradientBoostingClassifier(max_iter=60,learning_rate=0.08,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=1.0,early_stopping=True,validation_fraction=0.1,n_iter_no_change=8,random_state=2026))])
}
cv=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2026)
results=[]
pred_tables=[]
start=time.time()
for fold,(tr,te) in enumerate(cv.split(d[features],y,groups),1):
  print('fold',fold,'train',len(tr),'test',len(te),'districts',pd.Series(groups[te]).nunique(),flush=True)
  rng=np.random.default_rng(2026+fold)
  fit_tr=np.sort(rng.choice(tr,size=min(200000,len(tr)),replace=False))
  # Weighted majority-prior baseline without passing through the feature pipeline.
  dummy=DummyClassifier(strategy='prior').fit(np.zeros((len(tr),1)),y[tr],sample_weight=w[tr])
  dp=dummy.predict(np.zeros((len(te),1))); dprob=dummy.predict_proba(np.zeros((len(te),1)))
  db={'fold':fold,'model':'Prior baseline','n_train':len(tr),'n_test':len(te),
      'accuracy_w':accuracy_score(y[te],dp,sample_weight=w[te]),
      'macro_f1_w':f1_score(y[te],dp,labels=labels,average='macro',sample_weight=w[te],zero_division=0),
      'macro_recall_w':recall_score(y[te],dp,labels=labels,average='macro',sample_weight=w[te],zero_division=0),
      'weighted_f1_w':f1_score(y[te],dp,average='weighted',sample_weight=w[te],zero_division=0),
      'log_loss_w':log_loss(y[te],dprob,labels=dummy.classes_,sample_weight=w[te]),
      'macro_ap_w':float(np.mean([average_precision_score((y[te]==c).astype(int),dprob[:,j],sample_weight=w[te]) for j,c in enumerate(dummy.classes_)])),
      'collection_f1_w':0.0,'collection_recall_w':0.0,'fit_seconds':0.0}
  results.append(db)
  pd.DataFrame({'actual':y[te],'predicted':dp,'district':groups[te],'weight':w[te],'fold':fold,'model':'Prior baseline'}).to_csv(os.path.join(OUT,f'pred_{fold}_Prior.csv'),index=False)
  for name,model in models.items():
    t=time.time(); model.fit(d.iloc[fit_tr][features],y[fit_tr],model__sample_weight=w[fit_tr])
    pred=model.predict(d.iloc[te][features])
    prob=model.predict_proba(d.iloc[te][features])
    out={'fold':fold,'model':name,'n_train':len(tr),'n_train_fit':len(fit_tr),'n_test':len(te),
         'accuracy_w':accuracy_score(y[te],pred,sample_weight=w[te]),
         'macro_f1_w':f1_score(y[te],pred,labels=labels,average='macro',sample_weight=w[te],zero_division=0),
         'macro_recall_w':recall_score(y[te],pred,labels=labels,average='macro',sample_weight=w[te],zero_division=0),
         'weighted_f1_w':f1_score(y[te],pred,average='weighted',sample_weight=w[te],zero_division=0),
         'log_loss_w':log_loss(y[te],prob,labels=model.classes_,sample_weight=w[te])}
    aps=[]
    for j,c in enumerate(model.classes_):
      aps.append(average_precision_score((y[te]==c).astype(int),prob[:,j],sample_weight=w[te]))
    out['macro_ap_w']=float(np.mean(aps))
    # Secondary binary classification: collection-based routes from the questionnaire.
    collection={'Compaction truck','Other vehicles','Tricycle','Central container','Push carts/Walk-in attendant/Bicycle/Wheelbarrow'}
    yb=np.isin(y[te],list(collection)); pb=np.isin(pred,list(collection))
    out['collection_f1_w']=f1_score(yb,pb,sample_weight=w[te],zero_division=0)
    out['collection_recall_w']=recall_score(yb,pb,sample_weight=w[te],zero_division=0)
    out['fit_seconds']=time.time()-t
    results.append(out)
    pd.DataFrame({'actual':y[te],'predicted':pred,'district':groups[te],'weight':w[te],'fold':fold,'model':name}).to_csv(os.path.join(OUT,f'pred_{fold}_{name.split()[1]}.csv'),index=False)
    print(json.dumps(out),flush=True)
  # weighted class distributions by validation fold
  pd.DataFrame({'y':y[te],'w':w[te]}).groupby('y',dropna=False).w.sum().to_csv(os.path.join(OUT,f'class_support_fold_{fold}.csv'))

res=pd.DataFrame(results)
res.to_csv(os.path.join(OUT,'fold_metrics.csv'),index=False)
summary=res.groupby('model').agg({c:['mean','std'] for c in ['accuracy_w','macro_f1_w','macro_recall_w','weighted_f1_w','macro_ap_w','log_loss_w','collection_f1_w','collection_recall_w']})
summary.to_csv(os.path.join(OUT,'summary_metrics.csv'))
# S02 prevalence and weighted distribution on the retained occupied households.
dist=d.groupby('s02').agg(n=('s02','size'),weighted_n=('weight','sum')).sort_values('n',ascending=False)
dist['weighted_share']=dist.weighted_n/dist.weighted_n.sum();dist['sample_share']=dist.n/dist.n.sum()
dist.to_csv(os.path.join(OUT,'target_distribution.csv'))
# Fold-combined confusion and per-class metrics for both substantive models.
for name in ['Weighted logistic SGD','Weighted histogram gradient boosting']:
  p=[]
  for fold in range(1,5):
    ff=pd.read_csv(os.path.join(OUT,f'pred_{fold}_{name.split()[1]}.csv'))
    p.append(ff)
  q=pd.concat(p,ignore_index=True)
  cm=confusion_matrix(q.actual,q.predicted,labels=labels,sample_weight=q.weight)
  pd.DataFrame(cm,index=labels,columns=labels).to_csv(os.path.join(OUT,f'confusion_{name.split()[1]}.csv'))
  pr,rc,f1,sup=__import__('sklearn.metrics',fromlist=['precision_recall_fscore_support']).precision_recall_fscore_support(q.actual,q.predicted,labels=labels,sample_weight=q.weight,zero_division=0)
  pd.DataFrame({'class':labels,'weighted_precision':pr,'weighted_recall':rc,'weighted_f1':f1,'weighted_support':sup}).to_csv(os.path.join(OUT,f'per_class_{name.split()[1]}.csv'),index=False)
print('elapsed',round(time.time()-start,1),'s',flush=True)
