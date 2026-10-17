import argparse, os, time, json, zipfile, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss,
    matthews_corrcoef, log_loss)
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore', category=FutureWarning)
parser=argparse.ArgumentParser(description='Reproduce the survey-weighted district-held-out binary analysis.')
parser.add_argument('--data-zip',required=True,help='Path to the GSS household-and-sanitation ZIP file.')
parser.add_argument('--csv-name',default='housingandsanitation_10%_20221011d.csv',help='CSV member name inside the ZIP.')
parser.add_argument('--output-dir',default='robust_results',help='Directory for aggregate outputs.')
args=parser.parse_args()
ZIP=str(Path(args.data_zip).expanduser().resolve())
IN=args.csv_name
OUT=str(Path(args.output_dir).expanduser().resolve()); os.makedirs(OUT,exist_ok=True)
SEED=2026

num=['usual_members','h07a','h07b','h11b']
cat=['region','urbrur','h01','h02','h03','h04','h05','h06','h08a','h08b','h09a','h09b','h10','h11a','h11c','h12',
     'h13a1','h13a2','h13b1','h13b2','h13c1','h13c2','h13d1','h13d2','h13e1','h13e2','h13f1','h13f2',
     'h13g1','h13g2','h13h1','h13h2','h13i1','h13i2','h13j1','h13j2','h13k1','h13k2','h13l1','h13l2',
     'h13m1','h13m2','h13n1','h13n2','h13o1','h13o2','h13p1','h13p2','h13q1','h13q2','h13r1','h13r2','h13s1','h13s2']
features=num+cat
use=set(['s02','restype','distcode','weight']+features)
with zipfile.ZipFile(ZIP) as z:
    with z.open(IN) as f:
        d=pd.read_csv(f,usecols=lambda c:c in use,dtype='string',low_memory=False)
d=d[d.restype.eq('Occupied housing unit') & d.s02.notna() & d.s02.str.strip().ne('') & d.distcode.notna()].copy()
d.s02=d.s02.str.strip(); d.weight=pd.to_numeric(d.weight,errors='coerce'); d=d[d.weight.gt(0)].copy()
for c in num: d[c]=pd.to_numeric(d[c],errors='coerce')
for c in cat: d[c]=d[c].fillna('<Missing>').astype(str)
collection={'Compaction truck','Other vehicles','Tricycle','Central container','Push carts/Walk-in attendant/Bicycle/Wheelbarrow'}
y=d.s02.isin(collection).astype(np.int8).to_numpy(); w=d.weight.to_numpy(float)
groups=d.distcode.astype(str).to_numpy(); regions=d.region.astype(str).to_numpy()
X=d[features]
print('loaded',len(d),'districts',pd.Series(groups).nunique(),'regions',pd.Series(regions).nunique(),flush=True)

ordinal_prep=ColumnTransformer([
    ('num',SimpleImputer(strategy='median',add_indicator=True),num),
    ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                     ('enc',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,encoded_missing_value=-1,dtype=np.float32))]),cat)
],remainder='drop')
nnum=len(num) # no numeric missing-indicator columns are expected in this extract; categorical mask set after fit if needed

def make_hgb(max_iter=80):
    return Pipeline([('prep',clone(ordinal_prep)),('model',HistGradientBoostingClassifier(
        max_iter=max_iter,learning_rate=.07,max_leaf_nodes=31,min_samples_leaf=100,
        l2_regularization=1.0,early_stopping=True,validation_fraction=.1,n_iter_no_change=10,
        random_state=SEED))])

onehot_prep=ColumnTransformer([
    ('num',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler(with_mean=False))]),num),
    ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                     ('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=30,dtype=np.float32))]),cat)
],sparse_threshold=1.0)

def metrics(y0,p,prob,sw,thr=.5):
    return {'accuracy':accuracy_score(y0,p,sample_weight=sw),
            'balanced_accuracy':balanced_accuracy_score(y0,p,sample_weight=sw),
            'precision':precision_score(y0,p,sample_weight=sw,zero_division=0),
            'recall':recall_score(y0,p,sample_weight=sw,zero_division=0),
            'f1':f1_score(y0,p,sample_weight=sw,zero_division=0),
            'roc_auc':roc_auc_score(y0,prob,sample_weight=sw),
            'average_precision':average_precision_score(y0,prob,sample_weight=sw),
            'brier':brier_score_loss(y0,prob,sample_weight=sw),
            'mcc':matthews_corrcoef(y0,p,sample_weight=sw),
            'log_loss':log_loss(y0,np.c_[1-prob,prob],sample_weight=sw,labels=[0,1]),
            'threshold':thr}

all_rows=[]; all_oof=[]; threshold_rows=[]
seeds=[2026,3031,4049]
t0=time.time()
for rep,seed in enumerate(seeds,1):
    cv=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed)
    for fold,(tr,te) in enumerate(cv.split(X,y,groups),1):
        rng=np.random.default_rng(seed+fold)
        fitidx=np.sort(rng.choice(tr,size=min(200000,len(tr)),replace=False))
        # Threshold locked using an inner district split for every outer fold.
        gss=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=seed+100+fold)
        ii,iv=next(gss.split(X.iloc[fitidx],y[fitidx],groups[fitidx]))
        inner=make_hgb(60); inner.fit(X.iloc[fitidx[ii]],y[fitidx[ii]],model__sample_weight=w[fitidx[ii]])
        vp=inner.predict_proba(X.iloc[fitidx[iv]])[:,1]
        cand=np.arange(.15,.851,.01)
        fs=[f1_score(y[fitidx[iv]],vp>=t,sample_weight=w[fitidx[iv]]) for t in cand]
        locked=float(cand[int(np.argmax(fs))])
        model=make_hgb(80); model.fit(X.iloc[fitidx],y[fitidx],model__sample_weight=w[fitidx])
        prob=model.predict_proba(X.iloc[te])[:,1]; pred=(prob>=.5).astype(np.int8)
        row={'repeat':rep,'fold':fold,'seed':seed,'model':'Histogram gradient boosting','n_fit':len(fitidx),'n_test':len(te),
             'n_test_districts':pd.Series(groups[te]).nunique(),**metrics(y[te],pred,prob,w[te],.5)}
        all_rows.append(row)
        lp=(prob>=locked).astype(np.int8)
        threshold_rows.append({'repeat':rep,'fold':fold,'seed':seed,'locked_threshold':locked,**metrics(y[te],lp,prob,w[te],locked)})
        all_oof.append(pd.DataFrame({'row_id':te,'y':y[te],'prob':prob,'pred':pred,'weight':w[te],
                                     'district':groups[te],'region':regions[te],'repeat':rep,'fold':fold}))
        if rep==1:
            # Baselines on identical outer folds.
            baseline_prob=np.repeat(np.average(y[fitidx],weights=w[fitidx]),len(te)); baseline=(baseline_prob>=.5).astype(np.int8)
            all_rows.append({'repeat':rep,'fold':fold,'seed':seed,'model':'Weighted prevalence baseline','n_fit':len(fitidx),'n_test':len(te),
                             'n_test_districts':pd.Series(groups[te]).nunique(),**metrics(y[te],baseline,baseline_prob,w[te],.5)})
            logit=Pipeline([('prep',clone(onehot_prep)),('model',SGDClassifier(loss='log_loss',alpha=1e-5,max_iter=1000,tol=1e-4,
                            average=True,early_stopping=True,validation_fraction=.1,n_iter_no_change=10,random_state=seed+fold))])
            logit.fit(X.iloc[fitidx],y[fitidx],model__sample_weight=w[fitidx]); lprob=logit.predict_proba(X.iloc[te])[:,1]; lpred=(lprob>=.5).astype(np.int8)
            all_rows.append({'repeat':rep,'fold':fold,'seed':seed,'model':'Regularized logistic SGD','n_fit':len(fitidx),'n_test':len(te),
                             'n_test_districts':pd.Series(groups[te]).nunique(),**metrics(y[te],lpred,lprob,w[te],.5)})
            rf=Pipeline([('prep',clone(ordinal_prep)),('model',RandomForestClassifier(n_estimators=250,max_features='sqrt',
                         min_samples_leaf=30,n_jobs=4,random_state=seed+fold,class_weight=None))])
            rf.fit(X.iloc[fitidx],y[fitidx],model__sample_weight=w[fitidx]); rprob=rf.predict_proba(X.iloc[te])[:,1]; rpred=(rprob>=.5).astype(np.int8)
            all_rows.append({'repeat':rep,'fold':fold,'seed':seed,'model':'Random forest','n_fit':len(fitidx),'n_test':len(te),
                             'n_test_districts':pd.Series(groups[te]).nunique(),**metrics(y[te],rpred,rprob,w[te],.5)})
        print(json.dumps(row),flush=True)

pd.DataFrame(all_rows).to_csv(OUT+'/repeated_model_metrics.csv',index=False)
pd.DataFrame(threshold_rows).to_csv(OUT+'/locked_threshold_metrics.csv',index=False)
oof_all=pd.concat(all_oof,ignore_index=True).sort_values(['repeat','row_id'])
oof=oof_all[oof_all['repeat'].eq(1)].sort_values('row_id').copy()
oof.to_csv(OUT+'/primary_oof_predictions.csv',index=False)

# Calibration intercept, slope and ECE for each complete OOF repeat.
eps=1e-6; bins=np.linspace(0,1,11); calibration_rows=[]; bin_rows=[]
for rep in sorted(oof_all['repeat'].unique()):
    oo=oof_all[oof_all['repeat'].eq(rep)].copy(); pv=np.clip(oo.prob.to_numpy(float),eps,1-eps)
    logits=np.log(pv/(1-pv)).reshape(-1,1)
    cal=LogisticRegression(C=1e6,max_iter=1000).fit(logits,oo.y,sample_weight=oo.weight)
    bid=np.clip(np.digitize(oo.prob,bins)-1,0,9); ece=0
    for b in range(10):
        m=bid==b
        if not m.any(): continue
        ww=oo.weight[m]; obs=np.average(oo.y[m],weights=ww); pp=np.average(oo.prob[m],weights=ww); share=ww.sum()/oo.weight.sum()
        ece+=share*abs(obs-pp); bin_rows.append({'repeat':rep,'bin':b+1,'n':int(m.sum()),'mean_predicted':pp,'observed':obs,'weighted_share':share})
    calibration_rows.append({'repeat':rep,'calibration_intercept':float(cal.intercept_[0]),
                             'calibration_slope':float(cal.coef_[0,0]),'ece_10bin':ece,
                             'brier':brier_score_loss(oo.y,oo.prob,sample_weight=oo.weight)})
pd.DataFrame(bin_rows).to_csv(OUT+'/calibration_bins_by_repeat.csv',index=False)
pd.DataFrame(calibration_rows).to_csv(OUT+'/calibration_by_repeat.csv',index=False)
pd.DataFrame(bin_rows).groupby('bin',as_index=False).agg(n=('n','sum'),mean_predicted=('mean_predicted','mean'),
    observed=('observed','mean'),weighted_share=('weighted_share','mean')).to_csv(OUT+'/calibration_bins.csv',index=False)
caldf=pd.DataFrame(calibration_rows)
pd.DataFrame([{
    'calibration_intercept_mean':caldf.calibration_intercept.mean(),'calibration_intercept_sd':caldf.calibration_intercept.std(ddof=1),
    'calibration_slope_mean':caldf.calibration_slope.mean(),'calibration_slope_sd':caldf.calibration_slope.std(ddof=1),
    'ece_10bin_mean':caldf.ece_10bin.mean(),'ece_10bin_sd':caldf.ece_10bin.std(ddof=1),
    'brier_mean':caldf.brier.mean(),'brier_sd':caldf.brier.std(ddof=1)}]).to_csv(OUT+'/calibration_summary.csv',index=False)

# Threshold trade-off curve on OOF predictions (exploratory; locked thresholds are evaluated separately above).
trs=[]
for t in np.arange(.10,.901,.02):
    p=(oof.prob>=t).astype(np.int8); m=metrics(oof.y,p,oof.prob,oof.weight,float(t)); trs.append(m)
pd.DataFrame(trs).to_csv(OUT+'/threshold_curve.csv',index=False)

# District-cluster bootstrap confidence intervals using first-repeat OOF predictions.
rng=np.random.default_rng(7717); districts=np.array(sorted(oof.district.unique())); idxmap={g:np.flatnonzero(oof.district.to_numpy()==g) for g in districts}
boots=[]
for b in range(300):
    chosen=rng.choice(districts,size=len(districts),replace=True)
    idx=np.concatenate([idxmap[g] for g in chosen])
    yy=oof.y.to_numpy()[idx]; pp=oof.prob.to_numpy()[idx]; ww=oof.weight.to_numpy()[idx]; pr=(pp>=.5).astype(np.int8)
    boots.append(metrics(yy,pr,pp,ww,.5))
boot=pd.DataFrame(boots); boot.to_csv(OUT+'/district_bootstrap_metrics.csv',index=False)
ci=pd.DataFrame({'estimate':pd.Series(metrics(oof.y,oof.pred,oof.prob,oof.weight,.5)),
                 'ci_low':boot.quantile(.025),'ci_high':boot.quantile(.975)})
ci.to_csv(OUT+'/district_bootstrap_ci.csv')

# Leave-one-region-out transfer with 100,000 training records per region for computational comparability.
regrows=[]
for i,reg in enumerate(sorted(pd.unique(regions))):
    te=np.flatnonzero(regions==reg); tr=np.flatnonzero(regions!=reg)
    rng=np.random.default_rng(9000+i); fitidx=np.sort(rng.choice(tr,size=min(100000,len(tr)),replace=False))
    model=make_hgb(70); model.fit(X.iloc[fitidx],y[fitidx],model__sample_weight=w[fitidx]); prob=model.predict_proba(X.iloc[te])[:,1]; pred=(prob>=.5).astype(np.int8)
    regrows.append({'held_out_region':reg,'n_test':len(te),'prevalence':np.average(y[te],weights=w[te]),**metrics(y[te],pred,prob,w[te],.5)})
    print('region',reg,regrows[-1]['roc_auc'],flush=True)
pd.DataFrame(regrows).to_csv(OUT+'/leave_one_region_out.csv',index=False)

# Learning curves and permutation importance on the first fold of each repeat.
lc=[]; pi_rows=[]
for rep,seed in enumerate(seeds,1):
    tr,te=next(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed).split(X,y,groups))
    nested_order=np.random.default_rng(5500+seed).permutation(tr)
    for n in [25000,50000,100000,200000,400000]:
        fitidx=np.sort(nested_order[:min(n,len(nested_order))]); model=make_hgb(80)
        ts=time.time(); model.fit(X.iloc[fitidx],y[fitidx],model__sample_weight=w[fitidx]); prob=model.predict_proba(X.iloc[te])[:,1]
        lc.append({'repeat':rep,'seed':seed,'n_train':len(fitidx),'fit_seconds':time.time()-ts,**metrics(y[te],prob>=.5,prob,w[te],.5)})
    fitidx=np.sort(nested_order[:min(200000,len(nested_order))]); evalidx=np.sort(np.random.default_rng(6500+seed).choice(te,size=min(30000,len(te)),replace=False))
    model=make_hgb(80); model.fit(X.iloc[fitidx],y[fitidx],model__sample_weight=w[fitidx])
    pi=permutation_importance(model,X.iloc[evalidx],y[evalidx],scoring='roc_auc',n_repeats=5,random_state=7500+seed,n_jobs=1,sample_weight=w[evalidx])
    for feature,mean,sd in zip(features,pi.importances_mean,pi.importances_std):
        pi_rows.append({'repeat':rep,'seed':seed,'feature':feature,'importance_mean':mean,'within_repeat_sd':sd})
pd.DataFrame(lc).to_csv(OUT+'/learning_curve_by_repeat.csv',index=False)
pd.DataFrame(lc).groupby('n_train',as_index=False).agg(fit_seconds=('fit_seconds','mean'),roc_auc=('roc_auc','mean'),
    roc_auc_sd=('roc_auc','std'),average_precision=('average_precision','mean'),average_precision_sd=('average_precision','std'),
    brier=('brier','mean'),brier_sd=('brier','std')).to_csv(OUT+'/learning_curve.csv',index=False)
pd.DataFrame(pi_rows).to_csv(OUT+'/permutation_importance_by_repeat.csv',index=False)
pd.DataFrame(pi_rows).groupby('feature',as_index=False).agg(importance_mean=('importance_mean','mean'),
    importance_sd=('importance_mean','std'),within_repeat_sd=('within_repeat_sd','mean')).sort_values('importance_mean',ascending=False).to_csv(OUT+'/permutation_importance.csv',index=False)

# Publication figures.
mr=pd.DataFrame(all_rows)
summary=mr.groupby('model').agg({c:['mean','std'] for c in ['roc_auc','average_precision','f1','balanced_accuracy','brier','mcc']})
summary.to_csv(OUT+'/model_summary.csv')

cb=pd.read_csv(OUT+'/calibration_bins.csv'); fig,ax=plt.subplots(figsize=(5.8,5.0)); ax.plot([0,1],[0,1],'--',color='gray',lw=1)
ax.plot(cb.mean_predicted,cb.observed,'o-',color='#25597A',lw=2); ax.set(xlabel='Mean predicted probability',ylabel='Observed collection proportion',xlim=(0,1),ylim=(0,1));ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT+'/calibration_plot.png',dpi=600);plt.close(fig)

rr=pd.read_csv(OUT+'/leave_one_region_out.csv').sort_values('roc_auc');fig,ax=plt.subplots(figsize=(7.2,5.3));ax.barh(rr.held_out_region,rr.roc_auc,color='#527D8D');ax.axvline(.5,color='gray',ls='--',lw=1);ax.set(xlabel='ROC AUC in held-out region',xlim=(.45,1));ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(OUT+'/region_holdout_auc.png',dpi=600);plt.close(fig)

lcdf=pd.read_csv(OUT+'/learning_curve.csv');fig,ax=plt.subplots(figsize=(6.2,4.5));ax.errorbar(lcdf.n_train,lcdf.roc_auc,yerr=lcdf.roc_auc_sd,fmt='o-',label='ROC AUC');ax.errorbar(lcdf.n_train,lcdf.average_precision,yerr=lcdf.average_precision_sd,fmt='s-',label='Average precision');ax.set(xlabel='Training households',ylabel='Held-out performance',ylim=(.5,.95));ax.grid(alpha=.2);ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT+'/learning_curve.png',dpi=600);plt.close(fig)

pimp=pd.read_csv(OUT+'/permutation_importance.csv').head(15).sort_values('importance_mean');fig,ax=plt.subplots(figsize=(7.0,5.2));ax.barh(pimp.feature,pimp.importance_mean,xerr=pimp.importance_sd,color='#6C8F71',ecolor='#333333',capsize=2);ax.set(xlabel='Decrease in ROC AUC after permutation');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(OUT+'/permutation_importance.png',dpi=600);plt.close(fig)
print('complete elapsed',round(time.time()-t0,1),flush=True)
