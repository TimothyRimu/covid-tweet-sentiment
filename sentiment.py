"""COVID-19 tweet sentiment: 5-class and 3-class. Tuning by 3-fold CV on train only; test set scored once."""
import pandas as pd, numpy as np, re, json, sys
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
D = sys.argv[1] if len(sys.argv) > 1 else 'data'   # folder holding Corona_NLP_train.csv and Corona_NLP_test.csv
tr = pd.read_csv(D+'/Corona_NLP_train.csv', encoding='latin1'); te = pd.read_csv(D+'/Corona_NLP_test.csv', encoding='latin1')
def clean(s):
    s = re.sub(r'https?://\S+', ' url ', s); s = re.sub(r'@\w+', ' @user ', s)
    return re.sub(r'\s+', ' ', s).strip().lower()
Xtr, Xte = tr.OriginalTweet.map(clean), te.OriginalTweet.map(clean)
three = lambda s: s.str.replace('Extremely ', '', regex=False)
out = {}
for name, ytr, yte in (('5-class', tr.Sentiment, te.Sentiment), ('3-class', three(tr.Sentiment), three(te.Sentiment))):
    res = {}
    d = DummyClassifier(strategy='most_frequent').fit(Xtr, ytr); p = d.predict(Xte)
    res['majority baseline'] = (accuracy_score(yte, p), f1_score(yte, p, average='macro'))
    feats = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=60000, sublinear_tf=True)   # word uni+bigrams, fitted inside each CV fold
    for mname, est, grid in (('LogReg', LogisticRegression(max_iter=300, solver='lbfgs'), {'logisticregression__C':[3]}),
                             ('LinearSVC', LinearSVC(), {'linearsvc__C':[0.1,0.3]})):
        pipe = make_pipeline(feats, est)
        gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring='f1_macro', n_jobs=1).fit(Xtr, ytr)
        p = gs.predict(Xte)
        res[mname] = (accuracy_score(yte, p), f1_score(yte, p, average='macro'))
        res[mname+'_cv_f1'] = float(gs.best_score_); res[mname+'_params'] = gs.best_params_
        if mname == 'LogReg':
            rep = classification_report(yte, p, output_dict=True)
            res['LogReg_report'] = {k: {m: round(v[m],3) for m in ('precision','recall','f1-score')} for k, v in rep.items() if k in set(yte)}
            res['confusions'] = pd.crosstab(yte, p).to_dict()
    out[name] = {k: (list(map(float, v)) if isinstance(v, tuple) else v) for k, v in res.items()}
json.dump(out, open('results.json','w'), indent=1, default=str)
for n, r in out.items():
    print(n)
    for k, v in r.items():
        if isinstance(v, list): print(f'  {k:20s} acc={v[0]:.3f} macroF1={v[1]:.3f}')
    print('  LogReg per-class F1:', {k: r['LogReg_report'][k]['f1-score'] for k in r['LogReg_report']})
    print('  CV macro-F1 LogReg/SVC:', round(r['LogReg_cv_f1'],3), round(r['LinearSVC_cv_f1'],3), r['LogReg_params'], r['LinearSVC_params'])
