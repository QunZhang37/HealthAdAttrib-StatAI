import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def prepare_table(df: pd.DataFrame):
    # Features: channel counts + path length + demographics
    channels = ['Search','Social','Email','Display','Referral','Video','Direct','CallCenter']
    X = pd.DataFrame({f'cnt_{c}': df.path.str.count(c) for c in channels})
    X['unique_channels'] = df.path.apply(lambda s: len(set(s.split('>'))))
    X['length'] = df.path.apply(lambda s: len(s.split('>')))
    X['age'] = df['age']; X['chronic'] = df['chronic']
    y = df['convert']
    return X, y

def fit_logit(df: pd.DataFrame):
    X,y = prepare_table(df)
    pipe = Pipeline([('clf', LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:,1]
    return pipe, p

def channel_importance(pipe, df: pd.DataFrame):
    X,_ = prepare_table(df)
    coef = getattr(pipe.named_steps['clf'], 'coef_', None)
    if coef is None: return {}
    coef = coef[0]
    return {col: float(w) for col, w in zip(X.columns, coef)}
