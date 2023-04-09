import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss 
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.decomposition import PCA
from sklearn import inspection
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import shap


def feature_importance_spearmanr(data):
    # Calculate Spearman's rank correlation coefficient for each feature
    corrs = []
    for col in data.columns:
        corr, p = spearmanr(data[col], data["target"])
        corrs.append(corr)

    # Normalize the correlation coefficients
    corrs = np.array(corrs) / np.sum(corrs)

    corrs_df = pd.DataFrame({"column": data.columns, "importance": corrs})
    corrs_df = corrs_df[corrs_df['column']!='target']
    
    return corrs_df

def feature_importance_pearsonr(data):
    # Calculate Pearson's correlation coefficient for each feature
    corrs = []
    for col in data.columns:
        corr, p = pearsonr(data[col], data["target"])
        corrs.append(corr)

    # Normalize the correlation coefficients
    corrs = np.array(corrs) / np.sum(corrs)

    corrs_df = pd.DataFrame({"column": data.columns, "importance": corrs})
    corrs_df = corrs_df[corrs_df['column']!='target']
    
    return corrs_df

def perform_pca(data):
    X = data.drop(columns=['target'])
    pca = PCA(n_components=X.shape[1])
    X_pca = pca.fit_transform(X)
    new_features = ["Feature_"+str(x) for x in range(X.shape[1])]
    imp_df = pd.DataFrame({"column": new_features, 
                           "importance": np.round(pca.explained_variance_ratio_,2)})
    return imp_df
    

def feature_importance_model(df, model='RandomForestRegressor'):
    X = df.drop(columns=['target'])
    y = df['target']
    if model == "RandomForestRegressor":
        model = RandomForestRegressor()
        model.fit(X,y)
        importance = model.feature_importances_
    elif model == "RandomForestClassifier":
        model = RandomForestClassifier()
        model.fit(X,y)
        importance = model.feature_importances_
    
    imp_df = pd.DataFrame({"column": X.columns, "importance": importance})    
    return imp_df


def plot_importance(df, title="Feature Importance", xlabel="Importance"):
    fig, ax = plt.subplots(figsize=(8,8))
    df.sort_values(by = 'importance', ascending = True, inplace=True)
    ax.barh(df.column, df.importance)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.show()
    
def permutation_importance(df, model="LinearRegression", metric="MSE", random_state=42):
    X = df.drop(columns=['target'])
    y = df['target']
    if model == "LinearRegression":
        model = LinearRegression()
        model.fit(X,y)
    elif model == "LogisticRegression":
        model = LogisticRegression()
        model.fit(X,y)
    elif model == "RandomForestRegressor":
        model = RandomForestRegressor()
        model.fit(X,y)
    elif model == "RandomForestClassifier":
        model = RandomForestClassifier()
        model.fit(X,y)
    
    if metric == "MSE":
        metric = mean_squared_error
    if metric == "logloss":
        metric = log_loss

    np.random.seed(random_state)
    feature_importance = {}
    baseline_score = metric(y, model.predict(X))
    for feature in X.columns:
        permuted_score=0
        X_permuted = X.copy()
        for i in range(5):
            X_permuted[feature] = np.random.permutation(X_permuted[feature])
            permuted_score += metric(y, model.predict(X_permuted))
        feature_importance[feature] = baseline_score - (permuted_score)/5
    
    imp_df = pd.DataFrame({"column": feature_importance.keys(), 
                           "importance": feature_importance.values()})
    return imp_df

def calculate_mae(df, features):
    X = df.drop(columns=['target'])
    y = df['target']
    Xf= X[features]
    X_train, X_test, y_train, y_test = train_test_split(Xf, y, test_size=0.2, random_state=47)
    model=RandomForestRegressor()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_pred, y_test)

def feature_importance_all(df,K=None):
    if K==None:
        K=len(df.columns)-1
    mae_data={'spearmanr':[], 'pearsonr':[], 'permutation':[], 'pca':[], 'gini_importance':[]}
    for k in range(1,K):
        for f in mae_data.keys():
            if f == 'spearmanr':
                imp_df = feature_importance_spearmanr(df) 
            elif f == 'pearsonr':
                imp_df = feature_importance_pearsonr(df) 
            elif f == 'permutation':
                imp_df = permutation_importance(df) 
            elif f == 'spearmanr':
                imp_df = feature_importance_spearmanr(df) 
            elif f == 'spearmanr':
                imp_df = feature_importance_spearmanr(df)
            sorted_df = imp_df.sort_values(by='importance',key=lambda x: np.abs(x), ascending=False)
            features_selected = list(sorted_df['column'][:k])
            mae_data[f].append(calculate_mae(df, features_selected))
    return mae_data

def shap_importance(df):
    X = df.drop(columns=['target'])
    y = df['target']
    model = RandomForestRegressor()
    model.fit(X, y)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.plots.bar(shap_values.mean(0), show=False)

def plot_all(dict_data):
    fig, ax = plt.subplots(figsize=(8,8))
    m=['o','v','+','x', 's']
    i=0
    for d in dict_data:
        ax.plot(range(1,len(dict_data[d])+1), dict_data[d], marker = m[i], label=d)
        ax.set_xlabel("K")
        ax.set_ylabel("Mean Absolute Error")
        ax.set_title("Comparison of feature selection techniques using MAE", pad=15)
        plt.legend()
        i+=1