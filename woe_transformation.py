import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin


def woe_calc(X, y, feature, grp_feature):
        
    X_tmp = X.copy()
    X_tmp['target_flg'] = np.array(y.tolist())
        
    group_table = pd.DataFrame()
    group_table['feature_name'] = ()
    group_table['grp'] = ()    
    group_table['cnt'] = X_tmp[grp_feature].value_counts(dropna=False)
    group_table['bad'] = X_tmp.groupby(grp_feature)['target_flg'].sum().replace({0: 0.00001}) 
    group_table['good'] = (group_table['cnt'] - X_tmp.groupby(grp_feature)['target_flg'].sum()).replace({0: 0.00001})
    group_table['min'] = X_tmp.groupby(grp_feature)[feature].min()
    group_table['max'] = X_tmp.groupby(grp_feature)[feature].max()
    group_table['pct_total'] = group_table['cnt'] / group_table['cnt'].sum()
    group_table['pct_bad'] = group_table['bad'] / group_table['bad'].sum()
    group_table['pct_good'] = group_table['good'] / group_table['good'].sum()
    group_table['target_rate'] = group_table['bad'] / group_table['cnt']
    group_table['woe'] = np.log(group_table['pct_good'] / group_table['pct_bad'])
    group_table['iv'] = (group_table['pct_good'] - group_table['pct_bad']) * group_table['woe']
    group_table['feature_name'] = feature
    group_table['grp'] = group_table.index
    group_table['grp'] = group_table['grp'].astype(np.int64)
    
    group_table = group_table.sort_values(by=['min'])
    group_table = group_table.reset_index(drop=True)
        
    group_table = group_table.round(5)
        
    ivgini_table = pd.DataFrame({"feature_name": feature,
                                     "iv": [group_table[group_table['feature_name'] == feature]['iv'].sum()],
                                     "gini": [abs(roc_auc_score(X_tmp['target_flg'], X_tmp[grp_feature])*2-1)]})    
    
    return(group_table, ivgini_table)


def decode_categories(X, y, feature):
        
        X_tmp = X.copy()
        X_tmp['target_flg'] = np.array(y.tolist())        
        
        group_table = pd.DataFrame()
        group_table['cnt'] = X_tmp[feature].value_counts(dropna=False)
        group_table['bad'] = X_tmp.groupby(feature)['target_flg'].sum().replace({0: 0.00001})
        group_table['good'] = (group_table['cnt'] - X_tmp.groupby(feature)['target_flg'].sum().replace({0: 0.00001})).replace({0: 0.00001})
        group_table['pct_total'] = group_table['cnt'] / group_table['cnt'].sum()
        group_table['pct_bad'] = group_table['bad'] / group_table['bad'].sum()
        group_table['pct_good'] = group_table['good'] / group_table['good'].sum()
        group_table['target_rate'] = group_table['bad'] / group_table['cnt']
        group_table['woe'] = np.log(group_table['pct_good']/group_table['pct_bad'])
        group_table['iv'] = (group_table['pct_good'] - group_table['pct_bad']) * group_table['woe']

        X_tmp['DEC_'+feature] = X_tmp[feature]
        
        X_tmp = X_tmp.replace({'DEC_'+feature: group_table.to_dict('dict')['woe']})        
     
        group_table['feature_name'] = feature
        group_table.reset_index(inplace=True)
        group_table.rename(columns={feature: 'value'}, inplace=True)

        group_table = group_table.round(5)
        
        return(X_tmp['DEC_'+feature], group_table[['feature_name', 'value', 'woe']])   


def find_woe(value, table):
    
    i = 0

    while value < table['min'][i] or value > table['max'][i]:
        
        i += 1

    return table['woe'][i]


class FineClassing(BaseEstimator, TransformerMixin):

    def __init__(self, n_fine_grp=10, discrete_limit=20):

        self.n_fine_grp = n_fine_grp
        self.discrete_limit = discrete_limit
        self.fine_stats = pd.DataFrame()
        self.fine_iv_gini = pd.DataFrame()

    def fit(self, X, y):

        X_tmp = X.copy()
        y_tmp = y.copy()

        for f in X_tmp.columns.tolist():

            if X_tmp[f].dtype == 'object':

                dcd = decode_categories(X_tmp, y_tmp, f)

                try:
                    self.decoded_features
                
                except AttributeError:
                    self.decoded_features = pd.DataFrame()
                
                self.decoded_features = pd.concat([self.decoded_features, dcd[1]])

                X_tmp[f] = dcd[0].copy()

            if X_tmp[f].nunique() <= self.discrete_limit:

                limits = X_tmp[f].unique().tolist()
                limits.sort()
                limits.insert(0, -np.inf)
                limits.append(np.inf)

                bins = pd.cut(X_tmp[f], limits, labels=False)

            else:
                
                bins = pd.qcut(X_tmp[f], self.n_fine_grp,
                                      duplicates='drop', labels=False)
                
            X_tmp['GRP_'+f] = bins
            X_tmp['GRP_'+f].fillna(-1, inplace=True)
            X_tmp['GRP_'+f] = X_tmp['GRP_'+f].astype(int)

            stats_t, iv_gini_t = woe_calc(X_tmp, y_tmp, f, 'GRP_'+f)
            self.fine_stats = pd.concat([self.fine_stats, stats_t]).copy()
            self.fine_iv_gini = pd.concat([self.fine_iv_gini, iv_gini_t]).copy()
            self.fine_iv_gini = self.fine_iv_gini.reset_index(drop=True)            

        return self

    def transform(self, X):

        X_tmp = X.copy()

        for f in X_tmp.columns.tolist():

            if X_tmp[f].dtype == 'object':

                X_tmp = X_tmp.replace({f: self.decoded_features[self.decoded_features['feature_name'] == f]\
                                       .set_index('value').to_dict('dict')['woe']})

                X_tmp[f] = np.where(X_tmp[f].isin(list(self.decoded_features.loc[self.decoded_features['feature_name'] == f]['woe'])),
                                    X_tmp[f], np.nan)

            limits = self.fine_stats.loc[self.fine_stats['feature_name'] == f]['max'].tolist()
            limits.insert(0, -np.inf)
            limits = limits[0:-1]
            limits.append(np.inf)

            X_tmp[f] = pd.cut(np.array(X_tmp[f]), limits, labels=range(0, len(limits)-1))
            
            if X_tmp[f].isna().sum() != 0:

                X_tmp[f] = X_tmp[f].cat.add_categories(-1)
                X_tmp[f].fillna(-1, inplace=True)

            X_tmp[f] = X_tmp[f].astype(int)

        return X_tmp
    

def monotonic(x):
            
    dx = np.diff(x)
    
    return np.all(dx <= 0) or np.all(dx >= 0)
    

class CoarseClassing(BaseEstimator, TransformerMixin):

    def __init__(self, n_coarse_grp=5):

        self.n_coarse_grp = n_coarse_grp
        self.coarse_stats = pd.DataFrame()
        self.coarse_iv_gini = pd.DataFrame()

    def fit(self, X, y):

        X_tmp = X.copy()
        y_tmp = y.copy()   

        df_tmp = X_tmp.copy()
        df_tmp['target_flg'] = np.array(y_tmp.tolist()) 
        
        for f in X_tmp.columns.tolist():

            monotonous = False
            bins = self.n_coarse_grp

            while monotonous != True:
                
                dtree = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=0,
                                               max_leaf_nodes=bins, min_samples_leaf=0.1)\
                                               .fit(np.array(df_tmp.loc[df_tmp[f] != -1][f]).reshape(-1, 1),
                                                    df_tmp.loc[df_tmp[f] != -1]['target_flg'])
                
                thresholds = np.array([x for x in dtree.tree_.threshold if x != _tree.TREE_UNDEFINED]).tolist()
                thresholds.sort()
                thresholds.insert(0, -0.5)
                thresholds.append(np.inf)

                X_tmp['GRP_'+f] = pd.cut(np.array(X_tmp[f]), bins=thresholds, labels=range(0, len(thresholds)-1))

                if X_tmp['GRP_'+f].isna().sum() != 0:

                    X_tmp['GRP_'+f] = X_tmp['GRP_'+f].cat.add_categories(-1)
                    X_tmp['GRP_'+f].fillna(-1, inplace=True)

                stats, iv_gini = woe_calc(X_tmp, y_tmp, f, 'GRP_'+f)

                if monotonic(stats.loc[(stats['feature_name'] == f) & (stats['grp'] != -1)]['woe']):

                    monotonous = True
                    self.coarse_stats = pd.concat([self.coarse_stats, stats]).copy()
                    self.coarse_iv_gini = pd.concat([self.coarse_iv_gini, iv_gini]).reset_index(drop=True).copy()

                bins += -1

        return self
    
    def transform(self, X):

        X_tmp = X.copy()

        for f in X_tmp.columns.tolist():

            coarse_table = self.coarse_stats.loc[self.coarse_stats['feature_name'] == f][['min', 'max', 'woe']].copy()

            X_tmp[f] = X_tmp[f].apply(lambda x: find_woe(x, coarse_table))

        return X_tmp


class FilterByInformationValue(BaseEstimator, TransformerMixin):

    def __init__(self, iv_threshold=0.05):

        self.iv_threshold = iv_threshold

    def fit(self, X, y):

        X_tmp = X.copy()
        y_tmp = y.copy()

        self.feature_list = []

        for f in X_tmp.columns.tolist():

            if woe_calc(X_tmp, y_tmp, f, f)[1]['iv'][0] >= self.iv_threshold:

                self.feature_list.append(f)

        return self
    
    def transform(self, X):

        X_tmp = X.copy()

        return X_tmp[self.feature_list]
    

class FilterByCorrelation(BaseEstimator, TransformerMixin):

    def __init__(self, sorted=False, corr_method='spearman', corr_threshold=0.5):

        self.sorted = sorted
        self.corr_method = corr_method
        self.corr_threshold = corr_threshold

    def fit(self, X, y):

        X_tmp = X.copy()
        y_tmp = y.copy()

        if sorted == False:
            iv_tmp = pd.DataFrame()
            for f in X_tmp.columns.tolist():
                iv_tmp = pd.concat([iv_tmp, woe_calc(X_tmp, y_tmp, f, f)[1]]).copy()
            sorted_features = iv_tmp.sort_values(by='iv', ascending=False)['feature_name'].tolist()
        else:
            sorted_features = X_tmp.columns.tolist()

        self.before_corr_matrix = X_tmp[sorted_features].corr(method=self.corr_method)

        lower = self.before_corr_matrix.abs().where(np.tril(np.ones(self.before_corr_matrix.shape), k=-1).astype(bool))
        corr_cols = lower.columns

        for col in corr_cols:
    
            drop_list = []
    
            if col in lower.columns:
        
                for row in range(0, lower.shape[0]):
        
                    if (lower[col].iloc[row] > self.corr_threshold) & (np.isnan(lower[col].iloc[row]) == False):
                
                        drop_list.append(lower[col].index[row])
                        corr_cols.drop(lower[col].index[row])
                
                lower = lower.drop(columns = drop_list)
                lower = lower.drop(drop_list)

        self.feature_list = lower.columns.tolist()
        self.after_corr_matrix = X_tmp[self.feature_list].corr(method=self.corr_method)

        return self
    
    def transform(self, X):

        X_tmp = X.copy()

        return X_tmp[self.feature_list]


def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out = 0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit(disp=0)
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=0)
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

  


class StepwiseFeatureSelection(BaseEstimator, TransformerMixin):

    def __init__(self, threshold_in=0.05, threshold_out=0.05, verbose=False):

        self.threshold_in = threshold_in
        self.threshold_out = threshold_out
        self.verbose = verbose

    def fit(self, X, y):

        X_tmp = X.copy()
        y_tmp = y.copy()

        self.feature_list = stepwise_selection(X_tmp, y_tmp, [],
                                               self.threshold_in, self.threshold_out, self.verbose)
        
        return self
    
    def transform(self, X):

        X_tmp = X.copy()

        return X_tmp[self.feature_list]




class StepwiseFeatureSelectionCV(BaseEstimator, TransformerMixin):

    def __init__(self, cv, cv_min, threshold_in=0.05, threshold_out=0.05, verbose=False):

        self.cv = cv
        self.cv_min = cv_min
        self.threshold_in = threshold_in
        self.threshold_out = threshold_out
        self.verbose = verbose

    def fit(self, X, y):

        X_tmp = X.copy()
        y_tmp = y.copy()

        self.cv_features = {}
        for f in X_tmp.columns:
            self.cv_features[f] = 0

        for i, (train_index, test_index) in enumerate(self.cv.split(X_tmp)):

            if self.verbose:
                print(f'{i+1}. fold:')

            train_index = pd.Index(train_index)
            test_index = pd.Index(test_index)

            X_train = X_tmp.loc[X_tmp.index.isin(train_index)]
            X_test = X_tmp.loc[X_tmp.index.isin(test_index)]

            y_train = y_tmp.loc[y_tmp.index.isin(train_index)]
            y_test = y_tmp.loc[y_tmp.index.isin(test_index)]

            feature_list = stepwise_selection(X_train, y_train, [],
                                               self.threshold_in, self.threshold_out, self.verbose)
            
            for f in feature_list:
                self.cv_features[f] += 1
        
        self.feature_list = []
        for key in self.cv_features:
            if self.cv_features[key] >= self.cv_min:
                self.feature_list.append(key)
        
        return self
    
    def transform(self, X):

        X_tmp = X.copy()

        return X_tmp[self.feature_list]
