import numpy as np
import pandas as pd
from itertools import combinations

class Preprocessor:
    def __init__(self, target_col='Churn'):
        self.target_col = target_col
        self.cats = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
        self.nums = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.top_cats_for_ngram = [
            'Contract', 'InternetService', 'PaymentMethod',
            'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
        ]
        self.new_nums = []
        self.num_as_cat = [f'CAT_{col}' for col in self.nums]
        self.ngram_cols = []
        for c1, c2 in combinations(self.top_cats_for_ngram, 2):
            self.ngram_cols.append(f"BG_{c1}_{c2}")
        top4 = self.top_cats_for_ngram[:4]
        for c1, c2, c3 in combinations(top4, 3):
            self.ngram_cols.append(f"TG_{c1}_{c2}_{c3}")
        self.digit_cols = [
            'tenure_first_digit', 'tenure_last_digit', 'tenure_second_digit',
            'tenure_mod10', 'tenure_mod12', 'tenure_num_digits',
            'tenure_is_multiple_10', 'tenure_rounded_10', 'tenure_dev_from_round10',
            'mc_first_digit', 'mc_last_digit', 'mc_second_digit',
            'mc_mod10', 'mc_mod100', 'mc_num_digits', 
            'mc_is_multiple_10', 'mc_is_multiple_50',
            'mc_rounded_10', 'mc_fractional', 'mc_dev_from_round10',
            'tc_first_digit', 'tc_last_digit', 'tc_second_digit',
            'tc_mod10', 'tc_mod100', 'tc_num_digits',
            'tc_is_multiple_10', 'tc_is_multiple_100',
            'tc_rounded_100', 'tc_fractional', 'tc_dev_from_round100',
            'tenure_years', 'tenure_months_in_year', 'mc_per_digit', 'tc_per_digit'
        ]
        self.dist_features = [
            'pctrank_nonchurner_TC', 'zscore_churn_gap_TC', 'pctrank_churn_gap_TC',
            'resid_IS_MC', 'cond_pctrank_IS_TC', 'zscore_nonchurner_TC',
            'pctrank_orig_TC', 'pctrank_churner_TC', 'cond_pctrank_C_TC'
        ]
        self.q_features = []
        for q_label in ['q25', 'q50', 'q75']:
            self.q_features.extend([f'dist_To_ch_{q_label}', f'dist_To_nc_{q_label}', f'qdist_gap_To_{q_label}'])

        self.is_fitted = False
        self.freq_maps = {}
        self.churner_tc = None
        self.nonchurner_tc = None
        self.tc = None
        self.is_mc_mean = None
        self.cond_pctrank_refs = {}
        self.quantile_refs = {}
        self.total_charges_median = None

    @staticmethod
    def pctrank_against(values, reference):
        ref_sorted = np.sort(reference)
        return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')

    @staticmethod
    def zscore_against(values, reference):
        mu, sigma = np.mean(reference), np.std(reference)
        return (np.zeros(len(values), dtype='float32') if sigma == 0 
                else ((values - mu) / sigma).astype('float32'))

    def _fit_stats(self, train):
        for col in self.nums:
            self.freq_maps[col] = train[col].value_counts(normalize=True).astype('float32')
        self.churner_tc = train.loc[train[self.target_col] == 1, 'TotalCharges'].values.astype('float32')
        self.nonchurner_tc = train.loc[train[self.target_col] == 0, 'TotalCharges'].values.astype('float32')
        self.tc = train['TotalCharges'].values.astype('float32')
        self.is_mc_mean = train.groupby('InternetService')['MonthlyCharges'].mean().astype('float32')
        self.total_charges_median = train['TotalCharges'].median()
        self.cond_pctrank_refs = {}
        for cat_col in ['InternetService', 'Contract']:
            self.cond_pctrank_refs[cat_col] = {}
            for cat_val in train[cat_col].unique():
                ref = train.loc[train[cat_col] == cat_val, 'TotalCharges'].values.astype('float32')
                self.cond_pctrank_refs[cat_col][cat_val] = ref
        self.quantile_refs = {}
        for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
            self.quantile_refs[q_label] = {
                'ch': np.quantile(self.churner_tc, q_val),
                'nc': np.quantile(self.nonchurner_tc, q_val)
            }
        self.is_fitted = True

    def _create_frequency_encoding(self, df):
        for col in self.nums:
            df[f'FREQ_{col}'] = df[col].map(self.freq_maps[col]).fillna(0).astype('float32')

    def _create_arithmetic_interactions(self, df):
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
        # New features from Colab
        df['cost_per_service'] = (df['MonthlyCharges'] / (df['service_count'] + 1)).astype('float32')
        df['total_per_service'] = (df['TotalCharges'] / (df['service_count'] + 1)).astype('float32')

    def _create_service_counts(self, df):
        service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['service_count'] = (df[service_cols] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')

    def _apply_distribution_features(self, df):
        tc = df['TotalCharges'].values
        df['pctrank_nonchurner_TC'] = self.pctrank_against(tc, self.nonchurner_tc)
        df['pctrank_churner_TC'] = self.pctrank_against(tc, self.churner_tc)
        df['pctrank_orig_TC'] = self.pctrank_against(tc, self.tc)
        df['zscore_churn_gap_TC'] = (np.abs(self.zscore_against(tc, self.churner_tc)) - 
                                     np.abs(self.zscore_against(tc, self.nonchurner_tc))).astype('float32')
        df['zscore_nonchurner_TC'] = self.zscore_against(tc, self.nonchurner_tc)
        df['pctrank_churn_gap_TC'] = (self.pctrank_against(tc, self.churner_tc) - 
                                      self.pctrank_against(tc, self.nonchurner_tc)).astype('float32')
        df['resid_IS_MC'] = (df['MonthlyCharges'] - df['InternetService'].map(self.is_mc_mean).fillna(0)).astype('float32')
        for cat_col, name in [('InternetService', 'cond_pctrank_IS_TC'), ('Contract', 'cond_pctrank_C_TC')]:
            vals = np.zeros(len(df), dtype='float32')
            for cat_val, ref in self.cond_pctrank_refs[cat_col].items():
                mask = df[cat_col] == cat_val
                if mask.sum() > 0 and len(ref) > 0:
                    vals[mask] = self.pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
            df[name] = vals

    def _apply_quantile_distance_features(self, df):
        for q_label, q_dict in self.quantile_refs.items():
            ch_q = q_dict['ch']
            nc_q = q_dict['nc']
            df[f'dist_To_ch_{q_label}'] = np.abs(df['TotalCharges'] - ch_q).astype('float32')
            df[f'dist_To_nc_{q_label}'] = np.abs(df['TotalCharges'] - nc_q).astype('float32')
            df[f'qdist_gap_To_{q_label}'] = (df[f'dist_To_nc_{q_label}'] - df[f'dist_To_ch_{q_label}']).astype('float32')

    def _create_digit_features(self, df):
        t_str = df['tenure'].astype(str)
        df['tenure_first_digit'] = t_str.str[0].astype(int)
        df['tenure_last_digit'] = t_str.str[-1].astype(int)
        df['tenure_second_digit'] = t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tenure_mod10'] = df['tenure'] % 10
        df['tenure_mod12'] = df['tenure'] % 12
        df['tenure_num_digits'] = t_str.str.len()
        df['tenure_is_multiple_10'] = (df['tenure'] % 10 == 0).astype('float32')
        df['tenure_rounded_10'] = np.round(df['tenure'] / 10) * 10
        df['tenure_dev_from_round10'] = np.abs(df['tenure'] - df['tenure_rounded_10'])
        mc_str = df['MonthlyCharges'].astype(str).str.replace('.', '', regex=False)
        df['mc_first_digit'] = mc_str.str[0].astype(int)
        df['mc_last_digit'] = mc_str.str[-1].astype(int)
        df['mc_second_digit'] = mc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['mc_mod10'] = np.floor(df['MonthlyCharges']) % 10
        df['mc_mod100'] = np.floor(df['MonthlyCharges']) % 100
        df['mc_num_digits'] = np.floor(df['MonthlyCharges']).astype(int).astype(str).str.len()
        df['mc_is_multiple_10'] = (np.floor(df['MonthlyCharges']) % 10 == 0).astype('float32')
        df['mc_is_multiple_50'] = (np.floor(df['MonthlyCharges']) % 50 == 0).astype('float32')
        df['mc_rounded_10'] = np.round(df['MonthlyCharges'] / 10) * 10
        df['mc_fractional'] = df['MonthlyCharges'] - np.floor(df['MonthlyCharges'])
        df['mc_dev_from_round10'] = np.abs(df['MonthlyCharges'] - df['mc_rounded_10'])
        tc_str = df['TotalCharges'].astype(str).str.replace('.', '', regex=False)
        df['tc_first_digit'] = tc_str.str[0].astype(int)
        df['tc_last_digit'] = tc_str.str[-1].astype(int)
        df['tc_second_digit'] = tc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tc_mod10'] = np.floor(df['TotalCharges']) % 10
        df['tc_mod100'] = np.floor(df['TotalCharges']) % 100
        df['tc_num_digits'] = np.floor(df['TotalCharges']).astype(int).astype(str).str.len()
        df['tc_is_multiple_10'] = (np.floor(df['TotalCharges']) % 10 == 0).astype('float32')
        df['tc_is_multiple_100'] = (np.floor(df['TotalCharges']) % 100 == 0).astype('float32')
        df['tc_rounded_100'] = np.round(df['TotalCharges'] / 100) * 100
        df['tc_fractional'] = df['TotalCharges'] - np.floor(df['TotalCharges'])
        df['tc_dev_from_round100'] = np.abs(df['TotalCharges'] - df['tc_rounded_100'])
        df['tenure_years'] = df['tenure'] // 12
        df['tenure_months_in_year'] = df['tenure'] % 12
        df['mc_per_digit'] = df['MonthlyCharges'] / (df['mc_num_digits'] + 0.001)
        df['tc_per_digit'] = df['TotalCharges'] / (df['tc_num_digits'] + 0.001)

    def _create_num_as_cat(self, df):
        for col in self.nums:
            df[f'CAT_{col}'] = df[col].astype(str).astype('category')

    def _create_ngram_features(self, df):
        for c1, c2 in combinations(self.top_cats_for_ngram, 2):
            df[f"BG_{c1}_{c2}"] = (df[c1].astype(str) + "_" + df[c2].astype(str)).astype('category')
        top4 = self.top_cats_for_ngram[:4]
        for c1, c2, c3 in combinations(top4, 3):
            df[f"TG_{c1}_{c2}_{c3}"] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)).astype('category')

    def fit_transform(self, train, test, orig=None):
        for df in [train, test] + ([orig] if orig is not None else []):
            if df is not None and self.target_col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[self.target_col]):
                    df[self.target_col] = df[self.target_col].astype(str).str.strip().str.capitalize().map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
            if df is not None and 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        self._fit_stats(train)
        
        # NOTE: service_count must be created first because arithmetic interactions use it
        self._create_service_counts(train); self._create_service_counts(test)
        self.new_nums += ['service_count', 'has_internet', 'has_phone']
        
        self._create_frequency_encoding(train); self._create_frequency_encoding(test)
        self.new_nums += [f'FREQ_{col}' for col in self.nums]
        
        self._create_arithmetic_interactions(train); self._create_arithmetic_interactions(test)
        self.new_nums += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges', 'cost_per_service', 'total_per_service']
        
        self._apply_distribution_features(train); self._apply_distribution_features(test)
        self.new_nums += self.dist_features
        self._apply_quantile_distance_features(train); self._apply_quantile_distance_features(test)
        self.new_nums += self.q_features
        self._create_digit_features(train); self._create_digit_features(test)
        for df_ in [train, test]:
            for c in self.digit_cols: df_[c] = df_[c].astype('float32')
        self.new_nums += self.digit_cols
        self._create_num_as_cat(train); self._create_num_as_cat(test)
        self._create_ngram_features(train); self._create_ngram_features(test)
        self.all_cat_cols = self.num_as_cat + self.cats + self.ngram_cols
        all_features = self.nums + self.cats + self.new_nums + self.num_as_cat + self.ngram_cols
        return train, test, all_features

    def transform(self, df):
        if not self.is_fitted: raise ValueError("Preprocessor not fitted.")
        if self.target_col in df.columns and not pd.api.types.is_numeric_dtype(df[self.target_col]):
            df[self.target_col] = df[self.target_col].astype(str).str.strip().str.capitalize().map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(self.total_charges_median)
        
        self._create_service_counts(df)
        self._create_frequency_encoding(df)
        self._create_arithmetic_interactions(df)
        self._apply_distribution_features(df)
        self._apply_quantile_distance_features(df)
        self._create_digit_features(df)
        for c in self.digit_cols: df[c] = df[c].astype('float32')
        self._create_num_as_cat(df)
        self._create_ngram_features(df)
        return df