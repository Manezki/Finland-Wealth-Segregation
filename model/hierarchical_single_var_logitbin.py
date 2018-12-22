import pystan
import numpy as np
import pickle
import pandas as pd


with open('paavodata_cleaned_df.pkl', 'rb') as f:
    paavo_df = pickle.load(f)

n_affluent_households = paavo_df['n_households_highest_income_2015']
n_households_total = paavo_df['n_households_2015']
# Nan cleaning
not_isnan_ix = np.logical_not(np.logical_or(np.isnan(n_households_total), np.isnan(n_affluent_households)))
postal_region_ix = paavo_df['postal_region_ix'][not_isnan_ix]
n_affluent_households = n_affluent_households[not_isnan_ix]
n_households_total = n_households_total[not_isnan_ix]
n_postal_codes = np.sum(not_isnan_ix)
n_postal_regions = paavo_df['postal_region'][not_isnan_ix].nunique()
data = dict(
        n_postal_codes=n_postal_codes,
        n_postal_regions=n_postal_regions,
        postal_region_ix=postal_region_ix.values,
        n_affluent=n_affluent_households.values.astype(int),
        n_households=n_households_total.values.astype(int))
model = pystan.StanModel(file=op.join(op.dirname(__file__), "hierarchical_single_vat_logitbin.stan"))
fit = model.sampling(data=data, iter=5000, chains=2)
print(fit)
extracts = fit.extract(permuted=True)
posterior_samples = [extracts[param] for param in ['mu_national', 'mu_regional', 'sigma_national', 'sigma_regional', 'eta', 'log_lik']]

with open('affluence_hierarchical_logit_bin_fit.txt', "w+") as f:
    f.write(str(fit))

with open('affluence_hierarchical_logit_bin.pkl', 'wb') as f:
    pickle.dump((postal_region_ix, posterior_samples), f)
