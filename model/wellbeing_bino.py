import pystan
import numpy as np
import pickle
import pandas as pd
from os import path as op

with open('paavodata_cleaned_df.pkl', 'rb') as f:
    paavo_df = pickle.load(f)

# Areas with no households?
paavo_df = paavo_df.loc[~paavo_df["n_households_2015"].isna()]
#paavo_df["n_households_highest_income_2015"] = paavo_df["n_households_highest_income_2015"].fillna(0)

n_postal_regions = paavo_df['postal_region'].nunique()
n_postal_codes = paavo_df.shape[0]
postal_region_ix = paavo_df['postal_region_ix']
n_affluent_households = paavo_df['n_households_highest_income_2015']
n_households_total = paavo_df['n_households_2015']

# Nan cleaning
not_isnan_ix = np.logical_not(np.logical_or(np.isnan(n_households_total), np.isnan(n_affluent_households)))
postal_region_ix = paavo_df['postal_region_ix'][not_isnan_ix]
n_affluent_households = n_affluent_households[not_isnan_ix].astype(int)
n_households_total = n_households_total[not_isnan_ix].astype(int)
n_postal_codes = np.sum(not_isnan_ix)
n_postal_regions = paavo_df['postal_region'][not_isnan_ix].nunique()


print(f'n_postal_codes={n_postal_codes}, n_postal_regions={n_postal_regions}, postal_region_ix={len(postal_region_ix)}')
print(f'n_affluent_households={len(n_affluent_households)}, n_households={len(n_households_total)}')
data = dict(
    n_postal_codes=n_postal_codes,
    n_postal_regions=n_postal_regions,
    n_affluent_households=n_affluent_households,
    n_households=n_households_total,
    postal_region_ix=postal_region_ix)

model = pystan.StanModel(file=op.join(op.dirname(__file__),"single_param_bino.stan"))
fit = model.sampling(data=data, iter=5000, chains=2)
extracts = fit.extract(permuted=True)

posterior_samples = [extracts[param] for param in ['p_regional', 'log_lik', 'national_sigma', 'national_mu']]

with open('wellbeing_hierarchical_binomial_fit.txt', "w+") as f:
    f.write(str(fit))

with open('wellbeing_hierarchical_binomial.pkl', 'wb') as f:
    pickle.dump(posterior_samples, f)
