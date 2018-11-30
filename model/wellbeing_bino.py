import pystan
import numpy as np
import pickle
import pandas as pd
from os import path as op

with open('paavodata_cleaned_df.pkl', 'rb') as f:
    paavo_df = pickle.load(f)

# Areas with no households?
paavo_df = paavo_df.loc[~paavo_df["n_households_2015"].isna()]
paavo_df["n_households_highest_income_2015"] = paavo_df["n_households_highest_income_2015"].fillna(0)

n_postal_regions = paavo_df['postal_region'].nunique()
n_postal_codes = paavo_df.shape[0]
postal_region_ix = paavo_df['postal_region_ix']
n_affluent_households = paavo_df['n_households_highest_income_2015'].astype(int)
n_households = paavo_df['n_households_2015'].astype(int)

print(f'n_postal_codes={n_postal_codes}, n_postal_regions={n_postal_regions}, postal_region_ix={postal_region_ix}')
print(f'n_affluent_households={n_affluent_households}, n_households={n_households}')
data = dict(
    n_postal_codes=n_postal_codes,
    n_postal_regions=n_postal_regions,
    n_affluent_households=n_affluent_households,
    n_households=n_households,
    postal_region_ix=postal_region_ix)

model = pystan.StanModel(file=op.join(op.dirname(__file__),"single_param_bino.stan"))
fit = model.sampling(data=data, iter=1000, chains=4)
extracts = fit.extract(permuted=True)

posterior_samples = [extracts[param] for param in ['p_regional', 'log_lik']]

with open('wellbeing1_hierarchical_binomial_fit.txt', "x") as f:
    f.write(str(fit))

with open('wellbeing1_hierarchical_binomial.pkl', 'wb') as f:
    pickle.dump(posterior_samples, f)
