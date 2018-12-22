import pystan
import numpy as np
import pickle
import pandas as pd


with open('paavodata_cleaned_df.pkl', 'rb') as f:
    paavo_df = pickle.load(f)

n_postal_regions = paavo_df['postal_region'].nunique()
n_postal_codes = paavo_df.shape[0]
postal_region_ix = paavo_df['postal_region_ix']
pct_affluent_households = paavo_df['n_households_highest_income_2015_pc']
print(f'n_postal_codes={n_postal_codes}, n_postal_regions={n_postal_regions}, postal_region_ix={postal_region_ix}')
print(f'pct_affluent_households={pct_affluent_households}')

n_households_total = paavo_df['n_households_2015']
n_affluent_households = paavo_df['n_households_highest_income_2015']

not_isnan_ix = np.logical_not(np.logical_or(np.isnan(n_households_total), np.isnan(n_affluent_households)))
postal_region_ix = paavo_df['postal_region_ix'][not_isnan_ix]
n_affluent_households = n_affluent_households[not_isnan_ix]
n_households_total = n_households_total[not_isnan_ix]
pct_affluent_households = n_affluent_households/n_households_total
n_postal_codes = np.sum(not_isnan_ix)
n_postal_regions = paavo_df['postal_region'][not_isnan_ix].nunique()

data = dict(n_postal_codes=n_postal_codes,
            n_postal_regions=n_postal_regions,
            pct_affluent_households = pct_affluent_households,
            postal_region_ix=postal_region_ix)

model = pystan.StanModel(file=op.join(op.dirname(__file__), "hierarchical_gaussian.stan"))
fit = model.sampling(data=data, iter=1000, chains=4)
print(fit)
extracts = fit.extract(permuted=True)
posterior_samples = [extracts[param] for param in ['mu_national', 'mu_regional', 'sigma_national', 'sigma_regional', 'log_lik']]

with open('wellbeing_hierarchical_gaussian_fit.txt', "w+") as f:
    f.write(str(fit))

with open('wellbeing_hierarchical_gaussian.pkl', 'wb') as f:
    pickle.dump(posterior_samples, f)
