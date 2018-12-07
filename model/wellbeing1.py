import pystan
import numpy as np
import pickle
import pandas as pd

stan_code = """
data {
    int<lower=0> n_postal_codes; // number of postal code data points
    int <lower=0> n_postal_regions; // number of two-digit areas (groups)
    int<lower=1,upper=n_postal_regions> postal_region_ix[n_postal_codes]; // group indicator
    vector[n_postal_codes] pct_affluent_households; // observations
}
parameters {
  real<lower=0> mu_national;        // hyperprior mean
  real<lower=0> sigma_national;        // hyperprior mean
  vector<lower=0>[n_postal_regions] mu_regional;        // group means
  real<lower=0> sigma_regional;     // group stds
}

model {
  mu_regional ~ normal(mu_national, sigma_national);
  pct_affluent_households ~ normal(mu_regional[postal_region_ix], sigma_regional);
}
generated quantities {
    vector[n_postal_codes] log_lik;
    for (i in 1:n_postal_codes)
        log_lik[i] = normal_lpdf(pct_affluent_households[i] | mu_regional[postal_region_ix[i]], sigma_regional);
}

"""
with open('paavodata_cleaned_df.pkl', 'rb') as f:
    paavo_df = pickle.load(f)

n_postal_regions = paavo_df['postal_region'].nunique()
n_postal_codes = paavo_df.shape[0]
postal_region_ix = paavo_df['postal_region_ix']
pct_affluent_households = paavo_df['n_households_highest_income_2015_pc']
print(f'n_postal_codes={n_postal_codes}, n_postal_regions={n_postal_regions}, postal_region_ix={postal_region_ix}')
print(f'pct_affluent_households={pct_affluent_households}')
data = dict(n_postal_codes=n_postal_codes,n_postal_regions=n_postal_regions, pct_affluent_households = pct_affluent_households, postal_region_ix=postal_region_ix)

model = pystan.StanModel(model_code=stan_code)
fit = model.sampling(data=data, iter=1000, chains=4)
print(fit)
extracts = fit.extract(permuted=True)
posterior_samples = [extracts[param] for param in ['mu_national', 'mu_regional', 'sigma_national', 'sigma_regional', 'log_lik']]

with open('wellbeing_hierarchical_gaussian.pkl', 'wb') as f:
    pickle.dump(posterior_samples, f)
