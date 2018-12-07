import pystan
import numpy as np
import pickle
import pandas as pd

stan_code = """
data {
    int<lower=0> n_postal_codes; // number of postal code data points
    int <lower=0> n_postal_regions; // number of two-digit areas (groups)
    int<lower=1,upper=n_postal_regions> postal_region_ix[n_postal_codes]; // group indicator
    int n_affluent[n_postal_codes]; // observations
    int n_households[n_postal_codes]; // total number of households per postal code
}
parameters {
  real mu_national;        // hyperprior mean
  real<lower=0> sigma_national;        // hyperprior std
  vector[n_postal_regions] mu_regional;        // group means
  vector<lower=0>[n_postal_regions] sigma_regional;        // group std
  vector[n_postal_codes] eta; // logit proportion for reparametrized binomial
}
model {
  mu_national ~ normal(-1, 2);
  sigma_national ~ normal(0, 3);
  sigma_regional ~ normal(0, 3);
  mu_regional ~ normal(mu_national, sigma_national);
  eta ~ normal(mu_regional[postal_region_ix], sigma_regional[postal_region_ix]);
  n_affluent ~ binomial_logit(n_households, eta);
}
generated quantities {
    vector[n_postal_codes] log_lik;
    for (i in 1:n_postal_codes)
        log_lik[i] = normal_lpdf(eta[i]| mu_regional[postal_region_ix[i]], sigma_regional);
}
"""
with open('paavodata_cleaned_df.pkl', 'rb') as f:
    paavo_df = pickle.load(f)

n_affluent_households = paavo_df['n_households_highest_income_2015']
n_households_total = paavo_df['n_households_2015']
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
model = pystan.StanModel(model_code=stan_code)
fit = model.sampling(data=data, iter=6000, chains=4)
print(fit)
extracts = fit.extract(permuted=True)
posterior_samples = [extracts[param] for param in ['mu_national', 'mu_regional', 'sigma_national', 'sigma_regional', 'eta', 'log_lik']]

with open('affluence_hierarchical_logit_bin.pkl', 'wb') as f:
    pickle.dump((postal_region_ix, posterior_samples), f)
