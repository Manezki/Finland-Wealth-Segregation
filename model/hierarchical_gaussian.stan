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