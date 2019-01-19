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
        log_lik[i] = binomial_logit_lpmf(n_affluent[i] | n_households[i], eta[i]);
}