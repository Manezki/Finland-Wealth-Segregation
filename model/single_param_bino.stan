data {
    int<lower=0> n_postal_codes; // number of postal code data points
    int<lower=0> n_postal_regions; // number of two-digit areas (groups)
    int<lower=1,upper=n_postal_regions> postal_region_ix[n_postal_codes]; // group indicator
    int n_affluent_households[n_postal_codes]; // observations
    int n_households[n_postal_codes];
}
transformed data {
    int<lower=0> national_households;
    int<lower=0> national_affluent;
    national_households = sum(n_households);
    national_affluent = sum(n_affluent_households);
}
parameters {
    vector[n_postal_regions] logit_p_regional;
    real<lower=0> national_sigma;
    real<lower=0, upper=1> national_mu;
}
transformed parameters {
    vector<lower=0, upper=1>[n_postal_regions] p_regional;
    p_regional = inv_logit(logit_p_regional);
}
model {
    national_mu ~ beta(national_affluent, national_households);
    logit_p_regional ~ normal(national_mu, national_sigma);
    for (i in 1:n_postal_codes) {
          n_affluent_households[i] ~ binomial(n_households[i], p_regional[postal_region_ix[i]]);
    }
}
generated quantities {
    vector[n_postal_codes] log_lik;
    for (i in 1:n_postal_codes) {
        log_lik[i] = binomial_log(n_affluent_households[i], n_households[i], p_regional[postal_region_ix[i]]);
    }
}