data {
    int<lower=0> n_postal_codes; // number of postal code data points
    int<lower=0> n_postal_regions; // number of two-digit areas (groups)
    int<lower=1,upper=n_postal_regions> postal_region_ix[n_postal_codes]; // group indicator
    int n_affluent_households[n_postal_codes]; // observations
    int n_households[n_postal_codes];
}
transformed data {
    int<lower=0> n_national_households;
    int<lower=0> n_national_affluent;
    n_national_households = sum(n_households);
    n_national_affluent = sum(n_affluent_households);
}
parameters {
    vector[n_postal_regions] logit_p_regional;
    real<lower=0> national_sigma;
    real<lower=0, upper=1> national_mu;
}
transformed parameters {
    vector<lower=0, upper=1>[n_postal_regions] p_regional;
    // To use logit-normal, Stan likes to have untransformed variables on the left hand-side of
    // ~ -sign. Thus we will have 'logit_p_regional' follow normal, and only output the transformed
    // parameter 'p_regional' that we are interested in.
    p_regional = inv_logit(logit_p_regional);
}
model {
    national_mu ~ beta(n_national_affluent, n_national_households);
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