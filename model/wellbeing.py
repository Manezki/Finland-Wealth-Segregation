import pystan
import numpy as np
import pickle
import pandas as pd

stan_code = """
data {
    int<lower=0> N;
    int<lower=0> nchildless[N];
    vector<lower=0>[N] nfamilies;
}
parameters {
    vector[N] wellbeing;
    real beta_childless;
    real baseline_childless;
}
transformed parameters {

}
model {
    // nchildless ~ poisson_log(beta_childless*wellbeing + baseline_childless + nfamilies);
    // Changed data to logscale.
    nchildless ~ poisson_log(beta_childless*wellbeing + log(baseline_childless) + log(nfamilies));
}

"""
paavo_df = pd.read_csv('data/paavo_9_koko.csv', sep=';')

# Drop no observations from the DF
paavo_sub = paavo_df[(paavo_df['Young single persons, 2016 (TE)'] != ".") & (paavo_df["Young single persons, 2016 (TE)"] != "..")] # Exclude empty observations
paavo_sub = paavo_sub[paavo_sub['Young couples without children, 2016 (TE)'] != "."]


# Drop whole Finland
paavo_sub = paavo_sub.iloc[1:, :]

# Include nchildless
paavo_sub['nchildless_young'] = paavo_sub['Young single persons, 2016 (TE)'].astype(int) + paavo_sub['Young couples without children, 2016 (TE)'].astype(int)
paavo_sub['nchildless'] = paavo_sub['Adult households, 2016 (TE)'].astype(int) + paavo_sub['Pensioner households, 2016 (TE)'].astype(int)
paavo_sub["nfamilies"] = paavo_sub["Households with children, 2016 (TE)"].astype(int)

# IF one wants to work only with capital area.
# Contains Helsinki, Vantaa, Espoo + Some neighbours
paavo_sub = paavo_sub.loc[paavo_sub["Postal code area"].apply(lambda x: x[:2] in ["00", "01", "02"])]

data = dict(
    N=paavo_sub.shape[0],
    nchildless=paavo_sub['nchildless'],
    nfamilies=paavo_sub['nfamilies']
    )

model = pystan.StanModel(model_code=stan_code)
fit = model.sampling(data=data, iter=1000, chains=4)
print(fit)
extracts = fit.extract(permuted=True)

posterior_samples = [extracts[param] for param in ['wellbeing', 'beta_childless', 'baseline_childless']]

with open('posterior_samples.pkl', 'wb') as f:
    pickle.dump(posterior_samples, f)
