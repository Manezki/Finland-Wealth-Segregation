import pystan
import numpy as np
import pickle
import pandas as pd

stan_code = """
data {
    int<lower=0> N;
    nchildless[N];
    nfamilies[N];
}
parameters {
    vector[N] wellbeing;
    real beta_childless;
    real baseline_childless;
}
transformed parameters {
}
model {
    nchildless ~ poisson_log(beta_childless*wellbeing + baseline_childless + nfamilies);
}

"""
paavo_df = pd.read_csv('data/paavo_9_koko.csv', sep=';')

paavo_sub = paavo_df[paavo_df['Young single persons, 2016 (TE)'] != "."] # Exclude empty observations
paavo_sub = paavo_sub[paavo_sub['Young couples without children, 2016 (TE)'] != "."]
# TODO paavo_sub still contains ".." observations.

#paavo_sub['nchildless'] = paavo_sub['Young single persons, 2016 (TE)'].astype(int) + paavo_sub['Young couples without children, 2016 (TE)'].astype(int)
print(paavo_sub['nchildless'][:20])
'''
N = paavo_df.shape[0]
x = drowning_data[:,0]
y = drowning_data[:,1]
x_pred = 2019
tau = 26.787399999995834
data = dict(N=N,x=x, y=y, x_pred=x_pred, tau=tau)
model = pystan.StanModel(model_code=stan_code)
fit = model.sampling(data=data, iter=1000, chains=4)
print(fit)
extracts = fit.extract(permuted=True)

posterior_samples = [extracts[param] for param in ['alpha', 'beta', 'mu_y_mean', 'sigma', 'mu_y_pred', 'y_pred']]

with open('posterior_samples.pkl', 'wb') as f:
    pickle.dump(posterior_samples, f)
'''
