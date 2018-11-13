import pystan
import numpy as np
import pickle

stan_code = """
data {
    int<lower=0> N;  // number of data points
    vector[N] x;
    vector[N] y;
    real x_pred;
    real tau;
}
parameters {
    real alpha;
    real beta;
    real sigma;
    real y_pred;
}
transformed parameters {
    vector[N] mu_y_mean;
    real mu_y_pred;
    mu_y_mean = alpha + beta*x;
    mu_y_pred = alpha + beta*x_pred;
}
model {
    beta ~ normal(0, tau);
    y ~ normal(mu_y_mean, sigma);
    y_pred ~ normal(mu_y_pred, sigma);
}

"""
drowning_data = np.loadtxt('drowning.txt')
N = drowning_data.shape[0]
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
