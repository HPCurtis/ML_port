// Bayesian ridge regression based onhttps://github.com/sara-vanerp/bayesreg/blob/master/src/stan_files/lm_noNA_ridge_FB.stan
// Differences the DV and IV standardised within tranformed parameters block and y_test vectorised for greater speed.
data{
	int N_train; //number of observations training and validation set
	int p; //number of predictors
	vector[N_train] y_train; //response vector
	matrix[N_train, p] X_train; //model matrix
	//test set
	int N_test; //number of observations test set
	matrix[N_test, p] X_test; //model matrix test set
}
transformed data {
  matrix[N_train, p] xtrain_std;
  matrix[N_test, p] xtest_std;
  vector[N_train] ytrain_std;

  for (i in 1:p){
      xtrain_std[ ,i] = (X_train[, i]  - mean(X_train[, i])) / sd(X_train[ ,i]);
      xtest_std[ ,i] = (X_test[, i] - mean(X_test[ ,i])) / sd(X_test[ ,i]);
    }
    
  // Standardise Dependent variable
  ytrain_std = (y_train - mean(y_train)) / sd(y_train);
}
parameters{
	real mu; //intercept
	real<lower = 0> sigma2; //error variance
	vector[p] beta; // regression parameters
	//hyperparameters prior
	real<lower = 0> lambda; //penalty parameter
}
transformed parameters{
	real<lower = 0> sigma; //error sd
	vector[N_train] linpred; //mean normal model
	sigma = sqrt(sigma2);
	linpred = mu + xtrain_std*beta;
}
model{
 //prior regression coefficients: lasso
	beta ~ double_exponential(0, sigma/lambda);
	lambda ~ cauchy(0, 1);
	
 //priors nuisance parameters: uniform on log(sigma^2) & mu
	target += -2 * log(sigma); 
	
 //likelihood
	ytrain_std ~ normal(linpred, sigma);
}
generated quantities{ //predict responses test set
	real y_test[N_test]; //predicted responses
	y_test = normal_rng(mu + xtest_std * beta, sigma);
    vector[N] log_lik;
      for (n in 1:N) {
        log_lik[n] = normal_lpmf(ytrain_std[n] | mu + xtrain_std[n] * beta);
      }
}	