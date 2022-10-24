# Intro
The purpose of this repository is to hold test files and POCs as I learn about Machine Learning.

## Notes on perceptron convergence proof
- the proof uses the Cauchy–Schwarz inequality, an old friend from college.  It would be very useful to learn this again, in a few different contexts!

## Helpful concepts
- Bayes optimal classifier
- In some sense, all of ML aims to estimate the distribution `P(X,Y)`
  - Generative learning: estimate `P(X,Y) = P(X|Y)P(Y)`
  - Discriminative learning: only estimate `P(Y|X)` directly (most approaches today are discriminative)
  - note that if I know `P(Y|X)`, prediction is trivial.  Simply input X, and test Y to find `max(P(Y|X))`
- Maximum Likelihood Estimation (MLE): `θ_MLE=argmaxθ[P(D;θ)]`
- Maximum a Posteriori Probability Estimation (MAP): `θ_MAP=argmaxθ[P(θ∣D)]`
- Beyesian vs frequentist statistics
- Naive Bayes (`argmax_y[P(y)sum(log(P(x_α|y)))]`)
  - especially useful in instances where the dimensions of the feature are truly independent, like when there is a _causal relationship_ between the label and feature components.  For example, symptoms which are caused by the label y which is having the disease or not.
  - `P(X|Y) = Π_α(P(x_α|Y))` is an _overestimation_ in the case where P(x_1|x_2) is low.  P(email is spam | occurance of word Nigeria) is high; P(email is spam | occurance of word viagra) is high; P(email is spam | occurance of word Nigeria & viagara) is low

## Some notational stuff
- `|` vs `;`
  - `|` notates the classical _conditional probability density_
  - `P(x;θ)` notates fixed properties θ of a function; think of it like the _settings_ or (more formally) the _parameters_ of `P(x;θ)`.  It's as if we've created a new function `g(x)`; i.e. x is not _contingent upon_ θ
  - Frequentest vs. Bayesian (`P(D;θ)` vs `P(D|θ)`): Frequentists say θ is a parameter without an event or sample space or distribution P(θ); i.e. θ is not associated with an outcome.  Bayesians disagree, saying it is indeed a random variable that can be conditioned upon.  Bayesians include a prior belief about what `P(θ)` _should be_
  - the beauty of Beyesian stats: θ being a random variable means we can integrate over it (integrate over all possible models) to obtain a _model independent_ formula for `P(x|D)`

## Summary of MLE vs MAP
- Say we have data set D drawn from distribution P `D ~ P(x,y)`, say we have n draws of our training data
- We do not know P, otherwise we could use the trivial Bayes Optimal Classifier and know we can't do any better with a prediction
- So we _estimate_ P with a distribution we understand, say P_θ with parameters θ.  There are two methods to do this, MLE & MAP
  - MLE maximizes the likelihood of getting the data we observed: `θ^MLE=argmaxθP(D;θ)`
  - MAP says given that we observed the data, what is the most likely θ: `θ^MLE=argmaxθP(D;θ)`.  There is no sample space for θ, but that shouldn't stop us from finding the Prior Distribution `P(θ)`. `P(y|X=x) = ∫_θ P(y|θ)P(θ|D)dθ`