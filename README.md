# Intro

The purpose of this repository is to hold test files and POCs as I learn about Machine Learning.

## Notes on perceptron convergence proof

- the proof uses the Cauchy–Schwarz inequality, an old friend from college. It would be very useful to learn this again, in a few different contexts!

## Helpful concepts

- Bayes optimal classifier
- In some sense, all of ML aims to estimate the distribution `P(X,Y)`
  - Generative learning: estimate `P(X,Y) = P(X|Y)P(Y)`
  - Discriminative learning: only estimate `P(Y|X)` directly (most approaches today are discriminative)
  - note that if I know `P(Y|X)`, prediction is trivial. Simply input X, and test Y to find `max(P(Y|X))`
- Maximum Likelihood Estimation (MLE): `θ_MLE=argmaxθ[P(D;θ)]`
- Maximum a Posteriori Probability Estimation (MAP): `θ_MAP=argmaxθ[P(θ∣D)]`
- Beyesian vs frequentist statistics
- Naive Bayes (`argmax_y[P(y)sum(log(P(x_α|y)))]`)
  - especially useful in instances where the dimensions of the feature are truly independent, like when there is a _causal relationship_ between the label and feature components. For example, symptoms which are caused by the label y which is having the disease or not.
  - `P(X|Y) = Π_α(P(x_α|Y))` is an _overestimation_ in the case where P(x_1|x_2) is low. P(email is spam | occurance of word Nigeria) is high; P(email is spam | occurance of word viagra) is high; P(email is spam | occurance of word Nigeria & viagara) is low

### Categories of Machine Learning

- Supervised
  - classification
  - regression
- Unsupervised
  - dimensionality reduction
  - clustering
- Reinforcement learning (interactions between an _agent_ and its _environment_)

## Some notational stuff

- `|` vs `;`
  - `|` notates the classical _conditional probability density_
  - `P(x;θ)` notates fixed properties θ of a function; think of it like the _settings_ or (more formally) the _parameters_ of `P(x;θ)`. It's as if we've created a new function `g(x)`; i.e. x is not _contingent upon_ θ
  - Frequentest vs. Bayesian (`P(D;θ)` vs `P(D|θ)`): Frequentists say θ is a parameter without an event or sample space or distribution P(θ); i.e. θ is not associated with an outcome. Bayesians disagree, saying it is indeed a random variable that can be conditioned upon. Bayesians include a prior belief about what `P(θ)` _should be_
  - the beauty of Beyesian stats: θ being a random variable means we can integrate over it (integrate over all possible models) to obtain a _model independent_ formula for `P(x|D)`

## Summary of MLE vs MAP

- Say we have data set D drawn from distribution P `D ~ P(x,y)`, say we have n draws of our training data
- We do not know P, otherwise we could use the trivial Bayes Optimal Classifier and know we can't do any better with a prediction
- So we _estimate_ P with a distribution we understand, say P_θ with parameters θ. There are two methods to do this, MLE & MAP
  - MLE maximizes the likelihood of getting the data we observed: `θ^MLE=argmaxθP(D;θ)`
  - MAP says given that we observed the data, what is the most likely θ: `θ^MLE=argmaxθP(D;θ)`. There is no sample space for θ, but that shouldn't stop us from finding the Prior Distribution `P(θ)`. `P(y|X=x) = ∫_θ P(y|θ)P(θ|D)dθ`

## Continuous features (as opposed to categorical)

In the case of data with continuous features, a decent estimator for the probability distribution of a particular feature may be a gaussian distribution with parameters μ and σ^2. These parameters can be calculated easily (see 33:50 in Lecture 10) from the data set for use in MLE. The analog for categorial features is parameter θ in a binomial or multinomial distribution.

## Logistic Algorithm

The key difference between naive Bayes and the Perceptron is that naive Bayes first models the data and then throws it out, operating only off of the distributions `P(X|Y)`. Perceptron performs repeated calculation on the data itself.
Naive Bayes is also a linear classifier, both in the case of multinomial and gausian distributions. Demo idea: take a generic data set and an interface for a linear classifier and make the two algorithms compete.
The Logistic Algorithm is the discriminative counterpart to Naive Bayes. Instead of estimating the distributions, estimate `(w, b)` directly.
Start with the assumption that `P(Y|X) = 1/(1+e^(wTx))`, and use MLE to find the parameters `(w, b)`. -> `argmax[w,b]ΠiP(yi|x;w,b)`

Naive Bayes works well when you have very few data points, because you bring a gaussian assumption to the model to reduce the hyperplane search space. When you have many data points, logistic regression is preferable. (lec 12 min 7)

## Real world problem

### Problem statement

Given the words in the title of one of my todo app stories, classify it as belonging to one of my tags.
`Y ∈ {work, chess engine, music practice, todo app, life, research}`
`P(Y = work task | X = "draft design doc for API")`

I want to build a framework that tries out the 3 classifiers I know about so far

- k-nearest-neighbors
- Perceptron
- Naive Bayes

Requirements

- common interface
- should allow for smoothing
- in the case of perceptron, catch an infinite loop

KNN results (next try sharding the dataset)

k = 3
5250 6959 0.7544187383244719

k = 5
5351 6959 0.7689323178617617

k = 4 with tiebreaker `constant_classifier`
5411 6959 0.7775542462997557

k = 6 with tiebreaker `constant_classifier`
5424 6959 0.7794223307946544

k = 111
5347 6959 0.7683575226325622


NB results
5489 6959 0.7887627532691479
