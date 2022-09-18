# Intro
The purpose of this repository is to hold test files and POCs as I learn about Machine Learning.

## Notes on perceptron convergence proof
- the proof uses the Cauchy–Schwarz inequality, an old friend from college.  It would be very useful to learn this again, in a few different contexts!

## Helpful concepts
- Bayes optimal classifier
- In some sense, all of ML aims to estimate the distribution P(X,Y)
  - Generative learning: estimate P(X,Y) = P(X|Y)P(Y)
  - Discriminative learning: only estimate P(Y|X) directly (most approaches today are discriminative)
- Maximum Likelihood Estimation (MLE): θ^MLE=argmaxθP(D;θ)
- Maximum a Posteriori Probability Estimation (MAP): θ^MAP=argmaxθP(θ∣D)
- Beyesian vs frequentist statistics