# Sampling based mixture modeling in Matlab

Sampling based mixture modeling software tool. Gibbs sampler and split-merge / SAMS proposals for sampling clustering configuration. Metropolis-Hastings sampling of hyperparameters. 

We generate a small dataset that is used in the clustering with S subjects consisting of N observations of dimension T in K clusters.
```matlab
K=10;
N=100;
T=20;
S=3;
z=kron((1:K)’,ones(N/K,1));
% Clustering vector, see Fig. 4.1
muk=randn(T,K,S);
% Generate cluster means
x=repmat({zeros(T,N)},S,1);
% cell array with observations
for s=1:S
  for k=1:K
    % generate observations for cluster k
    x{s}(:,z==k)=bsxfun(@plus,muk(:,k,s),randn(T,sum(z==k),1));
  end;
end
```
We can use the infinite spherical Gaussian Mixture model to cluster the generated dataset by the following code snippet:
```matlab
z_init=randi(K,N,1);
m=igmmsmodel(x,z_init);
infsample(x,m);
```

List of probabilistic models: 
* Spherical Gaussian mixture model.
* Diagonal matrix Gaussian mixture model.
* Gaussian mixture model with Gaussian process prior on cluster means.
* von-Mises Fisher mixture model.

Two papers using this code will hopefully get published and may be referenced as: 

Roege, R. E., Madsen, K. H., Schmidt, M. N., Moerup, M. (Dec. 2016) Infinite von Mises-Fisher Modeling of Whole-Brain fMRI Data, Submitted.

Roege, R. E., Schmidt, M. N., Churchill, N. W., Madsen, K. H., Moerup, M. (Feb. 2017) Functional Whole-Brain Parcellation using Bayesian Non-Parametric Modeling, Manuscript.
