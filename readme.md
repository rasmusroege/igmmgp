# igmmgp

<code>
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
</code>

<code>
z_init=randi(K,N,1);
m=igmmsmodel(x,z_init);
infsample(x,m);
</code>
