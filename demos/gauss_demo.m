
addpath(genpath('..'));
K=10;
N=2000;
T=200;
S=1;
z=kron(1:K,ones(1,N/K,1))';
muk=randn(T,K,S);
x=repmat({zeros(T,N)},S,1);
for s=1:S
    for k=1:K
        x{s}(:,z==k)=bsxfun(@plus,10*muk(:,k,s),randn(T,sum(z==k),1));
    end
end
m=ivmfmodel(x,randi(K,N,1));
if isa(m,'AbsVMF')
    for s=1:S
        x{s}=x{s}./sqrt(sum(x{s}.^2));
    end
    m.calcss(x);
end
mfig('test');clf;hold on;
for l=1:20
    for j=1:length(m.get_samplers)
        feval(m.get_samplers{j},m,x,5);
    end
    plot(l,m.llh,'x');
    drawnow;
end

o=struct();
o.maxiter=30;
infsample(x,m,o);