function [z,best_sample,llh,noc,cputime]=infsample(X,model,opts)
% [best_sample,llh,noc,cputime]=infsample(X,model,opts)
[~,N]=size(X{1});
if nargin<3
    opts=struct;
end
global limit_sm;
global UseSequentialAllocation;
global UseSMspeedup;

if isfield(opts,'maxiter'); maxiter=opts.maxiter; else; maxiter=3   ;end;
if isfield(opts,'UseSequentialAllocation'); UseSequentialAllocation=opts.UseSequentialAllocation; else; UseSequentialAllocation=true; end
if isfield(opts,'UseSMspeedup'); UseSMspeedup=opts.UseSMspeedup; else; UseSMspeedup=true; end
if isfield(opts,'startiter'); startiter=opts.startiter; else, startiter=1;end;
if isfield(opts,'limit_sm'); limit_sm=opts.limit_sm; else limit_sm=inf;end
if isfield(opts,'llh'); llh=opts.llh; else, llh=model.llh;end;
if isfield(opts,'noc'); noc=opts.noc; else, noc=[];end;
if isfield(opts,'cputime'); cputime=opts.cputime; else, cputime=[];end;
if isfield(opts,'optim'); optim=opts.optim; else;  optim=0; end;
if isfield(opts,'best_sample');best_sample=opts.best_sample;else;best_sample=model;end;
if isfield(opts,'verbose');verbose=opts.verbose;else;verbose=1;end;

spacer = [repmat([repmat('-',1,14) '+'],1,4) repmat('-',1,13)];
dheader = sprintf(' %12s | %12s | %12s | %12s | %12s ','Iteration','logP', 'dlogP/|logP|', 'noc', class(model));

getnoc=@(x)length(unique(x.par.z));

model.calcss(X);
% Main loop
for iter=startiter:startiter+maxiter-1
    if mod(iter,30)==1 && verbose
        disp(spacer);disp(dheader);disp(spacer);
    end

    % Gibbs sample
    tic;
    if iter>=startiter+maxiter-1-optim
        gibbs_sample(X,model,randperm(N),[],[],optim);
    else
        gibbs_sample(X,model,randperm(N));
    end
    
    % SAMS / split-merge sample
    if isa(model,'AbsInfiniteModel')
        for k=1:max(max(model.par.z),5)
            if max(model.par.z)<limit_sm
                split_merge_sample(X,model);
            end
        end
    end
    
    % MH sample hyper parameters
    sample_hyperparameters(X,model);
    
    
    model.calcss(X);
    cputime=[cputime;toc];
    llh=[llh;model.llh];
    noc=[noc;getnoc(model)];
    
    if model.llh>best_sample.llh
        best_sample=copy(model);
    end
    
    dllh=(llh(end)-llh(end-1))/abs(llh(end));
    if verbose
        fprintf(' %12.0f | %12.4e | %12.4e | %12.0f | %s\n',iter,llh(end), dllh, getnoc(model),datestr(now));
    end
%     figure(1); plot(llh(2:end)); ylabel('Joint distribution'); xlabel('Iteration');  title(['number of components ' num2str(noc(end))]); drawnow;
end
z=model.par.z;

%--------------------------------------------------------------------
function model=sample_hyperparameters(X,model)
samplrstrs=model.get_samplers;
for sampler=1:length(samplrstrs)
    feval(str2func(samplrstrs{sampler}),model,X,5);
end

%--------------------------------------------------------------------
function split_merge_sample(X,model)
global UseSequentialAllocation;
global UseSMspeedup;
ngibbs_reps=0;
nsubjects=length(X);
[~,N]=size(X{1});
i1=ceil(N*rand);
i2=ceil(N*rand);
while i2==i1
    i2=ceil(N*rand);
end

if model.par.z(i1)==model.par.z(i2) % Split move
    % generate split configuration
    z_t=model.par.z;
    comp=[model.par.z(i1) max(model.par.z)+1];
    idx=(z_t==model.par.z(i1));
    if UseSequentialAllocation
        z_t(idx)=0;
    else
        z_t(idx)=comp(ceil(2*rand(sum(idx),1)));
    end
    z_t(i1)=comp(1);
    z_t(i2)=comp(2);
    idx(i1)=false;
    idx(i2)=false;
    
    model_split=model.initLaunch(X,z_t,comp);
    
    if sum(idx)>0
        nidx=find(idx)';
        rp_idx=randperm(sum(idx));
        for reps=1:ngibbs_reps+1
%             [z_t,par_t,logP_t,logQ]
            [logQ]=gibbs_sample(X,model_split,nidx(rp_idx),comp);
        end
    else
        logQ=0;
    end
    dllh=model_split.llh-model.llh;
    if rand<exp(dllh-logQ)
%         disp(['split component ' num2str(model.par.z(i1)) ' with delta llh: ' num2str(dllh)]);
        model.par.z=model_split.par.z;
        model.ss=model_split.ss;
        model.par.nk=model_split.par.nk;
        model.logPc=model_split.logPc;
        model.logZ=model_split.logZ;
        model.logP=model_split.logP;
        model_split.delete();
    else
        model_split.delete();
    end
else% merge move
    % generate merge configuration
    if model.par.z(i1)>model.par.z(i2);tmp=i2;i2=i1;i1=tmp;clear tmp;end;
    comp=[model.par.z(i1) model.par.z(i2)];
    z_merge=model.par.z;
    idx=(z_merge==z_merge(i1) | z_merge==z_merge(i2));
    z_merge(idx)=z_merge(i1);
    z_merge(z_merge>model.par.z(i2))=z_merge(z_merge>model.par.z(i2))-1;
    model_merge=model.initMerge(X,z_merge,comp);
    model_merge.updateLogZ;
    if UseSMspeedup
        accept_rate=rand();
    else
        accept_rate=-1;
    end
    if accept_rate<exp(model_merge.llh-model.llh)
        if ~UseSMspeedup
            accept_rate=rand();
        end
        z_t=model.par.z;
        if UseSequentialAllocation
            z_t(idx)=0;
        else
            z_t(idx)=comp(ceil(2*rand(sum(idx),1)));
        end
        z_t(i1)=model.par.z(i1);
        z_t(i2)=model.par.z(i2);
        idx(i1)=false;
        idx(i2)=false;
        model_launch=model.initLaunch(X,z_t,comp);
        if sum(idx)>0
            nidx=find(idx)';
            rp_idx=randperm(sum(idx));
            for reps=1:ngibbs_reps
                gibbs_sample(X,model_launch,nidx(rp_idx),comp);
            end
            [logQ]=gibbs_sample(X,model_launch,nidx(rp_idx),comp,model.par.z);
        else
            logQ=0;
        end
        
        if accept_rate<exp(model_merge.llh-model.llh+logQ)
%             disp(['merged component ' num2str(model.par.z(i1)) ' with component ' num2str(model.par.z(i2)) ' with delta llh: ' num2str(model_merge.llh-model.llh)]);
            model.par.z=model_merge.par.z;
            model.ss=model_merge.ss;
            model.par.nk=model_merge.par.nk;
            model.logPc=model_merge.logPc;
            model.logZ=model_merge.logZ;
            model.logP=model_merge.logP;
            model_launch.delete();
            model_merge.delete();
        else
            model_launch.delete();
            model_merge.delete();            
        end
    end
end
% remove empty clusters
model.remove_empty_clusters();

%----------------------------------------------------------------------------

function [logQ]=gibbs_sample(X,model,sample_idx,comp,forced,optim)
if nargin<6
    optim=0;
end
if nargin<5
    forced=[];
    optim=0;
end
if nargin<4
    comp=[];
    optim=0;
end
logQ=0;
% gibbs sample clusters
for n=sample_idx
    
    % if model.par.z(n)=0 the observation has not yet been assigned
    if model.par.z(n)~=0
        model.remove_observation(X,n)
    end
    
    % Evaluate the assignment of i'th covariance matrix to all clusters
    [categoricalDist,logPnew,logDiff,addss]=model.compute_categorical(X,n,comp);
    
    % sample from posterior
    if isempty(comp)
        if optim
            [~,knew]=max(categoricalDist);
            model.par.z(n)=knew;
        else
            model.par.z(n)=find(rand<cumsum(categoricalDist/sum(categoricalDist)),1,'first');
        end
        model.logPc(model.par.z(n),:)=logPnew(model.par.z(n),:);
    else
        if ~isempty(forced)
            model.par.z(n)=forced(n);
        else
            model.par.z(n)=comp(find(rand<cumsum(categoricalDist/sum(categoricalDist)),1,'first'));
        end
        q_tmp=logDiff-max(logDiff)+log(model.par.nk(comp));
        q_tmp=q_tmp-log(sum(exp(q_tmp)));
        logQ=logQ+q_tmp(model.par.z(n)==comp);
        model.logPc(model.par.z(n),:)=logPnew(comp==model.par.z(n),:);        
    end

    % Update sufficient statistics
    model.add_observation(n,model.par.z(n),addss,comp);
    
    % remove empty clusters
    model.remove_empty_clusters();
end
model.updateLogZ;
model.updateLogP(X);
