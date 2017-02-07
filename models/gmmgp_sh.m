classdef gmmgp_sh<handle & gmmgpmodel
    methods
        function obj=gmmgp_sh(x,z)
            if nargin== 0
                super_args={};
            elseif nargin==2
                super_args={x,z};
            end
            obj=obj@gmmgpmodel(super_args{:});
            if nargin==2
                for l=1:obj.par.nsubjects
                    obj.par.sub(l).w2=mean(obj.par.sub(l).w2)*ones(size(obj.par.sub(l).w2));
                end
                obj.calcss(x);
            end
        end
        
        function returnobj=copy(obj)
            returnobj=gmmgp_sh();
            returnobj.par=obj.par;
            returnobj.ss=obj.ss;
            returnobj.logZ=obj.logZ;
            returnobj.logP=obj.logP;
            returnobj.logPc=obj.logPc;
        end
        
        function sample_sig2_w2(obj,X,maxiter,sample_idx)
            % homoscedastic sig2
            % homoscedastic w2
            if nargin<5
                sample_idx=1:obj.par.N;
            end
            Nsample=length(sample_idx);
            stepsize_w2=0.05;
            stepsize_sig2=0.05;
            for j=1:maxiter
                s=obj.ss;
                sub=obj.par.sub;
                lpc=obj.logPc;
                lp=obj.logP;
                for l=1:obj.par.nsubjects
                    [T,~]=size(X{l});
                    sig2_new_vector=exp(log(sub(l).sig2(sample_idx))+stepsize_sig2*randn(Nsample,1));
                    lpc_local=lpc(:,l);
                    
                    for y=1:Nsample
                        i=sample_idx(y);
                        k=obj.par.z(i);
                        sig2_new=sig2_new_vector(y);
                        ids_tmp=s(l).ids(:,k)+sub(l).w2(i)*(1/sig2_new-1/sub(l).sig2(i));
                        logids_tmp=sum(log(ids_tmp));
                        Vtxcs_tmp=s(l).Vtxcs(:,k)+(sub(l).sig2(i)/sig2_new-1)*s(l).XtVs(:,i);
                        sxcvids_tmp=sum(Vtxcs_tmp.^2./ids_tmp);
                        logPc_tmp=-0.5*logids_tmp+0.5*sxcvids_tmp;
                        
                        logPdiff=-0.5*(1/sig2_new-1/sub(l).sig2(i))*sum(X{l}(:,i).^2)-T*0.5*(log(sig2_new)-log(sub(l).sig2(i)));
                        if rand<exp(logPc_tmp-lpc_local(k)+logPdiff)
                            s(l).ids(:,k)=ids_tmp;
                            s(l).logids(k)=logids_tmp;
                            s(l).XtVs(:,i)=sub(l).sig2(i)/sig2_new*s(l).XtVs(:,i);
                            s(l).Vtxcs(:,k)=Vtxcs_tmp;
                            s(l).sxcvids(k)=sxcvids_tmp;
                            lpc_local(k)=logPc_tmp;
                            sub(l).sig2(i)=sig2_new;
                            lp(l)=lp(l)+logPdiff;
                        end
                    end
                    
                    %evaluate w2 sample
                    w2_new_vector=(1./(1+1./mean(sub(l).w2)-1).*exp(-stepsize_w2*randn))*ones(size(sub(l).w2));
                    nu_tmp=zeros(max(obj.par.z),1);
                    for i=1:max(obj.par.z)
                        nu_tmp(i)=sum(w2_new_vector(obj.par.z==i)./sub(l).sig2(obj.par.z==i));
                    end
                    XtVs_tmp=bsxfun(@times,sqrt(w2_new_vector')./sub(l).sig2',s(l).V'*X{l});
                    ids_tmp=bsxfun(@plus,1./s(l).D,repmat(nu_tmp',T,1));
                    logids_tmp=sum(log(ids_tmp));
                    vtxcs_tmp=zeros(size(s(l).Vtxcs));
                    for i=1:max(obj.par.z)
                        vtxcs_tmp(:,i)=sum(XtVs_tmp(:,obj.par.z==i),2);
                    end
                    sxcvids_tmp=sum((vtxcs_tmp.^2./ids_tmp));
                    logPc_tmp=-0.5*logids_tmp+0.5*sxcvids_tmp;
                    logPrior=log(abs(sqrt(w2_new_vector(1))-sqrt(w2_new_vector(1))^2))...
                        -log(abs(sqrt(obj.par.sub(l).w2(1))-sqrt(obj.par.sub(l).w2(1))^2));
                    if rand()<exp(sum(logPc_tmp)-sum(lpc_local)+logPrior)
                        s(l).ids=ids_tmp;
                        s(l).logids=logids_tmp;
                        s(l).XtVs=XtVs_tmp;
                        s(l).Vtxcs=vtxcs_tmp;
                        s(l).sxcvids=sxcvids_tmp;
                        
                        sub(l).w2=w2_new_vector;
                        lpc_local=logPc_tmp;
                    end
                    lpc(:,l)=lpc_local;
                end
                obj.ss=s;
                obj.par.sub=sub;
                obj.logPc=lpc;
                obj.logP=lp;
            end
        end
    end
end