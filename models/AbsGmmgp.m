classdef AbsGmmgp<handle
    methods
        function obj=AbsGmmgp()
        end
        function sample_beta(obj,X,maxiter)
            % function sample_beta(obj,X,z,maxiter)
            % beta in R+
            % cov=beta*Sigma_rbf
            stepsize_beta=0.05;
            for j=1:maxiter*10
                for l=1:obj.par.nsubjects
                    new_model=copy(obj);
                    new_beta=exp(log(obj.par.sub(l).beta)+stepsize_beta*randn(1,1));
                    new_model.par.sub(l).beta=new_beta;
                    new_model.calcss(X,l);
                    
                    if rand<exp(new_model.llh-obj.llh)
                        obj.par.sub(l).beta=new_beta;
                        obj.calcss(X,l);
                    end
                end
            end
        end
        
        function sample_sig2het_w2homo(obj,X,maxiter)
            % heteroscedastic sig2
            % homoscedastic w2
            stepsize_w2=0.1;
            stepsize_sig2=0.1;
            z=obj.par.z;
            if isa(obj,'igmmgpmodel')
                K=max(z);
            else
                K=obj.par.K;
            end
            for j=1:maxiter
                s=obj.ss;
                sub=obj.par.sub;
                lpc=obj.logPc;
                lp=obj.logP;
                for l=1:obj.par.nsubjects
                    [T,~]=size(X{l});
                    sig2_new_vector=exp(log(sub(l).sig2)+stepsize_sig2*randn(obj.par.N,1));
                    lpc_local=lpc(:,l);
                    
                    for i=1:obj.par.N
                        k=z(i);
                        sig2_new=sig2_new_vector(i);
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
                    nu_tmp=zeros(K,1);
                    for i=1:K
                        nu_tmp(i)=sum(w2_new_vector(z==i)./sub(l).sig2(z==i));
                    end
                    XtVs_tmp=bsxfun(@times,sqrt(w2_new_vector')./sub(l).sig2',s(l).V'*X{l});
                    ids_tmp=bsxfun(@plus,1./s(l).D,repmat(nu_tmp',T,1));
                    logids_tmp=sum(log(ids_tmp));
                    vtxcs_tmp=zeros(size(s(l).Vtxcs));
                    for i=1:K
                        vtxcs_tmp(:,i)=sum(XtVs_tmp(:,z==i),2);
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
        
        function xmeans=getpostmeans(obj)
            z=obj.par.z;
            T=length(obj.ss(1).D);
            xmeans=zeros(T,max(z),obj.par.nsubjects);
            for l=1:obj.par.nsubjects
                for k=1:max(z)
                    xmeans(:,k,l)=(obj.ss(l).V*diag(1./(obj.ss(l).ids(:,k))))*obj.ss(l).Vtxcs(:,k);
                end
            end
        end        
        
        function sample_sig2homo_w2het(obj,X,maxiter)
            % homoscedastic sig2
            % heteroscedastic w2
            stepsize_w2=0.1;
            stepsize_sig2=0.1;
            if isa(obj,'igmmgpmodel')
                K=max(obj.par.z);
            else
                K=obj.par.K;
            end            
            for j=1:maxiter
                s=obj.ss;
                sub=obj.par.sub;
                M=obj.par.N;
                lpc=obj.logPc;
                lp=obj.logP;
                for l=1:obj.par.nsubjects
                    
                    T=size(X{l},1);
                    sig2_new_vector=exp(log(mean(sub(l).sig2))+stepsize_sig2*randn)*ones(size(sub(l).sig2));
                    
                    %evaluate sig2 sample
                    nu_tmp=zeros(K,1);
                    for i=1:K
                        nu_tmp(i)=sum(sub(l).w2(obj.par.z==i)./sig2_new_vector(obj.par.z==i));
                    end
                    XtVs_tmp=bsxfun(@times,sqrt(sub(l).w2')./sig2_new_vector',s(l).V'*X{l});
                    ids_tmp=bsxfun(@plus,1./s(l).D,repmat(nu_tmp',T,1));
                    logids_tmp=sum(log(ids_tmp));
                    vtxcs_tmp=zeros(size(s(l).Vtxcs));
                    for i=1:K
                        vtxcs_tmp(:,i)=sum(XtVs_tmp(:,obj.par.z==i),2);
                    end
                    sxcvids_tmp=sum((vtxcs_tmp.^2./ids_tmp));
                    lpc_new=-0.5*logids_tmp+0.5*sxcvids_tmp;
                    %                     logPdiff=-0.5*(1/sig2_new-1/sub(l).sig2(i))*sum(X{l}(:,i).^2)-T*0.5*(log(sig2_new)-log(sub(l).sig2(i)));
                    lp_new=-0.5*T*obj.par.N*log(2*pi)-0.5*T*sum(log(sig2_new_vector))-max(obj.par.z)/2*sum(log(s(l).D))-0.5*norm(bsxfun(@times,X{l},1./sqrt(sig2_new_vector')),'fro')^2;
                    if rand()<exp(sum(lpc_new)-sum(lpc(:,l))+lp_new-lp(l))
                        s(l).ids=ids_tmp;
                        s(l).logids=logids_tmp;
                        s(l).XtVs=XtVs_tmp;
                        s(l).Vtxcs=vtxcs_tmp;
                        s(l).sxcvids=sxcvids_tmp;
                        
                        sub(l).sig2=sig2_new_vector;
                        
                        lpc(:,l)=lpc_new;
                        lp(l)=lp_new;
                    end
                    
                    lpc_local=lpc(:,l);
                    w2_new_vector=max(1./(1+(1./sub(l).w2-1).*exp(-stepsize_w2*randn(obj.par.N,1))),1e-6*ones(obj.par.N,1));
                    for i=1:obj.par.N
                        k=obj.par.z(i);
                        w2_new=w2_new_vector(i);
                        ids_tmp=s(l).ids(:,k)+(w2_new-sub(l).w2(i))/sub(l).sig2(i);
                        logids_tmp=sum(log(ids_tmp));
                        Vtxcs_tmp=s(l).Vtxcs(:,k)+(sqrt(w2_new/sub(l).w2(i))-1)*s(l).XtVs(:,i);
                        sxcvids_tmp=sum(Vtxcs_tmp.^2./ids_tmp);
                        logPc_tmp=-0.5*logids_tmp+0.5*sxcvids_tmp;
                        changeOfVariableFactor=log(abs(sqrt(w2_new)-w2_new))-log(abs(sqrt(sub(l).w2(i))-sub(l).w2(i)));
                        if rand<exp(logPc_tmp-lpc_local(k)+changeOfVariableFactor)
                            s(l).XtVs(:,i)=sqrt(w2_new/sub(l).w2(i))*s(l).XtVs(:,i);
                            s(l).ids(:,k)=ids_tmp;
                            s(l).logids(k)=logids_tmp;
                            s(l).Vtxcs(:,k)=Vtxcs_tmp;
                            s(l).sxcvids(k)=sxcvids_tmp;
                            lpc_local(k)=logPc_tmp;
                            sub(l).w2(i)=w2_new;
                        end
                    end
                    lpc(:,l)=lpc_local;
                end
                obj.ss=s;
                obj.par.sub=sub;
                obj.logPc=lpc;
                obj.logP=lp;
            end
        end      
        
        function sample_sig2homo_w2homo(obj,X,maxiter)
            % homoscedastic sig2
            % homoscedastic w2
            stepsize_w2=0.1;
            stepsize_sig2=0.1;
            if isa(obj,'igmmgpmodel')
                K=max(obj.par.z);
            else
                K=obj.par.K;
            end              
            for j=1:maxiter
                for l=1:obj.par.nsubjects
                    T=size(X{l},1);
                    sig2_new_vector=exp(log(mean(obj.par.sub(l).sig2))+stepsize_sig2*randn)*ones(size(obj.par.sub(l).sig2));
                    
                    %evaluate sig2 sample
                    nu_tmp=zeros(K,1);
                    for i=1:K
                        nu_tmp(i)=sum(obj.par.sub(l).w2(obj.par.z==i)./sig2_new_vector(obj.par.z==i));
                    end
                    XtVs_tmp=bsxfun(@times,sqrt(obj.par.sub(l).w2')./sig2_new_vector',obj.ss(l).V'*X{l});
                    ids_tmp=bsxfun(@plus,1./obj.ss(l).D,repmat(nu_tmp',T,1));
                    logids_tmp=sum(log(ids_tmp));
                    vtxcs_tmp=zeros(size(obj.ss(l).Vtxcs));
                    for i=1:K
                        vtxcs_tmp(:,i)=sum(XtVs_tmp(:,obj.par.z==i),2);
                    end
                    sxcvids_tmp=sum((vtxcs_tmp.^2./ids_tmp));
                    lpc=-0.5*logids_tmp+0.5*sxcvids_tmp;
                    %                     logPdiff=-0.5*(1/sig2_new-1/sub(l).sig2(i))*sum(X{l}(:,i).^2)-T*0.5*(log(sig2_new)-log(sub(l).sig2(i)));
                    lp=-0.5*T*obj.par.N*log(2*pi)-0.5*T*sum(log(sig2_new_vector))-max(obj.par.z)/2*sum(log(obj.ss(l).D))-0.5*norm(bsxfun(@times,X{l},1./sqrt(sig2_new_vector')),'fro')^2;
                    if rand()<exp(sum(lpc)-sum(obj.logPc(:,l))+lp-obj.logP(l))
                        obj.ss(l).ids=ids_tmp;
                        obj.ss(l).logids=logids_tmp;
                        obj.ss(l).XtVs=XtVs_tmp;
                        obj.ss(l).Vtxcs=vtxcs_tmp;
                        obj.ss(l).sxcvids=sxcvids_tmp;
                        
                        obj.par.sub(l).sig2=sig2_new_vector;
                        
                        obj.logPc(:,l)=lpc;
                        obj.logP(l)=lp;
                    end
                    
                    %evaluate w2 sample
                    w2_new_vector=1./(1+(1./mean(obj.par.sub(l).w2)-1).*exp(-stepsize_w2*randn))*ones(size(obj.par.sub(l).w2));
                    nu_tmp=zeros(K,1);
                    for i=1:K
                        nu_tmp(i)=sum(w2_new_vector(obj.par.z==i)./obj.par.sub(l).sig2(obj.par.z==i));
                    end
                    XtVs_tmp=bsxfun(@times,sqrt(w2_new_vector')./obj.par.sub(l).sig2',obj.ss(l).V'*X{l});
                    ids_tmp=bsxfun(@plus,1./obj.ss(l).D,repmat(nu_tmp',T,1));
                    logids_tmp=sum(log(ids_tmp));
                    vtxcs_tmp=zeros(size(obj.ss(l).Vtxcs));
                    for i=1:K
                        vtxcs_tmp(:,i)=sum(XtVs_tmp(:,obj.par.z==i),2);
                    end
                    sxcvids_tmp=sum((vtxcs_tmp.^2./ids_tmp));
                    lpc=-0.5*logids_tmp+0.5*sxcvids_tmp;
                    logPrior=log(abs(sqrt(w2_new_vector(1))-sqrt(w2_new_vector(1))^2))...
                        -log(abs(sqrt(obj.par.sub(l).w2(1))-sqrt(obj.par.sub(l).w2(1))^2));
                    if rand()<exp(sum(lpc)-sum(obj.logPc(:,l))+logPrior)
                        obj.ss(l).ids=ids_tmp;
                        obj.ss(l).logids=logids_tmp;
                        obj.ss(l).XtVs=XtVs_tmp;
                        obj.ss(l).Vtxcs=vtxcs_tmp;
                        obj.ss(l).sxcvids=sxcvids_tmp;
                        
                        obj.logPc(:,l)=lpc;
                        
                        obj.par.sub(l).w2=w2_new_vector;
                    end
                end
                
            end
            
        end
        
        function sample_sig2_w2(obj,X,maxiter)
            z=obj.par.z;
            stepsize_w2=0.1;
            stepsize_sig2=0.1;
            
            for j=1:maxiter
                s=obj.ss;
                sub=obj.par.sub;
                M=obj.par.N;
                lpc=obj.logPc;
                lp=obj.logP;
                for l=1:obj.par.nsubjects
                    [T,~]=size(X{l});
                    sig2_new_vector=max(exp(log(sub(l).sig2)+stepsize_sig2*randn(obj.par.N,1)),1e-6*ones(obj.par.N,1));
                    w2_new_vector=max(1./(1+(1./sub(l).w2-1).*exp(-stepsize_w2*randn(obj.par.N,1))),1e-6*ones(obj.par.N,1));
                    lpc_local=lpc(:,l);
                    for i=1:obj.par.N
                        k=z(i);
                        sig2_new=sig2_new_vector(i);
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
                        
                        w2_new=w2_new_vector(i);
                        ids_tmp=s(l).ids(:,k)+(w2_new-sub(l).w2(i))/sub(l).sig2(i);
                        logids_tmp=sum(log(ids_tmp));
                        Vtxcs_tmp=s(l).Vtxcs(:,k)+(sqrt(w2_new/sub(l).w2(i))-1)*s(l).XtVs(:,i);
                        sxcvids_tmp=sum(Vtxcs_tmp.^2./ids_tmp);
                        logPc_tmp=-0.5*logids_tmp+0.5*sxcvids_tmp;
                        changeOfVariableFactor=log(abs(sqrt(w2_new)-w2_new))-log(abs(sqrt(sub(l).w2(i))-sub(l).w2(i)));
                        if rand<exp(logPc_tmp-lpc_local(k)+changeOfVariableFactor)
                            s(l).XtVs(:,i)=sqrt(w2_new/sub(l).w2(i))*s(l).XtVs(:,i);
                            s(l).ids(:,k)=ids_tmp;
                            s(l).logids(k)=logids_tmp;
                            s(l).Vtxcs(:,k)=Vtxcs_tmp;
                            s(l).sxcvids(k)=sxcvids_tmp;
                            lpc_local(k)=logPc_tmp;
                            sub(l).w2(i)=w2_new;
                        end
                    end
                    % update the parallel stuff
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