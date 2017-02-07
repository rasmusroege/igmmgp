classdef gmmddmodel<AbsGMMdd_sh & AbsFiniteModel & handle
    % gmmdmodel
    properties
        par;
        ss;
        logPc;
        logP;
        logZ;
    end
    
    methods
        function obj=gmmddmodel(x,z,K,par)
            if nargin==0
                super_args={};
            else
                super_args={x,z,K};
            end
            obj@AbsFiniteModel(super_args{:});
            obj.par.hypersamplers=[obj.par.hypersamplers {'sample_gamma','sample_v0','sample_lambda'}];
            if nargin==0
                return;
            end
            obj.logZ=-0.5*obj.par.N*obj.par.T*log(2*pi)*obj.par.nsubjects;
            for l=1:obj.par.nsubjects
                obj.par.sub(l).v0=obj.par.T;
                obj.par.sub(l).gamma=1;
                obj.par.sub(l).lambda=1;
                obj.par.sub(l).mu0=zeros(obj.par.T,1);
            end
            calcss(obj,x);
        end
        
        function samplerstr=get_samplers(obj)
            %samplerstr={'sample_gamma_diag','sample_v0_diag','sample_lambda','sample_sig2_w'};
            samplerstr=obj.par.hypersamplers;
        end
        
        function return_obj=copy(obj)
            return_obj=gmmddmodel();
            return_obj.par=obj.par;
            return_obj.ss=obj.ss;
            return_obj.logPc=obj.logPc;
            return_obj.logP=obj.logP;
            return_obj.logZ=obj.logZ;
        end
        
        function setpar(obj,par)
            obj.par.N=par.N;
            obj.par.T=par.p;
            obj.par.sigma0=par.Sigma0;
            obj.par.alpha=par.alpha;
            obj.par.nsubjects=length(par.lambda);
            for l=1:obj.par.nsubjects
                obj.par.sub(l).sig2=par.sig2{l};
                obj.par.sub(l).w=par.wei{l};
                obj.par.sub(l).v0=par.v0{l};
                obj.par.sub(l).gamma=par.gamma{l};
                obj.par.sub(l).lambda=par.lambda{l};
                obj.par.sub(l).mu0=par.mu0{l};
            end
        end
        
        function lq=sample_v0(obj,X,maxiter)
            lq=0;
            stepsize_v0=0.1;
            for i=1:maxiter
                v0=arrayfun(@(x)x.v0,obj.par.sub);
                n=copy(obj);
                v0new=exp(log(v0)+stepsize_v0*randn(size(v0)));
                for l=1:obj.par.nsubjects;n.par.sub(l).v0=v0new(l);end;
                n.calcss(X);
                for l=1:obj.par.nsubjects
                    %                     if rand<(v0new(l)/v0(l))*exp(sum(obj.logPc(:,l))-sum(n.logPc(:,l)));
                    if rand<exp(sum(n.logPc(:,l))-sum(obj.logPc(:,l)));
                        obj.ss(l)=n.ss(l);
                        obj.logPc(:,l)=n.logPc(:,l);
                        obj.par.sub(l).v0=v0new(l);
                    end
                end
            end
        end
        
        function remove_empty_clusters(obj)
        end
        
        function remove_observation(obj,x,n)
            k=obj.par.z(n);
            for l=1:obj.par.nsubjects
                if obj.par.nk(k)==1
                    nk=obj.par.nk(k);
                    obj.ss(l).R2(:,k)=0;
                    obj.ss(l).xt=x{l}(:,n);
                    obj.ss(l).Sigma_avg(:,k)=0;
                    obj.ss(l).x_avg(:,k)=0;
                    obj.ss(l).xt2=x{l}(:,n).^2;
                    nk=nk-1;
                    obj.logPc(k,l)=0;
                else
                    lam=obj.par.sub(l).lambda;
                    mu0=obj.par.sub(l).mu0;
                    nk=obj.par.nk(k);
                    obj.ss(l).R2(:,k)=obj.ss(l).R2(:,k)+0.5/(nk+lam)*(obj.ss(l).x_avg(:,k)+lam*mu0).^2;
                    obj.ss(l).xt=x{l}(:,n);
                    obj.ss(l).xt2=x{l}(:,n).^2;
                    nk=nk-1;
                    obj.ss(l).x_avg(:,k)=obj.ss(l).x_avg(:,k)-obj.ss(l).xt;
                    obj.ss(l).Sigma_avg(:,k)=obj.ss(l).Sigma_avg(:,k)-obj.ss(l).xt2;
                    obj.ss(l).R2(:,k)=obj.ss(l).R2(:,k)-0.5*obj.ss(l).xt2-0.5/(nk+lam)*(obj.ss(l).x_avg(:,k)+lam*mu0).^2;
                    nn=(obj.par.nk(k)-1)/2+obj.par.sub(l).v0;
                    obj.logPc(k,l)=obj.ss(l).logPrior-(sum(nn.*log(obj.ss(l).R2(:,k)))-obj.par.T*sum(gammaln(nn))-obj.par.T/2*log(lam/(nk+lam)));
                end
            end
            obj.par.nk(k)=obj.par.nk(k)-1;
        end
        
        function add_observation(obj,n,k,~,comp,~)
            obj.par.nk(k,1)=obj.par.nk(k,1)+1;
            for l=1:obj.par.nsubjects
                obj.ss(l).x_avg(:,k)=obj.ss(l).x_avg(:,k)+obj.ss(l).xt;
                obj.ss(l).Sigma_avg(:,k)=obj.ss(l).Sigma_avg(:,k)+obj.ss(l).xt2;
            end
            if ~isempty(comp)
                for l=1:obj.par.nsubjects
                    obj.ss(l).R2(:,k)=obj.ss(l).R2t(:,k==comp);
                end
            else
                for l=1:obj.par.nsubjects
                    obj.ss(l).R2(:,k)=obj.ss(l).R2t(:,k);
                end
            end
        end
        
        function [categoricalDist,logPnew,logdiff,addss]=compute_categorical(obj,x,n,comp)
            K=obj.par.K;
            logPnew = zeros(K,obj.par.nsubjects);
            for l=1:obj.par.nsubjects
                %                 s=obj.ss(l);
                %                 lam=obj.par.sub(l).lambda;
                %                 mu0=obj.par.sub(l).mu0;
                %                 for k=1:K
                %                     obj.ss(l).R2t(:,k)=obj.ss(l).R0+0.5*obj.ss(l).Sigma_avg(:,k)+0.5*s.xt2-0.5/(obj.par.nk(k)+1+obj.par.sub(l).lambda)*(obj.ss(l).x_avg(:,k)+s.xt+obj.par.sub(l).lambda*obj.par.sub(l).mu0).^2;
                %                 end
                s=obj.ss(l);
                sav=[obj.ss(l).Sigma_avg];
                xav=[obj.ss(l).x_avg];
                r0=obj.ss(l).R0;
                nk=[obj.par.nk]';
                lam=obj.par.sub(l).lambda;
                mu0=obj.par.sub(l).mu0;
                obj.ss(l).R2t=bsxfun(@plus,r0+1/2*s.xt2,1/2*sav)+bsxfun(@times,-0.5./(nk+1+lam),bsxfun(@plus,xav,s.xt+lam*mu0).^2);
                nn=(nk+1)/2+obj.par.sub(l).v0;
                logPnew(:,l)=obj.ss(l).logPrior-(sum(bsxfun(@times,nn,log(obj.ss(l).R2t)))-obj.par.T*gammaln(nn)-obj.par.T/2*log(lam./((nk+1)+lam)));
            end
            logdiff=sum(logPnew-obj.logPc,2);
            categoricalDist=[obj.par.nk+obj.par.alpha/obj.par.K].*exp(logdiff-max(logdiff));
            addss=[];
        end
        
        function updateLogP(obj,x)
            
        end
        
        function calcss(obj,x)
            z=obj.par.z;
            val=unique(z);
            val=setdiff(val,0);
            obj.logPc=zeros(obj.par.K,obj.par.nsubjects);
            obj.logZ=0;
            obj.par.nk=zeros(length(val),1);
            for k=1:obj.par.K
                obj.par.nk(k,1)=sum(z==k);
            end
            for l=1:obj.par.nsubjects
                obj.ss(l).logPrior=obj.par.T*sum(obj.par.sub(l).v0.*log(obj.par.sub(l).gamma))-obj.par.T*gammaln(obj.par.sub(l).v0);
                obj.ss(l).Sigma_avg=zeros(obj.par.T,obj.par.K);
                obj.ss(l).R2=zeros(obj.par.T,obj.par.K);
                obj.ss(l).x_avg=zeros(obj.par.T,obj.par.K);
                for k=1:length(val)
                    idx=(z==val(k));
                    obj.ss(l).Sigma_avg(:,val(k))=sum(x{l}(:,idx).^2,2);
                    obj.ss(l).x_avg(:,val(k))=sum(x{l}(:,idx),2);
                    obj.ss(l).R0=obj.par.sub(l).gamma+0.5*obj.par.sub(l).lambda*obj.par.sub(l).mu0.^2;
                    obj.ss(l).R2(:,val(k))=obj.ss(l).R0+0.5*obj.ss(l).Sigma_avg(:,val(k))-...
                        0.5/(obj.par.nk(val(k))+obj.par.sub(l).lambda)*(obj.ss(l).x_avg(:,val(k))+...
                        obj.par.sub(l).lambda*obj.par.sub(l).mu0).^2;
                    nn=bsxfun(@plus,obj.par.nk(val(k))/2,obj.par.sub(l).v0);
                    obj.logPc(val(k),l)=obj.ss(l).logPrior-(sum(nn.*log(obj.ss(l).R2(:,val(k))))-obj.par.T*gammaln(nn)-obj.par.T/2*log(obj.par.sub(l).lambda/(obj.par.nk(val(k))+obj.par.sub(l).lambda)));
                end
                obj.logP(l)=0;
            end
            obj.updateLogZ();
        end
    end
end