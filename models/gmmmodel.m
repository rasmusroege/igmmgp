classdef gmmmodel < AbsGMM & AbsFiniteModel & handle
    properties
        ss;
        par;
        logPc;
        logP;
        logZ;
    end
    
    methods
        function obj=gmmmodel(x,z,K)
            if nargin==0
                super_args={};
            else
                super_args={x,z,K};
            end
            obj=obj@AbsFiniteModel(super_args{:});
            obj.par.hypersamplers=[obj.par.hypersamplers {'sample_gamma','sample_lambda'}];
            if nargin==0
                return;
            end
            obj.logZ=-0.5*obj.par.N*obj.par.T*log(2*pi)*obj.par.nsubjects;
            for l=1:obj.par.nsubjects
                obj.par.sub(l).v0=obj.par.T;
                obj.par.sub(l).Sigma0=obj.par.T*eye(obj.par.T);
                obj.par.sub(l).gamma=1e-3;
                obj.par.sub(l).lambda=1e-6;
                obj.par.sub(l).mu0=zeros(obj.par.T,1);
            end
            calcss(obj,x);            
        end
        
        function cobj=copy(obj)
            cobj=gmmmodel();
            cobj.par=obj.par;
            cobj.ss=obj.ss;
            cobj.logPc=obj.logPc;
            cobj.logP=obj.logP;
            cobj.logZ=obj.logZ;
        end
        
        function remove_observation(obj,x,n)        
            if obj.par.z(n)~=0
                k=obj.par.z(n);
                for l=1:obj.par.nsubjects
                    xt=x{l}(:,n);
                    Xi=xt*xt';
                    obj.ss(l).xt=xt;
                    obj.ss(l).Xi=Xi;
                    obj.ss(l).R(:,:,k)=cholupdate(obj.ss(l).R(:,:,k),1/sqrt(obj.par.nk(k)+obj.par.sub(l).lambda)*(obj.ss(l).x_avg(:,k))+obj.par.sub(l).lambda*obj.par.sub(l).mu0,'+');
                    obj.ss(l).x_avg(:,k)=obj.ss(l).x_avg(:,k)-xt;
                    obj.ss(l).Sigma_avg(:,:,k)=obj.ss(l).Sigma_avg(:,:,k)-Xi;
                    obj.ss(l).R(:,:,k)=cholupdate(obj.ss(l).R(:,:,k),xt,'-');
                    obj.ss(l).R(:,:,k)=cholupdate(obj.ss(l).R(:,:,k),1/sqrt(obj.par.nk(k)-1+obj.par.sub(l).lambda)*(obj.ss(l).x_avg(:,k)+obj.par.sub(l).lambda*obj.par.sub(l).mu0),'-');
                    nn=obj.par.nk(k)-1+obj.par.sub(l).v0;
                    obj.logPc(k,l)=obj.ss(l).logPrior-(nn*sum(log(diag(obj.ss(l).R(:,:,k))))-nn*obj.par.T/2*log(2)-mvgammaln(obj.par.T,nn/2)-obj.par.T/2*log(obj.par.sub(l).lambda/(obj.par.nk(k)-1+obj.par.sub(l).lambda)));
                end
                obj.par.nk(k)=obj.par.nk(k)-1;
            end
        end
        
        function remove_empty_clusters(obj)
            
        end
        
        function add_observation(obj,n,k,~,comp,~)
            obj.par.nk(k)=obj.par.nk(k)+1;
            for l=1:obj.par.nsubjects
                obj.ss(l).Sigma_avg(:,:,k)=obj.ss(l).Sigma_avg(:,:,k)+obj.ss(l).Xi;
                obj.ss(l).x_avg(:,k)=obj.ss(l).x_avg(:,k)+obj.ss(l).xt;
                obj.ss(l).R(:,:,k)=obj.ss(l).Rt(:,:,k);
            end
        end
        
        function [categoricalDist,logPnew,logdiff,addss]=compute_categorical(obj,x,n,comp)
            addss=[];
            K=obj.par.K;
            logdet=zeros(1,K);
            logPnew=zeros(K,obj.par.nsubjects);
            for l=1:obj.par.nsubjects
                for k=1:K
                    obj.ss(l).Rt(:,:,k)=cholupdate(obj.ss(l).R(:,:,k),obj.ss(l).xt,'+');
                    obj.ss(l).Rt(:,:,k)=cholupdate(obj.ss(l).Rt(:,:,k),1/sqrt(obj.par.nk(k)+obj.par.sub(l).lambda)*(obj.ss(l).x_avg(:,k)+obj.par.sub(l).lambda*obj.par.sub(l).mu0),'+');
                    obj.ss(l).Rt(:,:,k)=cholupdate(obj.ss(l).Rt(:,:,k),1/sqrt(obj.par.nk(k)+1+obj.par.sub(l).lambda)*(obj.ss(l).x_avg(:,k)+obj.ss(l).xt+obj.par.sub(l).lambda*obj.par.sub(l).mu0),'-');
                    logdet(k)=2*sum(log(diag(obj.ss(l).Rt(:,:,k))));
                end
                nn=obj.par.nk+1+obj.par.sub(l).v0;
                logPnew(:,l)=obj.ss(l).logPrior-(nn/2.*logdet-nn*obj.par.T/2*log(2)-mvgammaln(obj.par.T,nn/2)-obj.par.T/2*log(obj.par.sub(l).lambda./(obj.par.sub(l).lambda+obj.par.nk+1)));
            end
            logdiff=sum(logPnew-obj.logPc,2);
            categoricalDist=(obj.par.nk+obj.par.alpha/K).*exp(logdiff-max(logdiff));
        end
        
        function updateLogP(obj,x)
            
        end
        
        function calcss(obj,x)
            z=obj.par.z;
            val=unique(z);
            val=setdiff(val,0);
            K=obj.par.K;
            obj.logPc=zeros(K,obj.par.nsubjects);
            obj.par.nk=zeros(1,K);
            for l=1:obj.par.nsubjects
                obj.ss(l).R0=chol(obj.par.sub(l).gamma*obj.par.sub(l).Sigma0);
                obj.ss(l).logPrior=obj.par.sub(l).v0*sum(log(diag(obj.ss(l).R0)))-obj.par.sub(l).v0*obj.par.T/2*log(2)-mvgammaln(obj.par.T,obj.par.sub(l).v0/2);
                obj.ss(l).Sigma_avg=zeros(obj.par.T,obj.par.T,obj.par.K);
                obj.ss(l).R=zeros(obj.par.T,obj.par.T,K);
                obj.ss(l).x_avg=zeros(obj.par.T,K);
                for k=1:length(val)
                    idx=(z==val(k));
                    obj.par.nk(val(k))=sum(idx);
                    obj.ss(l).Sigma_avg(:,:,val(k))=x{l}(:,idx)*x{l}(:,idx)';
                    obj.ss(l).x_avg(:,val(k))=sum(x{l}(:,idx),2);
                    obj.ss(l).R(:,:,val(k))=chol(obj.ss(l).Sigma_avg(:,:,val(k))+obj.par.sub(l).gamma*obj.par.sub(l).Sigma0+...
                        obj.par.sub(l).lambda*obj.par.sub(l).mu0*obj.par.sub(l).mu0'-1/(obj.par.nk(val(k))+obj.par.sub(l).lambda)*...
                        (obj.ss(l).x_avg(:,val(k))+obj.par.sub(l).lambda*obj.par.sub(l).mu0)*(obj.ss(l).x_avg(:,val(k))+...
                        obj.par.sub(l).lambda*obj.par.sub(l).mu0)');
                    logdet=2*sum(log(diag(obj.ss(l).R(:,:,val(k)))));
                    nn=(obj.par.nk(val(k))+obj.par.sub(l).v0);
                    obj.logPc(val(k),l)=obj.ss(l).logPrior-(nn/2*logdet-nn*obj.par.T/2*log(2)-mvgammaln(obj.par.T,nn/2)-obj.par.T/2*log(obj.par.sub(l).lambda/(obj.par.nk(val(k))+obj.par.sub(l).lambda)));
                end                
            end
            obj.updateLogZ();
        end
    end
end