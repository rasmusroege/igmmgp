classdef gmmsmodel < AbsGMMs_sh & AbsFiniteModel & handle
    properties
        ss;
        par;
        logPc;
        logP;
        logZ;
    end
    
    methods
        function obj=gmmsmodel(x,z,K)
            if nargin==0
                super_args={};
            else
                super_args={x,z,K};
            end
            obj=obj@AbsFiniteModel(super_args{:});
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
        
        function return_obj=copy(obj)
            return_obj=gmmsmodel();
            return_obj.par=obj.par;
            return_obj.ss=obj.ss;
            return_obj.logPc=obj.logPc;
            return_obj.logP=obj.logP;
            return_obj.logZ=obj.logZ;
        end
        
        function remove_empty_clusters(obj)
        end
        
        function remove_observation(obj,x,n)
            k=obj.par.z(n);
            obj.par.nk(k)=obj.par.nk(k)-1;
            for l=1:obj.par.nsubjects
                obj.ss(l).xt=x{l}(:,n);
                obj.ss(l).xt2=sum(x{l}(:,n).^2);
                obj.ss(l).x_avg(:,k)=obj.ss(l).x_avg(:,k)-obj.ss(l).xt;
                obj.ss(l).Sigma_avg(k)=obj.ss(l).Sigma_avg(k)-obj.ss(l).xt2;
                obj.ss(l).R2(k)=obj.ss(l).R0+1/2*obj.ss(l).Sigma_avg(k)-(1./(2*(obj.par.nk(k)+obj.par.sub(l).lambda)).*sum((obj.ss(l).x_avg(:,k)+obj.par.sub(l).lambda*obj.par.sub(l).mu0).^2));
                nn=obj.par.T*obj.par.nk(k)/2+obj.par.sub(l).v0;
                obj.logPc(k,l)=obj.ss(l).logPrior-((nn).*log(obj.ss(l).R2(k))-gammaln(nn)-obj.par.T/2*log(obj.par.sub(l).lambda./(obj.par.nk(k)+obj.par.sub(l).lambda)));
            end
            
        end
        
        function add_observation(obj,n,k,~,comp,~)
            obj.par.nk(k)=obj.par.nk(k)+1;
            for l=1:obj.par.nsubjects
                obj.ss(l).x_avg(:,k)=obj.ss(l).x_avg(:,k)+obj.ss(l).xt;
                obj.ss(l).Sigma_avg(k)=obj.ss(l).Sigma_avg(k)+obj.ss(l).xt2;
            end
            if ~isempty(comp)
                for l=1:obj.par.nsubjects
                    obj.ss(l).R2(k)=obj.ss(l).R2t(k==comp);
                end
            else
                for l=1:obj.par.nsubjects
                    obj.ss(l).R2(k)=obj.ss(l).R2t(k);
                end
            end
        end
        
        function [categoricalDist,logPnew,logdiff,addss]=compute_categorical(obj,x,n,comp)
            K=obj.par.K;
            logPnew = zeros(K,obj.par.nsubjects);
            for l=1:obj.par.nsubjects
                obj.ss(l).R2t=obj.ss(l).R0+1/2*(obj.ss(l).Sigma_avg+obj.ss(l).xt2)-(1./(2*(obj.par.nk+1+obj.par.sub(l).lambda))).*sum(bsxfun(@plus,obj.ss(l).x_avg,obj.par.sub(l).lambda*obj.par.sub(l).mu0+obj.ss(l).xt).^2,1)';
                nn=obj.par.T*(obj.par.nk+1)/2+obj.par.sub(l).v0;
                logPnew(:,l)=obj.ss(l).logPrior-((nn).*log(obj.ss(l).R2t)-gammaln(nn)-obj.par.T/2*log(obj.par.sub(l).lambda./(obj.par.nk+1+obj.par.sub(l).lambda)));
            end
            logdiff=sum(logPnew-obj.logPc,2);
            categoricalDist=(obj.par.nk+obj.par.alpha/obj.par.K).*exp(logdiff-max(logdiff));
            addss=[];
        end
        
        function calcss(obj,x)
            K=obj.par.K;
            z=obj.par.z;
            val=unique(z);
            val=setdiff(val,0);
            obj.logPc=zeros(K,obj.par.nsubjects);
            obj.logZ=0;
            obj.par.nk=zeros(K,1);
            for k=1:length(val)
                obj.par.nk(val(k))=sum(z==val(k));
            end
            
            for l=1:obj.par.nsubjects
                obj.ss(l).logPrior=obj.par.sub(l).v0.*log(obj.par.sub(l).gamma)-gammaln(obj.par.sub(l).v0);
                obj.ss(l).Sigma_avg=zeros(K,1);
                obj.ss(l).R2=zeros(K,1);
                obj.ss(l).x_avg=zeros(obj.par.T,K);
                obj.ss(l).R0=obj.par.sub(l).gamma+0.5*obj.par.sub(l).lambda*sum(obj.par.sub(l).mu0.^2);
                obj.ss(l).R2t=zeros(obj.par.K,1);
                for k=1:length(val)
                    idx=(z==val(k));
                    obj.ss(l).Sigma_avg(val(k))=sum(sum(x{l}(:,idx).^2,2));
                    obj.ss(l).x_avg(:,val(k))=sum(x{l}(:,idx),2);
                end
                obj.ss(l).R2=obj.ss(l).R0+1/2*obj.ss(l).Sigma_avg-(1./(2*(obj.par.nk+obj.par.sub(l).lambda))).*sum(bsxfun(@plus,obj.ss(l).x_avg,obj.par.sub(l).lambda*obj.par.sub(l).mu0).^2,1)';
                nn=obj.par.T*obj.par.nk/2+obj.par.sub(l).v0;
                obj.logPc(:,l)=obj.ss(l).logPrior-((nn).*log(obj.ss(l).R2)-gammaln(nn)-obj.par.T/2*log(obj.par.sub(l).lambda./(obj.par.nk+obj.par.sub(l).lambda)));
            end
            obj.updateLogZ();
        end
        
        function updateLogP(obj,x)
        end
    end
end
