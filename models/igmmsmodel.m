classdef igmmsmodel < AbsGMMs_sh & AbsInfiniteModel & handle
    properties
        par;
        ss;
        logPc;
        logP;
        logZ;
    end
    
    methods
        function obj=igmmsmodel(x,z)
            if nargin==0
                super_args={};
            else
                super_args={x,z};
            end
            obj@AbsInfiniteModel(super_args{:});
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
        
        function mo=initMerge(obj,X,z,comp)
            mo=copy(obj);
            mo.par.z=z;
            mo.par.nk(comp(1))=sum(mo.par.nk(comp));
            mo.par.nk(comp(2))=[];            
            for l=1:obj.par.nsubjects
                mo.ss(l).xt=[];
                mo.ss(l).xt2=[];
                mo.ss(l).Sigma_avg(comp(1))=sum(obj.ss(l).Sigma_avg(comp));
                mo.ss(l).Sigma_avg(comp(2))=[];
                
                mo.ss(l).x_avg(:,comp(1))=obj.ss(l).x_avg(:,comp(1))+obj.ss(l).x_avg(:,comp(2));
                mo.ss(l).x_avg(:,comp(2))=[];
                
                mo.ss(l).R2(comp(1))=mo.ss(l).R0+0.5*mo.ss(l).Sigma_avg(comp(1))-0.5/(mo.par.nk(comp(1))+mo.par.sub(l).lambda)*sum((mo.ss(l).x_avg(:,comp(1))+mo.par.sub(l).lambda*mo.par.sub(l).mu0).^2);
                mo.ss(l).R2(comp(2))=[];
                
                nn=obj.par.T*mo.par.nk(comp(1))/2+mo.par.sub(l).v0;
                mo.logPc(comp(1),l)=mo.ss(l).logPrior-(sum(nn(1).*log(mo.ss(l).R2(comp(1))))-sum(gammaln(nn(1)))-mo.par.T/2*log(mo.par.sub(l).lambda/(mo.par.nk(comp(1))+mo.par.sub(l).lambda)));
            end
            mo.logPc(comp(2),:)=[];
            mo.updateLogZ();
        end
        
        function launch_obj=initLaunch(obj,X,z,comp)
            launch_obj=copy(obj);
            launch_obj.par.z=z;
            launch_obj.par.nk(comp(1),1)=sum(z==comp(1));
            launch_obj.par.nk(comp(2),1)=sum(z==comp(2));
            for l=1:obj.par.nsubjects
                launch_obj.ss(l).xt=[];
                launch_obj.ss(l).xt2=[];
                
                launch_obj.ss(l).Sigma_avg(comp(1),1)=sum(sum(X{l}(:,z==comp(1)).^2,2));
                launch_obj.ss(l).Sigma_avg(comp(2),1)=sum(sum(X{l}(:,z==comp(2)).^2,2));
                
                launch_obj.ss(l).x_avg(:,comp(1))=sum(X{l}(:,z==comp(1)),2);
                launch_obj.ss(l).x_avg(:,comp(2))=sum(X{l}(:,z==comp(2)),2);
                
                launch_obj.ss(l).R2(comp(1),1)=launch_obj.ss(l).R0+0.5*launch_obj.ss(l).Sigma_avg(comp(1))...
                    -0.5/(launch_obj.par.nk(comp(1))+launch_obj.par.sub(l).lambda).*sum(bsxfun(@plus,launch_obj.ss(l).x_avg(:,comp(1)),launch_obj.par.sub(l).lambda*launch_obj.par.sub(l).mu0).^2)';
                launch_obj.ss(l).R2(comp(2),1)=launch_obj.ss(l).R0+0.5*launch_obj.ss(l).Sigma_avg(comp(2))...
                    -0.5/(launch_obj.par.nk(comp(2))+launch_obj.par.sub(l).lambda).*sum(bsxfun(@plus,launch_obj.ss(l).x_avg(:,comp(2)),launch_obj.par.sub(l).lambda*launch_obj.par.sub(l).mu0).^2)';
                %launch_obj.ss(l).R2(comp(2))=launch_obj.ss(l).R0+0.5*launch_obj.ss(l).Sigma_avg(:,comp(2))-0.5/(launch_obj.par.nk(comp(2))+launch_obj.par.sub(l).lambda)*(launch_obj.ss(l).x_avg(:,comp(2))+launch_obj.par.sub(l).lambda*launch_obj.par.sub(l).mu0).^2;
                
                nn=launch_obj.par.T*launch_obj.par.nk(comp)/2+launch_obj.par.sub(l).v0;
                launch_obj.logPc(comp(1),l)=launch_obj.ss(l).logPrior-(sum(nn(1).*log(launch_obj.ss(l).R2(comp(1))))-sum(gammaln(nn(1)))-obj.par.T/2*log(launch_obj.par.sub(l).lambda/(launch_obj.par.nk(comp(1))+launch_obj.par.sub(l).lambda)));
                launch_obj.logPc(comp(2),l)=launch_obj.ss(l).logPrior-(sum(nn(2).*log(launch_obj.ss(l).R2(comp(2))))-sum(gammaln(nn(2)))-obj.par.T/2*log(launch_obj.par.sub(l).lambda/(launch_obj.par.nk(comp(2))+launch_obj.par.sub(l).lambda)));
            end
            launch_obj.updateLogZ();
        end
        
        function return_obj=copy(obj)
            return_obj=igmmsmodel();
            return_obj.par=obj.par;
            return_obj.ss=obj.ss;
            return_obj.logPc=obj.logPc;
            return_obj.logP=obj.logP;
            return_obj.logZ=obj.logZ;
        end
        
        function remove_empty_clusters(obj)
            idx_empty=find(obj.par.nk==0);
            if ~isempty(idx_empty)
                obj.par.nk(idx_empty)=[];
                obj.par.z(obj.par.z>idx_empty)=obj.par.z(obj.par.z>idx_empty)-1;
                obj.logPc(idx_empty,:)=[];
                for l=1:obj.par.nsubjects
                    obj.ss(l).Sigma_avg(idx_empty)=[];
                    obj.ss(l).x_avg(:,idx_empty)=[];
                    obj.ss(l).R2(idx_empty)=[];
                end
            end
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
            if length(obj.par.nk)>=k
                obj.par.nk(k,1)=obj.par.nk(k)+1;
                for l=1:obj.par.nsubjects
                    obj.ss(l).x_avg(:,k)=obj.ss(l).x_avg(:,k)+obj.ss(l).xt;
                    obj.ss(l).Sigma_avg(k,1)=obj.ss(l).Sigma_avg(k)+obj.ss(l).xt2;
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
            else
                obj.par.nk(k,1)=1;
                for l=1:obj.par.nsubjects
                    obj.ss(l).x_avg(:,k)=obj.ss(l).xt;
                    obj.ss(l).Sigma_avg(k,1)=obj.ss(l).xt2;
                end
                if ~isempty(comp)
                    for l=1:obj.par.nsubjects
                        obj.ss(l).R2(k,1)=obj.ss(l).R2t(k==comp);
                    end
                else
                    for l=1:obj.par.nsubjects
                        obj.ss(l).R2(k,1)=obj.ss(l).R2t(k);
                    end
                end
            end
        end
        
        function [categoricalDist,logPnew,logdiff,addss]=compute_categorical(obj,x,n,comp)
            if isempty(comp)
                K=length(obj.par.nk);
                logPnew = zeros(K+1,obj.par.nsubjects);
                for l=1:obj.par.nsubjects
                    obj.ss(l).R2t=obj.ss(l).R0+1/2*([obj.ss(l).Sigma_avg;0]+obj.ss(l).xt2)-...
                        (1./(2*([obj.par.nk;0]+1+obj.par.sub(l).lambda))).*sum(bsxfun(@plus,[obj.ss(l).x_avg zeros(obj.par.T,1)],obj.par.sub(l).lambda*obj.par.sub(l).mu0+obj.ss(l).xt).^2,1)';
                    nn=obj.par.T*([obj.par.nk;0]+1)/2+obj.par.sub(l).v0;
                    logPnew(:,l)=obj.ss(l).logPrior-((nn).*log(obj.ss(l).R2t)...
                        -gammaln(nn)-obj.par.T/2*log(obj.par.sub(l).lambda./([obj.par.nk;0]+1+obj.par.sub(l).lambda)));
                end
                logdiff=sum(logPnew-[obj.logPc;zeros(1,obj.par.nsubjects)],2);
                categoricalDist=[obj.par.nk;obj.par.alpha].*exp(logdiff-max(logdiff));
                addss=[];
            else
                logPnew=zeros(2,obj.par.nsubjects);
                for l=1:obj.par.nsubjects
                    obj.ss(l).xt2=sum(x{l}(:,n).^2);
                    obj.ss(l).xt=x{l}(:,n);
                    obj.ss(l).R2t=obj.ss(l).R0+1/2*(obj.ss(l).Sigma_avg(comp)+obj.ss(l).xt2)-(1./(2*(obj.par.nk(comp)+1+obj.par.sub(l).lambda))).*sum(bsxfun(@plus,obj.ss(l).x_avg(:,comp),obj.par.sub(l).lambda*obj.par.sub(l).mu0+obj.ss(l).xt).^2,1)';
                    nn=obj.par.T*(obj.par.nk(comp)+1)/2+obj.par.sub(l).v0;
                    logPnew(:,l)=obj.ss(l).logPrior-((nn).*log(obj.ss(l).R2t)...
                        -gammaln(nn)-obj.par.T/2*log(obj.par.sub(l).lambda./(obj.par.nk(comp)+1+obj.par.sub(l).lambda)));
                end
                logdiff=sum(logPnew-obj.logPc(comp,:),2);
                categoricalDist=obj.par.nk(comp).*exp(logdiff-max(logdiff));
                addss=[];
            end
        end
        
        function calcss(obj,x)
            z=obj.par.z;
            val=unique(z);
            val=setdiff(val,0);
            K=max(val);
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
                obj.ss(l).R2t=zeros(K,1);
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


