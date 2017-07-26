classdef igmmddmodel<AbsGMMdd_sh & AbsInfiniteModel & handle
    % igmmddmodel
    properties
        ss;
        par;
        logPc;
        logP;
        logZ;
    end
    
    methods
        function obj=igmmddmodel(x,z,par)
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
        
        function return_obj=copy(obj)
            return_obj=igmmddmodel();
            return_obj.par=obj.par;
            return_obj.ss=obj.ss;
            return_obj.logPc=obj.logPc;
            return_obj.logP=obj.logP;
            return_obj.logZ=obj.logZ;
        end

        
        function merge_obj=initMerge(obj,X,z,comp)
            merge_obj=copy(obj);
            merge_obj.par.z=z;
            merge_obj.par.nk(comp(1))=obj.par.nk(comp(1))+obj.par.nk(comp(2));
            merge_obj.par.nk(comp(2))=[];            
            for l=1:obj.par.nsubjects
                merge_obj.ss(l).xt=[];
                merge_obj.ss(l).xt2=[];
                merge_obj.ss(l).Sigma_avg(:,comp(1))=obj.ss(l).Sigma_avg(:,comp(1))+obj.ss(l).Sigma_avg(:,comp(2));
                merge_obj.ss(l).Sigma_avg(:,comp(2))=[];
                
                merge_obj.ss(l).x_avg(:,comp(1))=obj.ss(l).x_avg(:,comp(1))+obj.ss(l).x_avg(:,comp(2));
                merge_obj.ss(l).x_avg(:,comp(2))=[];
                
                merge_obj.ss(l).R2(:,comp(1))=merge_obj.ss(l).R0+0.5*merge_obj.ss(l).Sigma_avg(:,comp(1))-...
                    0.5/(merge_obj.par.nk(comp(1))+merge_obj.par.sub(l).lambda)*(merge_obj.ss(l).x_avg(:,comp(1))+...
                    merge_obj.par.sub(l).lambda*merge_obj.par.sub(l).mu0).^2;
%                 obj.ss(l).R2(:,val(k))=obj.ss(l).R0+0.5*obj.ss(l).Sigma_avg(:,val(k))-0.5/(obj.par.nk(val(k))+obj.par.sub(l).lambda)*(obj.ss(l).x_avg(:,val(k))+obj.par.sub(l).lambda*obj.par.sub(l).mu0).^2;
                merge_obj.ss(l).R2(:,comp(2))=[];
                
                nn=bsxfun(@plus,(merge_obj.par.nk(comp(1)))/2,merge_obj.par.sub(l).v0);
                merge_obj.logPc(comp(1),l)=obj.ss(l).logPrior-(sum(nn.*log(merge_obj.ss(l).R2(:,comp(1))))-...
                    obj.par.T*sum(gammaln(nn))-obj.par.T/2*log(merge_obj.par.sub(l).lambda/(merge_obj.par.nk(comp(1))+...
                    merge_obj.par.sub(l).lambda)));
            end

            merge_obj.logPc(comp(2),:)=[];
            merge_obj.updateLogZ();
        end
        
        function launch_obj=initLaunch(obj,X,z,comp)
            launch_obj=copy(obj);
            launch_obj.par.z=z;
            launch_obj.par.nk(comp(1),1)=sum(z==comp(1));
            launch_obj.par.nk(comp(2),1)=sum(z==comp(2));
            for l=1:obj.par.nsubjects
                launch_obj.ss(l).xt=[];
                launch_obj.ss(l).xt2=[];
                
                launch_obj.ss(l).Sigma_avg(:,comp(1))=sum(X{l}(:,z==comp(1)).^2,2);
                launch_obj.ss(l).Sigma_avg(:,comp(2))=sum(X{l}(:,z==comp(2)).^2,2);
                
                launch_obj.ss(l).x_avg(:,comp(1))=sum(X{l}(:,z==comp(1)),2);
                launch_obj.ss(l).x_avg(:,comp(2))=sum(X{l}(:,z==comp(2)),2);
                
                launch_obj.ss(l).R2(:,comp(1))=launch_obj.ss(l).R0+0.5*launch_obj.ss(l).Sigma_avg(:,comp(1))-0.5/(launch_obj.par.nk(comp(1))+launch_obj.par.sub(l).lambda)*(launch_obj.ss(l).x_avg(:,comp(1))+launch_obj.par.sub(l).lambda*launch_obj.par.sub(l).mu0).^2;
                launch_obj.ss(l).R2(:,comp(2))=launch_obj.ss(l).R0+0.5*launch_obj.ss(l).Sigma_avg(:,comp(2))-0.5/(launch_obj.par.nk(comp(2))+launch_obj.par.sub(l).lambda)*(launch_obj.ss(l).x_avg(:,comp(2))+launch_obj.par.sub(l).lambda*launch_obj.par.sub(l).mu0).^2;
                
                nn=bsxfun(@plus,launch_obj.par.nk(comp)'/2,launch_obj.par.sub(l).v0);
                launch_obj.logPc(comp(1),l)=launch_obj.ss(l).logPrior-(sum(nn(:,1).*log(launch_obj.ss(l).R2(:,comp(1))))-obj.par.T*sum(gammaln(nn(:,1)))-obj.par.T/2*log(launch_obj.par.sub(l).lambda/(launch_obj.par.nk(comp(1))+launch_obj.par.sub(l).lambda)));
                launch_obj.logPc(comp(2),l)=launch_obj.ss(l).logPrior-(sum(nn(:,2).*log(launch_obj.ss(l).R2(:,comp(2))))-obj.par.T*sum(gammaln(nn(:,2)))-obj.par.T/2*log(launch_obj.par.sub(l).lambda/(launch_obj.par.nk(comp(2))+launch_obj.par.sub(l).lambda)));
            end
            
            launch_obj.updateLogZ();
        end
        
        function remove_empty_clusters(obj)
            idx_empty=find(obj.par.nk==0);
            if ~isempty(idx_empty)
                obj.par.nk(idx_empty)=[];
                obj.par.z(obj.par.z>idx_empty)=obj.par.z(obj.par.z>idx_empty)-1;
                obj.logPc(idx_empty,:)=[];
                for l=1:obj.par.nsubjects
                    obj.ss(l).Sigma_avg(:,idx_empty)=[];
                    obj.ss(l).x_avg(:,idx_empty)=[];
                    obj.ss(l).R2(:,idx_empty)=[];
                end
            end            
        end
        
        function remove_observation(obj,x,n)

            k=obj.par.z(n);
            for l=1:obj.par.nsubjects
                lam=obj.par.sub(l).lambda;
                mu0=obj.par.sub(l).mu0;
                nk=obj.par.nk(k);
                
                obj.ss(l).R2(:,k)=obj.ss(l).R2(:,k)+0.5/(nk+lam)*(obj.ss(l).x_avg(:,k)+lam*mu0).^2;
                nk=nk-1;
                obj.ss(l).xt=x{l}(:,n);
                obj.ss(l).xt2=x{l}(:,n).^2;
                obj.ss(l).x_avg(:,k)=obj.ss(l).x_avg(:,k)-obj.ss(l).xt;
                obj.ss(l).Sigma_avg(:,k)=obj.ss(l).Sigma_avg(:,k)-obj.ss(l).xt2;
                obj.ss(l).R2(:,k)=obj.ss(l).R2(:,k)-0.5*obj.ss(l).xt2-0.5/(nk+lam)*...
                    (obj.ss(l).x_avg(:,k)+lam*mu0).^2;
                nn=nk/2+obj.par.sub(l).v0;
                obj.logPc(k,l)=obj.ss(l).logPrior-(sum(nn.*log(obj.ss(l).R2(:,k)))-...
                obj.par.T*sum(gammaln(nn))-obj.par.T/2*log(lam/(nk+lam)));
            end
            obj.par.nk(k)=nk;
        end
        
        function add_observation(obj,n,k,~,comp,~)
            if k<=length(obj.par.nk)
                obj.par.nk(k,1)=obj.par.nk(k)+1;
                for l=1:obj.par.nsubjects
                    obj.ss(l).x_avg(:,k)=obj.ss(l).x_avg(:,k)+obj.ss(l).xt;
                    obj.ss(l).Sigma_avg(:,k)=obj.ss(l).Sigma_avg(:,k)+obj.ss(l).xt2;
                end
            else
                obj.par.nk(k,1)=1;
                for l=1:obj.par.nsubjects
                    obj.ss(l).x_avg(:,k)=obj.ss(l).xt;
                    obj.ss(l).Sigma_avg(:,k)=obj.ss(l).xt2;
                end                
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
            if isempty(comp)
                K=max(obj.par.z);
                logPnew = zeros(K+1,obj.par.nsubjects);
                if ~isfield(obj.ss(1),'R2t') || size(obj.ss(1).R2t,2)~=K
                    for l=1:obj.par.nsubjects
                        obj.ss(l).R2t=zeros(size(obj.ss(l).R2)+[0 1]);
                    end
                end
                for l=1:obj.par.nsubjects
                    s=obj.ss(l);
                    sav=[obj.ss(l).Sigma_avg  zeros(obj.par.T,1)];
                    xav=[obj.ss(l).x_avg zeros(obj.par.T,1)];
                    r0=obj.ss(l).R0;
                    nk=[obj.par.nk;0]';
                    lam=obj.par.sub(l).lambda;
                    mu0=obj.par.sub(l).mu0;
                    obj.ss(l).R2t=bsxfun(@plus,r0+1/2*s.xt2,1/2*sav)+bsxfun(@times,-0.5./(nk+1+lam),bsxfun(@plus,xav,s.xt+lam*mu0).^2);
                    nn=([obj.par.nk;0]'+1)/2+obj.par.sub(l).v0;
                    logPnew(:,l)=obj.ss(l).logPrior-(sum(bsxfun(@times,nn,log(obj.ss(l).R2t)))-obj.par.T*gammaln(nn)-obj.par.T/2*log(lam./(([obj.par.nk'+1 1])+lam)));
                end
                logdiff=sum(logPnew-[obj.logPc;zeros(1,obj.par.nsubjects)],2);
                categoricalDist=[obj.par.nk;obj.par.alpha].*exp(logdiff-max(logdiff));
            else
                K=max(obj.par.z);
                logPnew = zeros(2,obj.par.nsubjects);
                if ~isfield(obj.ss(1),'R2t') || size(obj.ss(1).R2t,2)~=2
                    for l=1:obj.par.nsubjects
                        obj.ss(l).R2t=zeros(size(obj.ss(l).R2,1),2);
                    end
                end

                for l=1:obj.par.nsubjects
                    obj.ss(l).xt=x{l}(:,n);
                    obj.ss(l).xt2=x{l}(:,n).^2;                    
                    s=obj.ss(l);
                    lam=obj.par.sub(l).lambda;
                    mu0=obj.par.sub(l).mu0;

                    for i=1:2
                        k=comp(i);
                        obj.ss(l).R2t(:,i)=obj.ss(l).R0+0.5*obj.ss(l).Sigma_avg(:,k)+0.5*s.xt2-0.5/(obj.par.nk(k)+1+obj.par.sub(l).lambda)*(obj.ss(l).x_avg(:,k)+s.xt+obj.par.sub(l).lambda*obj.par.sub(l).mu0).^2;
                    end
                    nn=(obj.par.nk(comp)'+1)/2+obj.par.sub(l).v0;
                    logPnew(:,l)=obj.ss(l).logPrior-(sum(bsxfun(@times,nn,log(obj.ss(l).R2t)))-obj.par.T*gammaln(nn)-obj.par.T/2*log(lam./((obj.par.nk(comp)'+1)+lam)));
                end
                logdiff=sum(logPnew-obj.logPc(comp,:),2);
                categoricalDist=obj.par.nk(comp).*exp(logdiff-max(logdiff));
            end
            addss=[];
        end
        
        function updateLogP(obj,x)
            
        end
        
        function calcss(obj,x)
            z=obj.par.z;
            
            val=unique(z);
            val=setdiff(val,0);
            K=length(val);
            obj.logPc=zeros(K,obj.par.nsubjects);
            obj.logZ=0;
            obj.par.nk=zeros(length(val),1);
            for k=1:K
                obj.par.nk(k)=sum(z==k);
            end
            for l=1:obj.par.nsubjects
                obj.ss(l).logPrior=obj.par.T*obj.par.sub(l).v0.*log(obj.par.sub(l).gamma)-obj.par.T*gammaln(obj.par.sub(l).v0);
                obj.ss(l).Sigma_avg=zeros(obj.par.T,K);
                obj.ss(l).R2=zeros(obj.par.T,K);
                obj.ss(l).x_avg=zeros(obj.par.T,K);
                for k=1:length(val)
                    idx=(z==val(k));
                    obj.ss(l).Sigma_avg(:,val(k))=sum(x{l}(:,idx).^2,2);
                    obj.ss(l).x_avg(:,val(k))=sum(x{l}(:,idx),2);
                    obj.ss(l).R0=obj.par.sub(l).gamma+0.5*obj.par.sub(l).lambda*obj.par.sub(l).mu0.^2;
                    obj.ss(l).R2(:,val(k))=obj.ss(l).R0+0.5*obj.ss(l).Sigma_avg(:,val(k))-0.5/(obj.par.nk(val(k))+obj.par.sub(l).lambda)*(obj.ss(l).x_avg(:,val(k))+obj.par.sub(l).lambda*obj.par.sub(l).mu0).^2;
                    nn=bsxfun(@plus,obj.par.nk(val(k))/2,obj.par.sub(l).v0);
                    obj.logPc(val(k),l)=obj.ss(l).logPrior-(sum(nn.*log(obj.ss(l).R2(:,val(k))))-obj.par.T*gammaln(nn)-obj.par.T/2*log(obj.par.sub(l).lambda/(obj.par.nk(val(k))+obj.par.sub(l).lambda)));
                end
                obj.logP(l)=0;
            end
            obj.updateLogZ();
        end
    end
end