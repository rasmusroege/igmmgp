classdef igmmmodel < AbsGMM & AbsInfiniteModel & handle
    properties
        ss;
        par;
        logPc;
        logP;
        logZ;
    end
    
    methods
        function obj=igmmmodel(x,z)
            if nargin==0
                super_args={};
            else
                super_args={x,z};
            end
            obj=obj@AbsInfiniteModel(super_args{:});
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
            cobj=igmmmodel();
            cobj.par=obj.par;
            cobj.ss=obj.ss;
            cobj.logPc=obj.logPc;
            cobj.logP=obj.logP;
            cobj.logZ=obj.logZ;
        end
        
        function merge_obj=initMerge(obj,x,z,comp)
            merge_obj=copy(obj);
            merge_obj.par.z=z;
            for l=1:merge_obj.par.nsubjects
                merge_obj.ss(l).xt=[];
                merge_obj.ss(l).xt2=[];
                for k=comp(1)'
                    idx=(z==k);
                    merge_obj.par.nk(k)=sum(idx);
                    merge_obj.ss(l).Sigma_avg(:,:,k)=x{l}(:,idx)*x{l}(:,idx)';
                    merge_obj.ss(l).x_avg(:,k)=sum(x{l}(:,idx),2);
                    merge_obj.ss(l).R(:,:,k)=chol(merge_obj.ss(l).Sigma_avg(:,:,k)+merge_obj.par.sub(l).gamma*merge_obj.par.sub(l).Sigma0+...
                        merge_obj.par.sub(l).lambda*merge_obj.par.sub(l).mu0*merge_obj.par.sub(l).mu0'-1/(merge_obj.par.nk(k)+merge_obj.par.sub(l).lambda)*...
                        (merge_obj.ss(l).x_avg(:,k)+merge_obj.par.sub(l).lambda*merge_obj.par.sub(l).mu0)*(merge_obj.ss(l).x_avg(:,k)+...
                        merge_obj.par.sub(l).lambda*merge_obj.par.sub(l).mu0)');
                    logdet=2*sum(log(diag(merge_obj.ss(l).R(:,:,k))));
                    nn=(merge_obj.par.nk(k)+merge_obj.par.sub(l).v0);
                    merge_obj.logPc(k,l)=merge_obj.ss(l).logPrior-(nn/2*logdet-nn*merge_obj.par.T/2*log(2)-mvgammaln(merge_obj.par.T,nn/2)-merge_obj.par.T/2*log(merge_obj.par.sub(l).lambda/(merge_obj.par.nk(k)+merge_obj.par.sub(l).lambda)));
                end
                merge_obj.ss(l).Sigma_avg(:,:,comp(2))=[];
                merge_obj.ss(l).x_avg(:,comp(2))=[];
                merge_obj.ss(l).R(:,:,comp(2))=[];
            end
            merge_obj.par.nk(comp(2))=[];
            merge_obj.logPc(comp(2),:)=[];
            merge_obj.updateLogZ();
        end
        
        function launch_obj=initLaunch(obj,x,z,comp)
            launch_obj=copy(obj);
            launch_obj.par.z=z;
            launch_obj.par.nk(comp(1),1)=sum(z==comp(1));
            launch_obj.par.nk(comp(2),1)=sum(z==comp(2));
            for l=1:obj.par.nsubjects
                launch_obj.ss(l).xt=[];
                launch_obj.ss(l).xt2=[];
                
                for k=comp(:)'
                    idx=(z==k);
                    launch_obj.ss(l).Sigma_avg(:,:,k)=x{l}(:,idx)*x{l}(:,idx)';
                    launch_obj.ss(l).x_avg(:,k)=sum(x{l}(:,idx),2);
                    launch_obj.ss(l).R(:,:,k)=chol(launch_obj.ss(l).Sigma_avg(:,:,k)+launch_obj.par.sub(l).gamma*launch_obj.par.sub(l).Sigma0+...
                        launch_obj.par.sub(l).lambda*launch_obj.par.sub(l).mu0*launch_obj.par.sub(l).mu0'-1/(launch_obj.par.nk(k)+launch_obj.par.sub(l).lambda)*...
                        (launch_obj.ss(l).x_avg(:,k)+launch_obj.par.sub(l).lambda*launch_obj.par.sub(l).mu0)*(launch_obj.ss(l).x_avg(:,k)+...
                        launch_obj.par.sub(l).lambda*launch_obj.par.sub(l).mu0)');
                    logdet=2*sum(log(diag(launch_obj.ss(l).R(:,:,k))));
                    nn=(launch_obj.par.nk(k)+launch_obj.par.sub(l).v0);
                    launch_obj.logPc(k,l)=launch_obj.ss(l).logPrior-(nn/2*logdet-nn*launch_obj.par.T/2*log(2)-mvgammaln(launch_obj.par.T,nn/2)-launch_obj.par.T/2*log(launch_obj.par.sub(l).lambda/(launch_obj.par.nk(k)+launch_obj.par.sub(l).lambda)));                
                end
            end
            launch_obj.updateLogZ();
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
            idx_empty=find(obj.par.nk==0);
            if ~isempty(idx_empty)
                obj.par.nk(idx_empty)=[];
                obj.par.z(obj.par.z>idx_empty)=obj.par.z(obj.par.z>idx_empty)-1;
                obj.logPc(idx_empty,:)=[];
                for l=1:obj.par.nsubjects
                    obj.ss(l).Sigma_avg(:,:,idx_empty)=[];
                    obj.ss(l).x_avg(:,idx_empty)=[];
                    obj.ss(l).R(:,:,idx_empty)=[];
                end
            end
        end
        
        function add_observation(obj,n,k,~,comp,~)
            if k<=length(obj.par.nk)
                obj.par.nk(k)=obj.par.nk(k)+1;
                for l=1:obj.par.nsubjects
                    obj.ss(l).Sigma_avg(:,:,k)=obj.ss(l).Sigma_avg(:,:,k)+obj.ss(l).Xi;
                    obj.ss(l).x_avg(:,k)=obj.ss(l).x_avg(:,k)+obj.ss(l).xt;
                    obj.ss(l).R(:,:,k)=obj.ss(l).Rt(:,:,k);
                end
            else
                obj.par.nk(k)=1;
                for l=1:obj.par.nsubjects
                    obj.ss(l).Sigma_avg(:,:,k)=obj.ss(l).Xi;
                    obj.ss(l).x_avg(:,k)=obj.ss(l).xt;
                    obj.ss(l).R(:,:,k)=obj.ss(l).Rt(:,:,k);
                end
            end
        end
        
        function [categoricalDist,logPnew,logdiff,addss]=compute_categorical(obj,x,n,comp)
            addss=[];
            K=max(obj.par.z);
            if isempty(comp)
                logdet=zeros(K+1,1);
                logPnew=zeros(K+1,obj.par.nsubjects);
                for l=1:obj.par.nsubjects
                    for k=1:K
                        obj.ss(l).Rt(:,:,k)=cholupdate(obj.ss(l).R(:,:,k),obj.ss(l).xt,'+');
                        obj.ss(l).Rt(:,:,k)=cholupdate(obj.ss(l).Rt(:,:,k),1/sqrt(obj.par.nk(k)+obj.par.sub(l).lambda)*(obj.ss(l).x_avg(:,k)+obj.par.sub(l).lambda*obj.par.sub(l).mu0),'+');
                        obj.ss(l).Rt(:,:,k)=cholupdate(obj.ss(l).Rt(:,:,k),1/sqrt(obj.par.nk(k)+1+obj.par.sub(l).lambda)*(obj.ss(l).x_avg(:,k)+obj.ss(l).xt+obj.par.sub(l).lambda*obj.par.sub(l).mu0),'-');
                        logdet(k)=2*sum(log(diag(obj.ss(l).Rt(:,:,k))));
                    end
                    obj.ss(l).Rt(:,:,K+1)=cholupdate(obj.ss(l).R0,obj.ss(l).xt,'+');
                    obj.ss(l).Rt(:,:,K+1)=cholupdate(obj.ss(l).Rt(:,:,K+1),1/sqrt(1+obj.par.sub(l).lambda)*(obj.ss(l).xt+obj.par.sub(l).lambda*obj.par.sub(l).mu0),'-');
                    logdet(K+1)=2*sum(log(diag(obj.ss(l).Rt(:,:,K+1))));
                    nn=[obj.par.nk+1;1]+obj.par.sub(l).v0;
                    logPnew(:,l)=obj.ss(l).logPrior-(nn/2.*logdet-nn*obj.par.T/2*log(2)-mvgammaln(obj.par.T,nn/2)-obj.par.T/2*log(obj.par.sub(l).lambda./(obj.par.sub(l).lambda+[obj.par.nk+1;1])));
                end
                logdiff=sum(logPnew-[obj.logPc;zeros(1,obj.par.nsubjects)],2);
                categoricalDist=[obj.par.nk;obj.par.alpha].*exp(logdiff-max(logdiff));
            else
                logdet=zeros(2,1);
                logPnew=zeros(2,obj.par.nsubjects);
                for l=1:obj.par.nsubjects
                    obj.ss(l).xt=x{l}(:,n);
                    obj.ss(l).Xi=obj.ss(l).xt*obj.ss(l).xt';
                    for k=comp
                        obj.ss(l).Rt(:,:,k)=cholupdate(obj.ss(l).R(:,:,k),obj.ss(l).xt,'+');
                        obj.ss(l).Rt(:,:,k)=cholupdate(obj.ss(l).Rt(:,:,k),1/sqrt(obj.par.nk(k)+obj.par.sub(l).lambda)*(obj.ss(l).x_avg(:,k)+obj.par.sub(l).lambda*obj.par.sub(l).mu0),'+');
                        obj.ss(l).Rt(:,:,k)=cholupdate(obj.ss(l).Rt(:,:,k),1/sqrt(obj.par.nk(k)+1+obj.par.sub(l).lambda)*(obj.ss(l).x_avg(:,k)+obj.ss(l).xt+obj.par.sub(l).lambda*obj.par.sub(l).mu0),'-');
                        logdet(k)=2*sum(log(diag(obj.ss(l).Rt(:,:,k))));
                    end
                    nn=obj.par.nk(comp)+1+obj.par.sub(l).v0;
                    logPnew(:,l)=obj.ss(l).logPrior-(nn/2.*logdet(comp)-nn*obj.par.T/2*log(2)-mvgammaln(obj.par.T,nn/2)-obj.par.T/2*log(obj.par.sub(l).lambda./(obj.par.sub(l).lambda+obj.par.nk(comp)+1)));
                end
                logdiff=sum(logPnew-obj.logPc(comp,:),2);
                categoricalDist=obj.par.nk(comp).*exp(logdiff-max(logdiff));
            end
        end
        
        function updateLogP(obj,x)
            
        end
        
        function calcss(obj,x)
            z=obj.par.z;
            val=unique(z);
            val=setdiff(val,0);
            K=max(val);
            obj.logPc=zeros(K,obj.par.nsubjects);
            obj.par.nk=zeros(K,1);
            for l=1:obj.par.nsubjects
                obj.ss(l).R0=chol(obj.par.sub(l).gamma*obj.par.sub(l).Sigma0);
                obj.ss(l).logPrior=obj.par.sub(l).v0*sum(log(diag(obj.ss(l).R0)))-obj.par.sub(l).v0*obj.par.T/2*log(2)-mvgammaln(obj.par.T,obj.par.sub(l).v0/2);
                obj.ss(l).Sigma_avg=zeros(obj.par.T,obj.par.T,K);
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