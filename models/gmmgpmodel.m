classdef gmmgpmodel < AbsGmmgp & AbsFiniteModel & handle
    % igmmgp_model
    properties
        par;
        ss;% D,V,ID, XtVs, Vtxcs, sxcvids, ids, logids, nk
        logPc=0; % nsubjects x K
        logP=0; % nsubjects x 1
        logZ=0; % 1 x 1
    end
    
    methods
        function obj=gmmgpmodel(x,z,K,swsampler)
            % constructor obj=igmmgp_model(X,z)
            if nargin==0
                super_args={};
            else
                super_args={x,z,K};
            end
            obj@AbsFiniteModel(super_args{:});
            obj.par.hypersamplers=[obj.par.hypersamplers {'sample_beta','sample_sig2_w2'}];
            if nargin==0
                return;
            end
            for l=1:obj.par.nsubjects
                tmp=var(x{l})';
                obj.par.sub(l).sig2=max(tmp,1e-6*ones(obj.par.N,1));
                obj.par.sub(l).w2=max(0.5*ones(size(var(x{l})')),1e-6*ones(obj.par.N,1));
                obj.par.sub(l).beta=mean(tmp);
            end
            obj.par.TR=2.49;
            if nargin==4
                switch swsampler
                    case 'ss'
                        obj.par.hypersamplers{end}='sample_sig2homo_w2homo';
                        for l=1:obj.par.nsubjects
                            obj.par.sub(l).sig2=mean(obj.par.sub(l).sig2)*ones(size(obj.par.sub(l).sig2));
                            obj.par.sub(l).w2=mean(obj.par.sub(l).w2)*ones(size(obj.par.sub(l).w2));
                        end
                    case 'sh'
                        obj.par.hypersamplers{end}='sample_sig2homo_w2het';
                        for l=1:obj.par.nsubjects
                            obj.par.sub(l).sig2=mean(obj.par.sub(l).sig2)*ones(size(obj.par.sub(l).sig2));
                        end
                    case 'hs'
                        obj.par.hypersamplers{end}='sample_sig2het_w2homo';
                        for l=1:obj.par.nsubjects
                            obj.par.sub(l).w2=mean(obj.par.sub(l).w2)*ones(size(obj.par.sub(l).w2));
                        end
                    case 'hh'
                        obj.par.hypersamplers{end}='sample_sig2_w2';
                    otherwise
                        error('sdf');
                end
            end
            calcss(obj,x);
        end
        function return_obj=copy(obj)
            return_obj=gmmgpmodel();
            return_obj.par=obj.par;
            return_obj.ss=obj.ss;
            return_obj.logPc=obj.logPc;
            return_obj.logP=obj.logP;
            return_obj.logZ=obj.logZ;
        end
        function [categoricalDist,logPnew,logdiff,addss]=compute_categorical(obj,~,n,comp)
            nsubjects=obj.par.nsubjects;
            
            nu_tmp=zeros(nsubjects,1);
            ids_tmp=cell(nsubjects,1);
            logids_tmp=cell(nsubjects,1);
            Vtxcs_tmp=cell(nsubjects,1);
            sxcvids_tmp=cell(nsubjects,1);
            
            if isempty(comp)
                K=obj.par.K;
                logPnew=zeros(K,nsubjects);
                sub=obj.par.sub;
                s=obj.ss;
                for l=1:nsubjects
                    nu_tmp(l)=sub(l).w2(n)/sub(l).sig2(n);
                    ids_tmp{l}=s(l).ids+nu_tmp(l);
                    logids_tmp{l}=sum(log(ids_tmp{l}));
                    Vtxcs_tmp{l}=bsxfun(@plus,s(l).Vtxcs,s(l).XtVs(:,n));
                    sxcvids_tmp{l}=sum(Vtxcs_tmp{l}.^2./ids_tmp{l});
                    logPnew(1:K,l)=(-0.5*logids_tmp{l}+0.5*sxcvids_tmp{l})';
                end
                logdiff=sum(logPnew-obj.logPc,2);
                categoricalDist=(obj.par.nk+obj.par.alpha/obj.par.K).*exp(logdiff-max(logdiff));
                addss.ids_tmp=ids_tmp;
                addss.logids_tmp=logids_tmp;
                addss.Vtxcs_tmp=Vtxcs_tmp;
                addss.sxcvids_tmp=sxcvids_tmp;
            else
                logPnew=zeros(2,obj.par.nsubjects);
                for l=1:obj.par.nsubjects
                    nu_tmp=obj.par.sub(l).w2(n)/obj.par.sub(l).sig2(n);
                    ids_tmp{l}(:,comp)=obj.ss(l).ids(:,comp)+nu_tmp;
                    logids_tmp{l}(comp)=sum(log(ids_tmp{l}(:,comp)));
                    Vtxcs_tmp{l}(:,comp)=bsxfun(@plus,obj.ss(l).Vtxcs(:,comp),obj.ss(l).XtVs(:,n));
                    sxcvids_tmp{l}(comp)=sum(Vtxcs_tmp{l}(:,comp).^2./(ids_tmp{l}(:,comp)));
                    logPnew(:,l)=-0.5*logids_tmp{l}(comp)+0.5*sxcvids_tmp{l}(comp);
                end
                logdiff=sum(logPnew-obj.logPc(comp,:),2);
                categoricalDist=(obj.par.nk(comp)+obj.par.alpha/obj.par.K).*exp(logdiff-max(logdiff));
                addss.ids_tmp=ids_tmp;
                addss.logids_tmp=logids_tmp;
                addss.Vtxcs_tmp=Vtxcs_tmp;
                addss.sxcvids_tmp=sxcvids_tmp;
            end
        end
        function remove_observation(obj,~,n)
            k=obj.par.z(n);
            obj.par.nk(k)=obj.par.nk(k)-1;
            for l=1:obj.par.nsubjects
                nu_tmp=obj.par.sub(l).w2(n)/obj.par.sub(l).sig2(n);
                %                 obj.ss.nu{l}(z(n))=obj.ss.nu{l}(z(n))-nu_tmp;
                obj.ss(l).ids(:,k)=obj.ss(l).ids(:,k)-nu_tmp;
                obj.ss(l).logids(:,k)=sum(log(obj.ss(l).ids(:,k)));
                obj.ss(l).Vtxcs(:,k)=obj.ss(l).Vtxcs(:,k)-obj.ss(l).XtVs(:,n);
                obj.ss(l).sxcvids(k)=sum(obj.ss(l).Vtxcs(:,k).^2./obj.ss(l).ids(:,k));
                obj.logPc(k,l)=-0.5*obj.ss(l).logids(k)+0.5*obj.ss(l).sxcvids(k);
            end
        end
        function add_observation(obj,n,k,addss,~,~)
            if k>length(obj.par.nk)
                obj.par.nk(k,1)=1;
                for l=1:obj.par.nsubjects
                    nu_tmp=obj.par.sub(l).w2(n)/obj.par.sub(l).sig2(n);
                    obj.ss(l).ids(:,k)=obj.ss(l).ID+nu_tmp;
                    obj.ss(l).logids(:,k)=sum(log(obj.ss(l).ids(:,k)));
                    obj.ss(l).Vtxcs(:,k)=obj.ss(l).XtVs(:,n);
                    obj.ss(l).sxcvids(k)=sum(obj.ss(l).Vtxcs(:,k).^2./obj.ss(l).ids(:,k));
                    %                     obj.logPc(k,l)=-0.5*obj.ss(l).logids(k)+0.5*obj.ss(l).sxcvids(k);
                end
            else
                obj.par.nk(k,1)=obj.par.nk(k)+1;
                for l=1:obj.par.nsubjects
                    %                     nu_tmp=obj.par.sub(l).w2(n)/obj.par.sub(l).sig2(n);
                    obj.ss(l).ids(:,k)=addss.ids_tmp{l}(:,k);%obj.ss(l).ids(:,k)+nu_tmp;
                    obj.ss(l).logids(:,k)=addss.logids_tmp{l}(k);%sum(log(obj.ss(l).ids(:,k)));
                    obj.ss(l).Vtxcs(:,k)=addss.Vtxcs_tmp{l}(:,k);%obj.ss(l).Vtxcs(:,k)+obj.ss(l).XtVs(:,n);
                    obj.ss(l).sxcvids(k)=addss.sxcvids_tmp{l}(k);%sum(obj.ss(l).Vtxcs(:,k).^2./obj.ss(l).ids(:,k));
                    %                     obj.logPc(k,l)=-0.5*obj.ss(l).logids(k)+0.5*obj.ss(l).sxcvids(k);
                end
            end
        end     
        function remove_empty_clusters(~)
        end        
        function calcss(obj,X,subjects)
            z=obj.par.z;
            if max(z)>obj.par.K
                error('Bad clustering, max(z)>K');
            end
            if nargin==2
                subjects=1:obj.par.nsubjects;
                obj.ss=[];
                obj.par.nk=ones(obj.par.K,1);
                obj.logPc=zeros(obj.par.K,obj.par.nsubjects);
                obj.logP=zeros(obj.par.nsubjects,1);
                for i=1:obj.par.K
                    obj.par.nk(i)=sum(z==i);
                end
            end
            %kernel
            rbf_kernel = @(p,q,kernel_parms_rbf) exp(-1/(2*kernel_parms_rbf^2)*(repmat(p',size(q)) - repmat(q,size(p'))).^2);
            for l=subjects
                T=size(X{l},1);
                obj.par.sub(l).Sigma=obj.par.sub(l).beta*rbf_kernel(1:T,1:T,obj.par.TR);
                obj.par.sub(l).Sigma=obj.par.sub(l).Sigma+1e-9*diag(diag(obj.par.sub(l).Sigma));
                [obj.ss(l).V,obj.ss(l).D]=eig(obj.par.sub(l).Sigma);
                obj.ss(l).D=diag(obj.ss(l).D);
                obj.ss(l).ID=1./obj.ss(l).D;
                obj.ss(l).slD=sum(log(obj.ss(l).D));
            end
            
            for l=subjects
                [T,~]=size(X{l});
                obj.ss(l).XtVs = bsxfun(@times, sqrt(obj.par.sub(l).w2')./obj.par.sub(l).sig2',obj.ss(l).V'*X{l});
                nu_tmp=zeros(obj.par.K,1);
                for i=1:obj.par.K
                    idx=(z==i);
                    obj.ss(l).Vtxcs(:,i)=sum(obj.ss(l).XtVs(:,idx),2);
                    nu_tmp(i)=sum(obj.par.sub(l).w2(idx)./obj.par.sub(l).sig2(idx));
                end
                
                obj.ss(l).ids=bsxfun(@plus,1./obj.ss(l).D,repmat(nu_tmp',T,1));
                obj.ss(l).logids=sum(log(obj.ss(l).ids));
                obj.ss(l).sxcvids=sum(obj.ss(l).Vtxcs.^2./obj.ss(l).ids);
                
                obj.logPc(:,l)=-0.5*obj.ss(l).logids+0.5*obj.ss(l).sxcvids;
                obj.logP(l)=-0.5*T*obj.par.N*log(2*pi)-0.5*T*sum(log(obj.par.sub(l).sig2))-obj.par.K/2*sum(log(obj.ss(l).D))-0.5*norm(bsxfun(@times,X{l},1./sqrt(obj.par.sub(l).sig2')),'fro')^2;
            end
            obj.updateLogP(X);
            obj.updateLogZ();
        end 
        function updateLogP(obj,x)
            for l=1:obj.par.nsubjects
                obj.logP(l)=-0.5*obj.par.T*obj.par.N*log(2*pi)-0.5*obj.par.T*sum(log(obj.par.sub(l).sig2))-obj.par.K/2*sum(log(obj.ss(l).D))-0.5*norm(bsxfun(@times,x{l},1./sqrt(obj.par.sub(l).sig2')),'fro')^2;
            end
        end        
    end
end