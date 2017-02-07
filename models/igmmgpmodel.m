classdef igmmgpmodel < AbsGmmgp & AbsInfiniteModel & handle
    % igmmgp_model
    properties
        par;
        ss;% D,V,ID, XtVs, Vtxcs, sxcvids, ids, logids, nk
        logPc=0; % nsubjects x K
        logP=0; % nsubjects x 1
        logZ=0; % 1 x 1
    end
    
    methods
        function obj=igmmgpmodel(x,z,swsampler)
            % constructor obj=igmmgp_model(X,z)
            if nargin==0
                super_args={};
            else
                super_args={x,z};
            end
            obj@AbsInfiniteModel(super_args{:});
            obj.par.hypersamplers=[obj.par.hypersamplers {'sample_beta','sample_sig2_w2'}];
            if nargin==0
                return;
            end
            for l=1:obj.par.nsubjects
                obj.par.sub(l).sig2=max(var(x{l})',1e-6*ones(obj.par.N,1));
                obj.par.sub(l).w2=max(0.5*ones(size(var(x{l})')),1e-6*ones(obj.par.N,1));
                obj.par.sub(l).beta=mean(var(x{l}));
            end
            obj.par.TR=2.49;
            if nargin==3
                switch swsampler
                    case 'ss'
                        obj.par.hypersamplers{end}='sample_sig2homo_w2homo';
                        for l=1:obj.par.nsubjects
                            obj.par.sub(l).sig2=mean(obj.par.sub(l).sig2)*ones(size(obj.par.sub(l).sig2));
                            obj.par.sub(l).w2=mean(obj.par.sub(l).w2)*ones(size(obj.par.sub(l).w2));
                        end
                    case 'hs'
                        obj.par.hypersamplers{end}='sample_sig2het_w2homo';
                        for l=1:obj.par.nsubjects
                            obj.par.sub(l).w2=mean(obj.par.sub(l).w2)*ones(size(obj.par.sub(l).w2));
                        end
                    case 'sh'
                        obj.par.hypersamplers{end}='sample_sig2homo_w2het';
                        for l=1:obj.par.nsubjects
                            obj.par.sub(l).sig2=mean(obj.par.sub(l).sig2)*ones(size(obj.par.sub(l).sig2));
                            
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
            return_obj=igmmgpmodel();
            return_obj.par=obj.par;
            return_obj.ss=obj.ss;
            return_obj.logPc=obj.logPc;
            return_obj.logP=obj.logP;
            return_obj.logZ=obj.logZ;
        end
        
        function merge_obj=initMerge(obj,~,z,comp)
            merge_obj=copy(obj);
            merge_obj.par.nk(comp(1))=sum(merge_obj.par.nk(comp));
            merge_obj.par.z=z;
            for l=1:merge_obj.par.nsubjects
                merge_obj.ss(l).Vtxcs(:,comp(1))=sum(merge_obj.ss(l).Vtxcs(:,comp),2);
                merge_obj.ss(l).ids(:,comp(1))=bsxfun(@plus,merge_obj.ss(l).ID,sum(merge_obj.par.sub(l).w2(z==comp(1))./merge_obj.par.sub(l).sig2(z==comp(1))));
                merge_obj.ss(l).logids(comp(1))=sum(log(merge_obj.ss(l).ids(:,comp(1))));
                merge_obj.ss(l).sxcvids(comp(1))=sum(merge_obj.ss(l).Vtxcs(:,comp(1)).^2./merge_obj.ss(l).ids(:,comp(1)));
                merge_obj.logPc(comp(1),l)=-0.5*merge_obj.ss(l).logids(comp(1))+0.5*merge_obj.ss(l).sxcvids(comp(1));
                
                merge_obj.ss(l).Vtxcs(:,comp(2))=[];
                merge_obj.ss(l).ids(:,comp(2))=[];
                merge_obj.ss(l).logids(comp(2))=[];
                merge_obj.ss(l).sxcvids(comp(2))=[];
                merge_obj.logP(l)=merge_obj.logP(l)+1/2*sum(log(merge_obj.ss(l).D));
            end
            merge_obj.par.nk(comp(2))=[];
            merge_obj.logPc(comp(2),:)=[];
            merge_obj.updateLogZ();
        end
        function launch_obj=initLaunch(obj,~,z,comp)
            launch_obj=copy(obj);
            launch_obj.par.nk(comp(1),1)=sum(z==comp(1));
            launch_obj.par.nk(comp(2),1)=sum(z==comp(2));
            launch_obj.par.z=z;
            for l=1:launch_obj.par.nsubjects
                launch_obj.ss(l).Vtxcs(:,comp(1))=launch_obj.ss(l).XtVs(:,z==comp(1));
                launch_obj.ss(l).Vtxcs(:,comp(2))=launch_obj.ss(l).XtVs(:,z==comp(2));
                launch_obj.ss(l).ids(:,comp(1))=bsxfun(@plus,launch_obj.ss(l).ID,sum(launch_obj.par.sub(l).w2(z==comp(1))./launch_obj.par.sub(l).sig2(z==comp(1))));
                launch_obj.ss(l).ids(:,comp(2))=bsxfun(@plus,launch_obj.ss(l).ID,sum(launch_obj.par.sub(l).w2(z==comp(2))./launch_obj.par.sub(l).sig2(z==comp(2))));
                launch_obj.ss(l).logids(comp(1))=sum(log(launch_obj.ss(l).ids(:,comp(1))));
                launch_obj.ss(l).logids(comp(2))=sum(log(launch_obj.ss(l).ids(:,comp(2))));
                launch_obj.ss(l).sxcvids(comp(1))=sum(launch_obj.ss(l).Vtxcs(:,comp(1)).^2./launch_obj.ss(l).ids(:,comp(1)));
                launch_obj.ss(l).sxcvids(comp(2))=sum(launch_obj.ss(l).Vtxcs(:,comp(2)).^2./launch_obj.ss(l).ids(:,comp(2)));
                
                launch_obj.logPc(comp(1),l)=-0.5*launch_obj.ss(l).logids(comp(1))+0.5*launch_obj.ss(l).sxcvids(comp(1));
                launch_obj.logPc(comp(2),l)=-0.5*launch_obj.ss(l).logids(comp(2))+0.5*launch_obj.ss(l).sxcvids(comp(2));
                launch_obj.logP(l)=launch_obj.logP(l)-1/2*sum(log(launch_obj.ss(l).D));
            end
            launch_obj.updateLogZ();
        end
        function z=remove_empty_clusters(obj)
            idx_empty=find(obj.par.nk==0);
            if ~isempty(idx_empty)
                obj.par.nk(idx_empty)=[];
                obj.par.z(obj.par.z>idx_empty)=obj.par.z(obj.par.z>idx_empty)-1;
                for l=1:obj.par.nsubjects
                    obj.ss(l).ids(:,idx_empty)=[];
                    obj.ss(l).logids(idx_empty)=[];
                    obj.ss(l).Vtxcs(:,idx_empty)=[];
                    obj.ss(l).sxcvids(idx_empty)=[];
                end
                obj.logPc(idx_empty,:)=[];
                %                 obj.removed_something=obj.removed_something+1;
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
        function [categoricalDist,logPnew,logdiff,addss]=compute_categorical(obj,~,n,comp)
            nsubjects=obj.par.nsubjects;
            
            nu_tmp=zeros(nsubjects,1);
            ids_tmp=cell(nsubjects,1);
            logids_tmp=cell(nsubjects,1);
            Vtxcs_tmp=cell(nsubjects,1);
            sxcvids_tmp=cell(nsubjects,1);
            
            if isempty(comp)
                K=max(obj.par.z);
                logPnew=zeros(K,nsubjects);
                sub=obj.par.sub;
                s=obj.ss;
                for l=1:nsubjects
                    nu_tmp(l)=sub(l).w2(n)/sub(l).sig2(n);
                    ids_tmp{l}=s(l).ids+nu_tmp(l);
                    logids_tmp{l}=sum(log(ids_tmp{l}));
                    Vtxcs_tmp{l}=bsxfun(@plus,s(l).Vtxcs,s(l).XtVs(:,n));
                    sxcvids_tmp{l}=sum(Vtxcs_tmp{l}.^2./ids_tmp{l});
                    logPnew(1:K,l)=[-0.5*logids_tmp{l}+0.5*sxcvids_tmp{l}]';
                    logPnew(K+1,l)=-0.5*sum(log(obj.ss(l).ID+nu_tmp(l)))+0.5*sum(obj.ss(l).XtVs(:,n).^2./(obj.ss(l).ID+nu_tmp(l)));
                end
                logdiff=sum(logPnew-[obj.logPc;+0.5*arrayfun(@(x)x.slD,obj.ss)],2);
                categoricalDist=[obj.par.nk;obj.par.alpha].*exp(logdiff-max(logdiff));
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
                categoricalDist=obj.par.nk(comp).*exp(logdiff-max(logdiff));
                addss.ids_tmp=ids_tmp;
                addss.logids_tmp=logids_tmp;
                addss.Vtxcs_tmp=Vtxcs_tmp;
                addss.sxcvids_tmp=sxcvids_tmp;
            end
        end
        
        function calcss(obj,X,subjects)
            z=obj.par.z;
            val=unique(z);
            val=setdiff(val,0);
            if nargin==2
                subjects=1:obj.par.nsubjects;
                obj.ss=[];
                obj.par.nk=ones(max(z),1);
                obj.logPc=zeros(max(z),obj.par.nsubjects);
                obj.logP=zeros(obj.par.nsubjects,1);
                for i=1:max(z)
                    obj.par.nk(i)=sum(z==i);
                end
            end
            %kernel
            rbf_kernel = @(p,q,kernel_parms_rbf) exp(-1/(2*kernel_parms_rbf^2)*(repmat(p',size(q)) - repmat(q,size(p'))).^2);
            for l=subjects
                T=size(X{l},1);
                obj.par.sub(l).Sigma=obj.par.sub(l).beta*rbf_kernel(1:T,1:T,2.49/obj.par.TR*1.85);
                obj.par.sub(l).Sigma=obj.par.sub(l).Sigma+1e-9*diag(diag(obj.par.sub(l).Sigma));
                [obj.ss(l).V,obj.ss(l).D]=eig(obj.par.sub(l).Sigma);
                obj.ss(l).D=diag(obj.ss(l).D);
                obj.ss(l).ID=1./obj.ss(l).D;
                obj.ss(l).slD=sum(log(obj.ss(l).D));
            end
            
            for l=subjects
                [T,~]=size(X{l});
                obj.ss(l).XtVs = bsxfun(@times, sqrt(obj.par.sub(l).w2')./obj.par.sub(l).sig2',obj.ss(l).V'*X{l});
                nu_tmp=zeros(max(z),1);
                for k=1:length(val)
                    i=val(k);
                    idx=(z==val(k));
                    obj.ss(l).Vtxcs(:,i)=sum(obj.ss(l).XtVs(:,idx),2);
                    nu_tmp(i)=sum(obj.par.sub(l).w2(idx)./obj.par.sub(l).sig2(idx));
                end
                
                obj.ss(l).ids=bsxfun(@plus,1./obj.ss(l).D,repmat(nu_tmp',T,1));
                obj.ss(l).logids=sum(log(obj.ss(l).ids));
                obj.ss(l).sxcvids=sum(obj.ss(l).Vtxcs.^2./obj.ss(l).ids);
                
                obj.logPc(:,l)=-0.5*obj.ss(l).logids+0.5*obj.ss(l).sxcvids;
            end
            obj.updateLogZ();
            obj.updateLogP(X);
        end
        function updateLogP(obj,x)
            for l=1:obj.par.nsubjects
                obj.logP(l)=-0.5*obj.par.T*obj.par.N*log(2*pi)-0.5*obj.par.T*sum(log(obj.par.sub(l).sig2))-max(obj.par.z)/2*sum(log(obj.ss(l).D))-0.5*norm(bsxfun(@times,x{l},1./sqrt(obj.par.sub(l).sig2')),'fro')^2;
            end
        end
    end
end