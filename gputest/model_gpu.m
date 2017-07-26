classdef model_gpu < handle
    properties
        par;
        ss;
        logPc;
        logP;
        logZ;
    end
    
    methods
        function obj=model_gpu(m)
            obj.par=m.par;
            for l=1:obj.par.nsubjects
                obj.ss(l).R2=gpuarray(obj.ss(l).R2);
                obj.ss(l).xt=gpuarray(obj.ss(l).xt);
                obj.ss(l).xt2=gpuarray(obj.ss(l).xt2);
                obj.ss(l).x_avg=gpuarray(obj.ss(l).x_avg);
                
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
        
        function ret=calc(obj)
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
        end
    end
end