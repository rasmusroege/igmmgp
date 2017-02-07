classdef AbsGMMdd_sh < handle
    
    properties
        
    end
    methods
        function obj=AbsGMMdd_sh()
        end
        function xmeans=getpostmeans(obj)
            z=obj.par.z;
            T=obj.par.T;
            xmeans=zeros(T,max(z),obj.par.nsubjects);
            for l=1:obj.par.nsubjects
                xmeans(:,:,l)=bsxfun(@rdivide,1/(1+obj.par.sub(l).lambda)*(obj.ss(l).x_avg+obj.par.sub(l).lambda*obj.par.sub(1).mu0),obj.par.nk');
            end
        end    
        
        function [E_Sigma]=compExpSig(obj)
            E_Sigma=repmat({zeros(obj.par.T,obj.par.T,length(obj.par.nk))},obj.par.nsubjects,1);
            for l=1:obj.par.nsubjects
                for k=1:length(obj.par.nk)
                    E_Sigma{l}(:,:,k)=diag((obj.ss(l).Sigma_avg+2*obj.par.sub(l).gamma+obj.par.sub(l).lambda*obj.par.sub(l).mu0.^2-1/(obj.par.nk(k)+obj.par.sub(l).lambda)*(obj.ss(l).x_avg(:,k)+obj.par.sub(l).lambda*obj.par.sub(l).mu0).^2)./(obj.par.nk(k)+obj.par.sub(l).v0-1));
                end
            end
        end
        
        function lq=sample_lambda(obj,X,maxiter)
            lq=0;
            stepsize_lambda=0.1;
            m=copy(obj);
            ln=exp(log(obj.par.sub(1).lambda)+stepsize_lambda*randn);
            for l=1:obj.par.nsubjects
                m.par.sub(l).lambda=ln;
            end
            m.calcss(X);
            if rand<ln/obj.par.sub(l).lambda*exp(m.llh-obj.llh)
                for l=1:obj.par.nsubjects
                    obj.par.sub(l).lambda=ln;
                end
                obj.ss=m.ss;
                obj.logPc=m.logPc;
            end
        end
        
        function lq=sample_gamma(obj,X,maxiter)
            lq=0;
            stepsize_gamma=0.1;
            m=copy(obj);
            gn=exp(log(obj.par.sub(1).gamma)+stepsize_gamma*randn);
            for l=1:obj.par.nsubjects
                m.par.sub(l).gamma=gn;
            end
            m.calcss(X);
            if rand<gn/obj.par.sub(l).gamma*exp(m.llh-obj.llh)
                for l=1:obj.par.nsubjects
                    obj.par.sub(l).gamma=gn;
                end
                obj.ss=m.ss;
                obj.logPc=m.logPc;
            end
        end
        
        function lq=sample_v0(obj,X,maxiter)
            lq=0;
            stepsize_v0=0.1;
            m=copy(obj);
            v0n=exp(log(obj.par.sub(1).v0)+stepsize_v0*randn);
            for l=1:obj.par.nsubjects
                m.par.sub(l).v0=v0n;
            end
            m.calcss(X);
            if rand<v0n/obj.par.sub(l).v0*exp(m.llh-obj.llh)
                for l=1:obj.par.nsubjects
                    obj.par.sub(l).v0=v0n;
                end
                obj.ss=m.ss;
                obj.logPc=m.logPc;
            end                        
        end
    end
end