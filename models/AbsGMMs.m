classdef AbsGMMs < handle
    properties
        
    end
    
    methods
        function obj=AbsGMMs()
        end
        
        function lq=sample_lambda(obj,X,maxiter)
            lq=0;
            stepsize_lambda=0.1;
            for l=1:obj.par.nsubjects
                nn=obj.par.T*obj.par.nk/2+obj.par.sub(l).v0;
                for sample_iter=1:maxiter
                    ln=exp(log(obj.par.sub(l).lambda)+stepsize_lambda*randn);
                    R0=obj.par.sub(l).gamma+0.5*ln*sum(obj.par.sub(l).mu0.^2);
                    R2=R0+1/2*obj.ss(l).Sigma_avg-(1./(2*(obj.par.nk+ln))).*sum(bsxfun(@plus,obj.ss(l).x_avg,obj.par.sub(l).lambda*obj.par.sub(l).mu0).^2,1)';
                    lpc=obj.ss(l).logPrior-((nn).*log(R2)-gammaln(nn)-obj.par.T/2*log(ln./(obj.par.nk+ln)));
                    
                    if rand<(ln/obj.par.sub(l).lambda)*exp(sum(lpc)-sum(obj.logPc(:,l)))
                        lq=lq+sum(lpc)-sum(obj.logPc(:,l));
                        obj.par.sub(l).lambda=ln;
                        obj.logPc(:,l)=lpc;
                        obj.ss(l).R2=R2;
                        obj.ss(l).R0=R0;
                    end
                end
            end
        end
        
        function lq=sample_gamma(obj,X,maxiter)
            lq=0;
            stepsize_gamma=0.1;
            for j=1:maxiter
                for l=1:obj.par.nsubjects
                    gn=exp(log(obj.par.sub(l).gamma)+stepsize_gamma*randn);
                    logPrior=obj.par.sub(l).v0.*log(gn)-gammaln(obj.par.sub(l).v0);
                    R0=gn+0.5*obj.par.sub(l).lambda*sum(obj.par.sub(l).mu0.^2);
                    R2=R0+1/2*obj.ss(l).Sigma_avg-(1./(2*(obj.par.nk+obj.par.sub(l).lambda))).*sum(bsxfun(@plus,obj.ss(l).x_avg,obj.par.sub(l).lambda*obj.par.sub(l).mu0).^2,1)';
                    nn=obj.par.T*obj.par.nk/2+obj.par.sub(l).v0;
                    lpc=logPrior-((nn).*log(R2)-gammaln(nn)-obj.par.T/2*log(obj.par.sub(l).lambda./(obj.par.nk+obj.par.sub(l).lambda)));
                    if rand<(gn/obj.par.sub(l).gamma)*exp(sum(lpc)-sum(obj.logPc(:,l)));
                        lq=lq+sum(lpc)-sum(obj.logPc(:,l));
                        obj.par.sub(l).gamma=gn;
                        obj.ss(l).logPrior=logPrior;
                        obj.ss(l).R0=R0;
                        obj.ss(l).R2=R2;
                        obj.logPc(:,l)=lpc;
                    end
                end
            end
        end
        
        function lq=sample_v0(obj,X,maxiter)
            lq=0;
            stepsize_gamma=0.1;
            for j=1:maxiter
                for l=1:obj.par.nsubjects
                    v0n=exp(log(obj.par.sub(l).v0)+stepsize_gamma*randn);
                    logPrior=v0n.*log(obj.par.sub(l).gamma)-gammaln(v0n);
                    nn=obj.par.T*obj.par.nk/2+v0n;
                    lpc=logPrior-((nn).*log(obj.ss(l).R2)-gammaln(nn)-obj.par.T/2*log(obj.par.sub(l).lambda./(obj.par.nk+obj.par.sub(l).lambda)));
                    if rand<(v0n/obj.par.sub(l).v0)*exp(sum(lpc)-sum(obj.logPc(:,l)))
                        lq=lq+sum(lpc)-sum(obj.logPc(:,l));
                        obj.par.sub(l).v0=v0n;
                        obj.ss(l).logPrior=logPrior;
                        obj.logPc(:,l)=lpc;
                    end
                end
            end
            
        end
    end
end