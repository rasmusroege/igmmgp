classdef AbsGMM < handle
    
    properties
        
    end
    methods
        function obj=AbsGMM()
        end
        
        function lq=sample_lambda(obj,X,maxiter)
            lq=0;
            stepsize_lambda=0.1;
            for i=1:maxiter
                lambda=arrayfun(@(x)x.lambda,obj.par.sub);
                n=copy(obj);
                lambdanew=exp(log(lambda)+stepsize_lambda*randn(size(lambda)));
                for l=1:obj.par.nsubjects;n.par.sub(l).lambda=lambdanew(l);end;
                n.calcss(X);
                for l=1:obj.par.nsubjects
                    if rand<exp(sum(n.logPc(:,l))-sum(obj.logPc(:,l)));
                        obj.ss(l)=n.ss(l);
                        obj.logPc(:,l)=n.logPc(:,l);
                        obj.par.sub(l).lambda=lambdanew(l);
                    end
                end
            end
        end
        
        function lq=sample_gamma(obj,X,maxiter)
            lq=0;
            stepsize_gamma=0.1;
            for i=1:maxiter
                gamma=arrayfun(@(x)x.gamma,obj.par.sub);
                n=copy(obj);
                gammanew=max(exp(log(gamma)+stepsize_gamma*randn(size(gamma))),1e-9*ones(size(gamma)));
                for l=1:obj.par.nsubjects;n.par.sub(l).gamma=gammanew(l);end;
                n.calcss(X);
                for l=1:obj.par.nsubjects
                    if rand<exp(sum(n.logPc(:,l))-sum(obj.logPc(:,l)));
                        obj.ss(l)=n.ss(l);
                        obj.logPc(:,l)=n.logPc(:,l);
                        obj.par.sub(l).gamma=gammanew(l);
                    end
                end
            end
        end
    end
end