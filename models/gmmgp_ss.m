classdef gmmgp_ss<handle & gmmgpmodel
    methods
        function obj=gmmgp_ss(x,z,K)
            if nargin== 0
                super_args={};
            else
                super_args={x,z,K};
            end
            obj=obj@gmmgpmodel(super_args{:});
            if nargin==2
                for l=1:obj.par.nsubjects
                    obj.par.sub(l).sig2=mean(obj.par.sub(l).sig2)*ones(size(obj.par.sub(l).sig2));
                    obj.par.sub(l).w2=mean(obj.par.sub(l).w2)*ones(size(obj.par.sub(l).w2));
                end
                obj.calcss(x);
            end
        end
        
        function returnobj=copy(obj)
            returnobj=gmmgp_ss();
            returnobj.par=obj.par;
            returnobj.ss=obj.ss;
            returnobj.logZ=obj.logZ;
            returnobj.logP=obj.logP;
            returnobj.logPc=obj.logPc;
        end
        
        
    end
end