classdef igmmgp_sh<handle & igmmgpmodel
    methods
        function obj=igmmgp_sh(x,z)
            if nargin== 0
                super_args={};
            elseif nargin==2
                super_args={x,z};
            end
            obj=obj@igmmgpmodel(super_args{:});
            if nargin==2
                for l=1:obj.par.nsubjects
                    obj.par.sub(l).sig2=mean(obj.par.sub(l).sig2)*ones(size(obj.par.sub(l).w2));
                end
                obj.calcss(x);
            end
        end
        
        function returnobj=copy(obj)
            returnobj=igmmgp_sh();
            returnobj.par=obj.par;
            returnobj.ss=obj.ss;
            returnobj.logZ=obj.logZ;
            returnobj.logP=obj.logP;
            returnobj.logPc=obj.logPc;
        end
        
        
    end
end