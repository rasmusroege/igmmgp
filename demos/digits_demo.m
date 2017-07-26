addpath(genpath('..'));
path=mfilename('fullpath');

data=load(fullfile(fileparts(path),'..','Data','digits.mat'));

clf;
imagesc(reshape(data.X(2,:),[16 16])')
axis equal;

x={data.X'};
m=igmmddmodel(x,data.y+1);
m.initmodel(x,100);
o=struct();o.maxiter=100;
infsample(x,m,o)