addpath(genpath('..'));
path=mfilename('fullpath');
% load data
for dataset=1:7
    % load dataset
    data=load(fullfile(fileparts(path),'..','Data',sprintf('synth%d.mat',dataset)));
    
    % get data in a cell
    x={data.X'};
    
    % initialize model
    m=igmmmodel(x,randi(data.C,data.N,1));
    
    % setup number of iterations
    o=struct();
    o.maxiter=30;
    
    % perform clustering
    infsample(x,m,o);
    
    % plot results
    figure(dataset);clf; hold on,
    set(gcf,'windowstyle','docked');
    colors=get(gca,'colororder');
    cluster_mean=bsxfun(@rdivide,m.ss(1).x_avg,m.par.nk');
    for k=1:length(m.par.nk)
        plot(cluster_mean(1,k),cluster_mean(2,k),'o','markersize',10,'color',colors(k,:));
        plot(x{1}(1,m.par.z==k),x{1}(2,m.par.z==k),'x','color',colors(k,:));
    end
end