% Code by Haonan Huang, Guangdong University of Technology, 2022.07.07
% Code for paper "Huang, et al., Diverse Deep Matrix Factorization with Hypergraph Regularization for Multiview Data Representation, IEEE JAS'22"
% contact libertyhhn@foxmail.com if you have any questions

% Courtesy to Haodong Zhao for the codes, from the following paper:
% "Zhao, et al., Multi-View Clustering via Deep Matrix Factorization, AAAI'17"

addpath(genpath('.'));
clear;
% warning off;
%%
dataname = {'ORL40_3_400'};
numdata = length(dataname);

for cdata = 1:numdata
%% read data
idata = cdata;
datadir = 'data/';
dataset = char(dataname(idata));
dataf = [datadir, cell2mat(dataname(idata))];
load(dataf);
C=length(unique(label));
for v = 1:length(data)
     data{v} = NormalizeFea(data{v}, 0);
end
gnd = label';

%%
savePath = './results_HDDMF/';

layers = [100 50] ; % the configuration of layer sizes
mu = 0.0001;        % the parameter of diversity
beta = 0.1;         % the parameter of hyper-graph

for i = 1:10
[ Z, H, dnorm ] = DiDMF_DE_hyper( data, layers,'gnd',gnd, 'beta', beta, 'mu',mu);
[acc1(i), nmii1(i),pur1(i),ari1(i),f1(i),pre1(i),rec1(i)]= evalResults_multiview(H, gnd); % spectral clustering
end

acc1m = mean(acc1); nmii1m = mean(nmii1);pur1m = mean(pur1);ari1m = mean(ari1);f1m= mean(f1);pre1m= mean(pre1);rec1m =mean(rec1);
acc1s = std(acc1);nmii1s = std(nmii1);pur1s = std(pur1);ari1s = std(ari1);f1s= std(f1);pre1s= std(pre1);rec1s =std(rec1);
eva_spe = [acc1m,acc1s,nmii1m,nmii1s,pur1m,pur1s,ari1m,ari1s,f1m,f1s,pre1m,pre1s,rec1m,rec1s]*100;
eva_spe = roundn(eva_spe,-2);
fprintf('10times Spe_Clu ac: %0.2f\tnmi:%0.2f\tpur:%0.2f\tar:%0.2f\tf_sc:%0.2f\tpre:%0.2f\trec:%0.2f\n', acc1m*100, nmii1m*100,pur1m*100,ari1m*100,f1m*100,pre1m*100,rec1m*100);

Tname = [savePath,dataset,'.txt'];
dlmwrite(Tname,eva_spe,'-append','delimiter','\t','newline','pc');
objectname = [savePath, dataset, '.mat' ];
save(objectname,'dnorm');
end
return