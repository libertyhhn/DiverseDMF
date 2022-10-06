function [ ac, nmi, pur,ari,f,pre,rec,ac_std, nmi_std, pur_std,ari_std,f_std,pre_std,rec_std ] = evalResults_multiview( H, gnd,varargin )

if length(varargin)>0
    kk = varargin{1};
end

nClass = length(unique(gnd));

if iscell(H)
    H = H{:};
end

for iLoop = 1:1
    
    options = [];
    options.k = 5;
    
    options.WeightMode = 'HeatKernel';
    options.t = 1;
    A = constructW(H', options);
    gt = gnd; clusNum = nClass;
    
    C = SpectralClustering_GPU(A,clusNum);
    
    [A nmii(iLoop) avgent] = compute_nmi(gt,C);
    [F(iLoop),P(iLoop),R(iLoop)] = compute_f(gt,C);
    [AR(iLoop)]=RandIndex(gt,C);
    Pur(iLoop) = purity(gt,C);  
    C = bestMap(gt,C);
    ACC(iLoop) = length(find(gt == C))/length(gt);
end
ac = mean(ACC); ac_std = std(ACC);
nmi = mean(nmii); nmi_std = std(nmii);
pur = mean(Pur); pur_std = std(nmii);
ari = mean(AR); ari_std = std(AR);
f = mean(F); f_std = std(F);
pre = mean(P); pre_std = std(P);
rec = mean(R); rec_std = std(R);
fprintf(1, 'Spectral Clustering ACC is %.4f, NMI is %.4f, Pur is %.4f,ARI is %.4f, F_score is %.4f, Pre is %.4f, Rec is %.4f\n',...
    ac, nmi,pur,ari,f,pre,rec);
end

