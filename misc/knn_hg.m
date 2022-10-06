function L = knn_hg(X, k,ll)

nodes = size(X,1);
edges = nodes;

dist = pdist(X,'euclidean');
dist = squareform(dist);
[~, id] = sort(dist,2);
avg_dist = mean(mean(dist));
neighbors = id(:,1:k+1);
neighbors_val = dist(:);
neighbors_heat_weight = exp(-neighbors_val.^2/avg_dist.^2);
neighbors_heat_weight = reshape(neighbors_heat_weight', nodes, edges)';

H = zeros(edges,nodes);
W = zeros(1,nodes);
for i = 1:edges
    idx1 = neighbors(i,:);
    A = neighbors_heat_weight(idx1,idx1);
    A = A - eye(k+1,k+1);
    H(i,idx1) = 1;
    %W(i) = sum(A(:))/2;
    W(i) = sum(A(:))/sum(sum(A~=0));
end
% sum_W = sum(W);
% W = W./sum_W;   

H = H';% ¶¥µãx³¬±ß
H(ll) = 0;
De = sum(H,1);
Dv = sum(H.*repmat(W,nodes,1),2);

Dv_sqrt = diag(power(Dv,-0.5));
De_inv = diag(power(De, -1));
L = diag(Dv) - H*diag(W)*De_inv*H';%
S = Dv_sqrt*H*diag(W)*De_inv*H'*Dv_sqrt;
L_nor = eye(nodes,edges) - S;
