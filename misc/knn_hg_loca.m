function H = knn_hg_loca(X, k)

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
