function [ll,L ] = gsp_nn_hypergraph_multioptimaloca( Xin, param )
%GSP_NN_HYPERGRAPH Create a nearest neighbors hypergraph from a point cloud
%   Usage :  G = gsp_nn_hypergraph( Xin );
%            G = gsp_nn_hypergraph( Xin, param );
%
%   Input parameters:
%       Xin         : Input points
%       param       : Structure of optional parameters
%
%   Output parameters:
%       G           : Resulting graph
%
%   Example:
%
%           P = rand(100,2);
%           G = gsp_nn_hypergraph(P)
%
%   Additional parameters
%   ---------------------
%
%    param.use_flann : [0, 1]              use the FLANN library
%    param.center    : [0, 1]              center the data
%    param.rescale   : [0, 1]              rescale the data (in a 1-ball)
%    param.sigma     : float               the variance of the distance kernel
%    param.k         : int                 number of neighbors for knn
%
%   See also: gsp_nn_graph
%
%
%   Url: http://lts2research.epfl.ch/gsp/doc/graphs/gsp_nn_hypergraph.php

% Copyright (C) 2013-2016 Nathanael Perraudin, Johan Paratte, David I Shuman.
% This file is part of GSPbox version 0.7.0
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

% If you use this toolbox please kindly cite
%     N. Perraudin, J. Paratte, D. Shuman, V. Kalofolias, P. Vandergheynst,
%     and D. K. Hammond. GSPBOX: A toolbox for signal processing on graphs.
%     ArXiv e-prints, Aug. 2014.
% http://arxiv.org/abs/1408.5781

% Author: Nathanael Perraudin
% Date: 21 October 2015
% Testing: test_rmse
% Modified by Haonan Huang, for the optimal manifolds [ref to 20'ICDE
% Date: 7.10 2021

%   * *param.type*      : ['knn', 'radius']   the type of graph (default 'knn')
%   * *param.epsilon*   : float               the radius for the range search
%   * *param.use_l1*    : [0, 1]              use the l1 distance

    if nargin < 2
    % Define parameters
        param = {};
    end
    
    %Parameters
%     if ~isfield(param, 'type'), param.type = 'knn'; end
    if ~isfield(param, 'use_flann'), param.use_flann = 0; end
    if ~isfield(param, 'center'), param.center = 0; end
    if ~isfield(param, 'rescale'), param.rescale = 1; end
    if ~isfield(param, 'k'), param.k = 10; end

    param.type = 'knn';
%     if ~isfield(param, 'epsilon'), param.epsilon = 0.01; end
%     if ~isfield(param, 'use_l1'), param.use_l1 = 0; end
%     if ~isfield(param, 'target_degree'), param.target_degree = 0; end;
    paramnn = param;
%     paramnn.k = param.k +1;
    m = numel(Xin);
    for i = 1:m
        [indx{i}, ~, dist{i}] = gsp_nn_distanz(Xin{i}',Xin{i}',paramnn);
    end
%     switch param.type
%         case 'knn'
%             if param.use_l1
%                 if ~isfield(param, 'sigma'), param.sigma = mean(dist); end
%             else
%                 if ~isfield(param, 'sigma'), param.sigma = mean(dist)^2; end
%             end
%         case 'radius'
%             if param.use_l1
%                 if ~isfield(param, 'sigma'), param.sigma = epsilon/2; end
%             else
%                 if ~isfield(param, 'sigma'), param.sigma = epsilon.^2/2; end
%             end
%         otherwise
%             error('Unknown graph type')
%     end
    
    G.N = size(Xin{1},1);
    G.Ne = G.N;
    G.W = sparse(G.N,G.Ne);
    G.H = size(G.N,G.Ne);
    Wall = zeros(G.N,G.Ne);
    H_all = zeros(G.N,G.Ne);
    G.E = cell(G.Ne,1);
    k = param.k;
    for i = 1:m
        if ~isfield(param, 'sigma'), param.sigma = mean(dist{i})^2; end
        w = exp(-dist{i}.^2/param.sigma);
        for ii = 1:G.Ne
            edge = indx{i}((1:k)+(ii-1)*k);
            G.E{ii} = edge;
            % Here we use H for HW...
            G.H(edge,ii) = 1;
            G.W(edge,ii) = sqrt(sum(w(edge)));
        end
%         G.Hcell{i,1} = G.H;
        Wall = G.W+Wall;
        H_all = G.H+H_all;
        W{i} = G.W;
    end

%     for i = 1:m
%         Wopt(:,:,i) = G.Hcell{i,1};
%     end
%     Wopt = min(Wopt,[],3);
%     ll = find(Wopt==1); % the location of '1'
    ll = H_all>1;
%     Wopt(ll) = Wall(ll)./m;
%     G.W = Wopt;
for i = 1:m
    W{i}(ll) = 0;
    G.W = W{i};
    G.hypergraph = 1;
    G.directed = 0;
    
    %Fill in the graph structure
    G.coords = Xin{i};

    G.type = 'Nearest neighboors hypergraph';
    G.lap_type = 'un-normalized';
    G.sigma = param.sigma;
    G = gsp_graph_default_parameters(G);
    L{i}= G.L;
end
end


