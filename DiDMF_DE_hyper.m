function [ Z, Hstar,derror ] = DiDMF_DE_hyper( XX, layers, varargin )

pnames = { ...
    'z0' 'h0' 'bUpdateH' 'bUpdateLastH' 'maxiter' 'TolFun', ...
    'verbose', 'bUpdateZ', 'cache', 'gnd',  'beta','mu'...
    };


numOfView = numel(XX);
num_of_layers = numel(layers);
numOfSample = size(XX{1,1},2);
% D:V types of features
N = numOfSample;

E = ones(size(layers(num_of_layers),N));
Z = cell(numOfView, num_of_layers);
H = cell(numOfView, num_of_layers);
H_cell = cell(numOfView,1);
dflts  = {0, 0, 1, 1, 100, 1e-5, 1, 1, 0, 0};

[z0, h0, bUpdateH, bUpdateLastH, maxiter, tolfun, verbose, bUpdateZ, cache, gnd, beta, mu] = ...
    internal.stats.parseArgs(pnames,dflts,varargin{:});

A_graph = cell(1,numOfView);
D_graph = cell(1,numOfView);
L_graph = cell(1,numOfView);

param.k = 5;

for v_ind = 1:numOfView
    X = XX{v_ind};
    X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
    
%     A_graph{v_ind} = constructW(X', options);
%     D_graph{v_ind} = diag(sum(constructW(X', options),2));
%     L_graph{v_ind} = D_graph{v_ind} - A_graph{v_ind};
     Weight{v_ind} = constructW_PKN(X, param.k);
     Diag_tmp = diag(sum(Weight{v_ind}));
     L_graph{v_ind} = Diag_tmp - Weight{v_ind};    

    if  ~iscell(h0)
        for i_layer = 1:length(layers)
            if i_layer == 1
                % For the first layer we go linear from X to Z*H, so we use id
                V = X;
            else
                V = H{v_ind,i_layer-1};
            end
            
            if verbose
                display(sprintf('Initialising Layer #%d with k=%d with size(V)=%s...', i_layer, layers(i_layer), mat2str(size(V))));
            end
            if ~iscell(z0)
                % For the later layers we use nonlinearities as we go from
                % g(H_{k-1}) to Z*H_k
                [Z{v_ind,i_layer}, H{v_ind,i_layer}, ~] = ...
                    seminmf(V, ...
                    layers(i_layer), ...
                    'maxiter', maxiter, ...
                    'bUpdateH', true, 'bUpdateZ', bUpdateZ, 'verbose', verbose, 'save', cache, 'fast', 1);
            else
                display('Using existing Z');
                [Z{v_ind,i_layer}, H{v_ind,i_layer}, ~] = ...
                    seminmf(V, ...
                    layers(i_layer), ...
                    'maxiter', 1, ...
                    'bUpdateH', true, 'bUpdateZ', 0, 'z0', z0{i_layer}, 'verbose', verbose, 'save', cache, 'fast', 1);
            end
        end
        
    else
        Z=z0;
        H=h0;
        
        if verbose
            display('Skipping initialization, using provided init matrices...');
        end
    end
    
%     dnorm0(v_ind) = cost_function(X, Z(v_ind,:), H(v_ind,:)); %, L_graph{v_ind}, beta,gamma
%     dnorm(v_ind) = dnorm0(v_ind) + 1;
   
end
% get the error for each view
for v_ind = 1:numOfView
    X = XX{v_ind};
    X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
    dnorm(v_ind) = cost_function_graph(X, Z(v_ind,:), H(v_ind,:), L_graph{v_ind}, beta);
    for www = 1:numOfView
        if (abs(www-v_ind)>0)
            dorm_diver_www(www) = mu*trace(H{v_ind,num_of_layers}'*H{v_ind,num_of_layers}*H{www,num_of_layers}'*H{www,num_of_layers});
        end
    end   
    dorm_diver(v_ind) = sum(dorm_diver_www(www));
    dnorm_all(v_ind) = dnorm(v_ind)+dorm_diver(v_ind);   
end
maxDnorm = sum(dnorm_all);
dnorm0 = maxDnorm;
if verbose
    display(sprintf('#%d error: %f', 0, sum(dnorm0)));
end

%% Error Propagation

if verbose
    display('Finetuning...');
end
H_err = cell(numOfView, num_of_layers);
derror = [];
start = 1;
for iter = 1:maxiter
    Hm_a = 0; Hm_b = 0;
    for v_ind = 1:numOfView
        X = XX{v_ind};       
        X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
        
       H_err{v_ind,numel(layers)} = H{v_ind,numel(layers)};
       H_last = H_err{v_ind,numel(layers)};

       if start==1
         Weight{v_ind} = constructW_PKN(H_last, param.k);
         Diag_tmp = diag(sum(Weight{v_ind}));
         L_graph{v_ind} = Diag_tmp - Weight{v_ind};
       else
%        ------------modified to hyper-graph---------------
%          P =  (abs(H_last)+abs(H_last'))./2;
         HG = gsp_nn_hypergraph(H_last', param);
         L_graph{v_ind} = HG.L;
       end    
       start=0;
        for i_layer = numel(layers)-1:-1:1
            H_err{v_ind,i_layer} = Z{v_ind,i_layer+1} * H_err{v_ind,i_layer+1};
        end
        
        for i = 1:numel(layers)
 %           E = eye(size(H_err{v_ind,i},1));
            if bUpdateZ
%                 try
                    if i == 1
                         Z{v_ind,i} = X  * pinv(H_err{v_ind,1});
 %                       Z{v_ind,i} = (X  * H_err{v_ind,1}')* inv(H_err{v_ind,1}*H_err{v_ind,1}');
                    else
                        Z{v_ind,i} = pinv(D') * X * pinv(H_err{v_ind,i});
 %                       Z{v_ind,i} = pinv(D') * (X* H_err{v_ind,i}')* inv(H_err{v_ind,i}*H_err{v_ind,i}');
                    end
%                 catch
%                     display(sprintf('Convergance error %f. min Z{i}: %f. max %f', norm(Z{v_ind,i}, 'fro'), min(min(Z{v_ind,i})), max(max(Z{v_ind,i}))));
%                 end
            end
            
            if i == 1
                D = Z{v_ind,1}';
            else
                D = Z{v_ind,i}' * D;
            end
            
            if bUpdateH && (i < numel(layers) || (i == numel(layers) && bUpdateLastH))   
                % original one
                A = D * X;
                
                Ap = (abs(A)+A)./2;
                An = (abs(A)-A)./2;               
                
                % original noe
                B = D * D';
                Bp = (abs(B)+B)./2;
                Bn = (abs(B)-B)./2;
                                                                
                % Hm*L -> HmL
                HmL = H{v_ind,i}* L_graph{v_ind};
                HmLp = (abs(HmL)+HmL)./2;
                HmLn = (abs(HmL)-HmL)./2;                  
                % update graph part                
                H{v_ind,i} = H{v_ind,i} .* sqrt((Ap + Bn* H{v_ind,i} ) ./ max(An + Bp* H{v_ind,i}, 1e-10));
                                            
                % update the last consensus layer
                if i == numel(layers)
                    % Hm 
                    Hm = H{v_ind,i};
                    Hmp = (abs(Hm)+Hm)./2;
                    Hmn = (abs(Hm)-Hm)./2;   
                    
                R =zeros(size(H{v_ind,num_of_layers}*H{v_ind,num_of_layers}'*H{v_ind,num_of_layers}));            
                for k=1:numOfView
                    if (k==v_ind) 
                        continue;
                    end
                    R =  R + H{v_ind,num_of_layers}*H{k,num_of_layers}'*H{k,num_of_layers}; 
                end 

                    Hm_a = (Ap + Bn* H{v_ind,i} + beta*(HmLn)); %+gamma*Hmn
                    Hm_b = (max(An + Bp* H{v_ind,i} + beta*(HmLp), 1e-10));%+gamma*Hmp
                    H{v_ind,i} = H{v_ind,i} .* sqrt((Hm_a) ./ (Hm_b+mu*R));
                end
            end
        end        
        assert(i == numel(layers));
    end

    Hstar = zeros(layers(num_of_layers),N); 
    dorm_diver_www = [];
    for v_ind = 1:numOfView    
        X = XX{v_ind};
        X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
        
        % update Hm
        Hstar = Hstar + H{v_ind,num_of_layers};%
        H_cell{v_ind}= H{v_ind,num_of_layers};
        % get the error for each view
        dnorm(v_ind) = cost_function_graph(X, Z(v_ind,:), H(v_ind,:), L_graph{v_ind}, beta);
        for www = 1:numOfView
            if (abs(www-v_ind)>0)
                dorm_diver_www(www) = mu*trace(H{v_ind,num_of_layers}'*H{v_ind,num_of_layers}*H{www,num_of_layers}'*H{www,num_of_layers});
%                 dorm_diver(v_ind) = a;
            end
        end   
        dorm_diver(v_ind) = sum(dorm_diver_www(www));
        dnorm_all(v_ind) = dnorm(v_ind)+dorm_diver(v_ind);

        % the following two lines are used for calculating weight
%         tmpNorm = cost_function_graph(X, Z(v_ind,:), H(v_ind,:), 1, L_graph{v_ind}, beta);
%         dnorm_w(v_ind) = (gamma*(tmpNorm))^(1/(1-gamma));
    end
    % update alpha
%     for v_ind = 1:numOfView,      
%         alpha(v_ind) = dnorm_w(v_ind)/sum(dnorm_w);
%     end
              
    % finish update Z H and other variables in each view
    % disp result
    
    maxDnorm = sum(dnorm_all);
    if verbose
        display(sprintf('#%d error: %f', iter, maxDnorm));
        derror(iter) = maxDnorm;
    end
    
    %     assert(dnorm <= dnorm0 + 0.01, ...
    %         sprintf('Rec. error increasing! From %f to %f. (%d)', ...
    %         dnorm0, dnorm, iter) ...
    %     );
    
%    if verbose && length(gnd) > 1 && iter>1
%        if mod(iter, 1) == 0|| iter ==1
%            [acc, nmii, ~ ]= evalResults_multiview(Hstar, gnd);
%            ac = mean(acc);
%            ac_std = std(acc);
%           nmi = mean(nmii);
%            nmi_std = std(nmii);
%            fprintf(1, 'Clustering accuracy is %.4f, NMI is %.4f\n', ac, nmi);
%        end
%    end
    
%                  if iter>1 && dnorm0-maxDnorm <= tolfun*max(1,dnorm0)
%                      if verbose
%                          display( ...
%                              sprintf('Stopped at %d: dnorm: %f, dnorm0: %f', ...
%                                  iter, maxDnorm, dnorm0 ...
%                              ) ...
%                          );
%                      end
%                      break;
%                  end 
    dnorm0 = maxDnorm;
end      

% for v_ind = 1:numOfView
%     Hstar = Hstar + H{v_ind,num_of_layers};%���ӵ�
% %         if v_ind == 1%ƴ�ӵ�
% %             Hstar = H{v_ind,num_of_layers};
% %         else
% %             Hstar = [Hstar;H{v_ind,num_of_layers}];
% %         end
% end
    Hstar = Hstar./numOfView;%
end

function error = cost_function_graph(X, Z, H, A, beta)%delete 
out = H{numel(H)};
error =norm(X - reconstruction(Z, H), 'fro')^2 + beta* trace(out*A*out');
end

function [ out ] = reconstruction( Z, H )

out = H{numel(H)};

for k = numel(H) : -1 : 1
    out =  Z{k} * out;
end

end
