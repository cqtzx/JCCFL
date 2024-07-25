function [L,D,W] = computeL(X,manifold)
%% input 
%%% X: fea*n
%%% manifold: the construct options of graph
    %% Construct graph Laplacian
    if ~isfield(manifold,'normr')
        manifold.normr=1;
    end
    n=size(X,2);
    W = lapgraph(X',manifold);
    D=diag(sparse(sum(W)));

   
        L = D-W;
   L=sparse(L);
    
end

