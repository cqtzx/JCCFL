function [P,T] = JCCFL(Xs,Y_train,Xt,Xt_label,alpha,lambda,delta,...
    gamma,sigma)
maxIter = 100;
rho = 1.1;
tua = 1;
obj1 = 1; er = 1; accc = 1;
[X,ns,nt,n,m,C] = datasetMsg(Xs,Y_train,Xt);
% Construct graph Laplacian
k = 5;
options.ReducedDim = C;
[P,~] = PCA1(Xs',options);
manifold.k = k;
manifold.Metric = 'Cosine';
manifold.WeightMode = 'Cosine';
manifold.NeighborMode = 'KNN';
[L,~] = computeL(P'*X,manifold);
L=full(L);
tol1 = 1e-5;              % threshold for the error in constraint
max_mu = 1e10;
mu = 0.1;%e-1;
knn_model = fitcknn(Xs',Y_train,'NumNeighbors',1);
Ytpesudo = knn_model.predict((Xt)');
RY_train = hotmatrix(Y_train,C)';
RY_test = hotmatrix(Ytpesudo,C)';
RY = [RY_train RY_test];
[Yrow,~] = size(RY);
%% Initializing optimization variables
% intializing
E = zeros(C,nt);
Y1 = zeros(C,nt);
Y3 = zeros(C+1, ns+nt);
Z = zeros(ns,nt);
T = zeros(Yrow, C+1);
Y4 = zeros(m,C);
F = zeros(m,C);
D = [P'*X;ones(1,size(X,2))];
%% Start main loop
normfX = norm(P'*Xs,'fro');
for iter = 1:maxIter
    PXs = P'*Xs;
    RY = [RY_train RY_test];
    
    N=0;
    if ~isempty(Ytpesudo)
        Mc=conditionalDistribution(Xs,Xt,Y_train,Ytpesudo,C);
        M0 = marginalDistribution(Xs,Xt,C);
        N= M0+Mc;
        N=N/norm(N,'fro');
    end
    
    %update T
    T = RY*D'/(D*D'+tua*eye(size(D,1)));
    
    %update D
    R = alpha*(T'*T)+mu*eye(size(T,2));
    D = (R)\(mu*[P'*X;ones(1,size(D,2))]-Y3+alpha*T'*RY);
    
    %update Z
    norm2X = norm(PXs,2);
    eta = norm2X*norm2X*1.1*mu;   %eta needs to be larger than ||X||_2^2, but need not be too large.
    M = mu*PXs'*(PXs*Z-P'*Xt+E-Y1/mu);
    M = Z-M/(eta);
    [U, S, V] = svd((M+eps),'econ');
    S = diag(S);
    svp = length(find(S>1/(eta)));
    if svp>=1
        S = S(1:svp)-1/eta;
    else
        svp = 1;
        S = 0;
    end
    AU = U(:, 1:svp);
    As = S(1:svp);
    AV = V(:, 1:svp);
    Z = AU*diag(As)*AV';
    
    %update E
    temp = P'*Xt-PXs*Z+Y1/mu;
    %E = max(0,temp-lambda/mu)+min(0,temp+lambda/mu);
    %E = max(0,E);
    for j = 1 : size(E, 2)
        if norm(temp(:, j), 2) > lambda/mu
            E(:, j) =( norm(temp(:,j), 2) - lambda/mu)/norm(temp(:,j),2)*temp(:,j);
        else
            E(:, j) = 0;
        end
    end
    
    %update F
    temp_F = P+Y4/mu;
    for j = 1 : size(F, 2)
        if norm(temp_F(:, j), 2) > gamma/mu
            F(:, j) =( norm(temp_F(:,j), 2) - gamma/mu)/norm(temp_F(:,j),2)*temp_F(:,j);
        else
            F(:, j) = 0;
        end
    end
    
    %update P
    AA = [Xt-Xs*Z X];
    BB = [E-Y1/mu D(1:C,:)+Y3(1:C,:)/mu];
    AAA = AA*AA'+(delta*X*N*X'+sigma*X*L*X')+eye(m);
    AAA = (AAA+AAA')/2;
    BBB = AA*BB'+F-Y4/mu;
    P = GPI(P,AAA,BBB,1);
%    %update P
%    V1 = Xt-Xs*Z;
%    V2 = E-Y1/mu;
%    V3 = D(1:C,:)+Y3(1:C,:)/mu;
%    AAA = (mu/2)*(V1*V1'+X*X'+eye(m))+delta*X*N*X'+sigma*X*L*X';
%    AAA = (AAA+AAA')/2;
%    BBB = (mu/2)*(V1*V2'+X*V3'+F-Y4/mu);
%    P = GPI(P,AAA,BBB,1);
    %update L
    [L,~] = computeL(P'*X,manifold);
    
    %update Y
    X_train = T*[(P')*Xs;ones(1,size(Xs,2))];
    RY_test = T*[(P')*Xt;ones(1,size(Xt,2))];
    X_train = X_train./repmat(sqrt(sum(X_train.^2)),[size(X_train,1) 1]);
    Y_test = RY_test;
    Y_test = Y_test./repmat(sqrt(sum(Y_test.^2)),[size(Y_test,1) 1]);
    knn_model = fitcknn(X_train',Y_train,'NumNeighbors',1);
    Ytpesudo = predict(knn_model,Y_test');
    %% convergence
    [~,DDD,~] = svd(Z);
    obj = norm([RY_train RY_test]-T*[(P')*X;ones(1,size(X,2))],'fro')+...
        sum(abs(diag(DDD)))+delta*trace(P'*X*N*X'*P)+sigma*trace(P'*X*L*X'*P)+sum(sum(P));
    obj1 = [obj1;obj];
    dY1 = P'*Xt - P'*Xs*Z - E;
    recErr1 = norm(dY1,'fro')/normfX;
    er = [er;recErr1];
    dY4 =  P - F;
    recErr4 = norm(dY4,'fro')/normfX;
    dY3 = D-[P'*X;ones(1,size(D,2))];
    recErr3 = norm(dY3,'fro')/normfX;
    recErr = max(max(recErr1, recErr3),recErr4);
    
    convergenced = recErr <tol1;
    if convergenced
        break;
    else
        Y1 = Y1 + mu*dY1;
        Y3 = Y3 + mu*dY3;
        Y4 = Y4 + mu*dY4;
        mu = min(max_mu, mu*rho);
    end
    %     %预测精度的收敛性
    X_train1 = T*[(P')*Xs;ones(1,size(Xs,2))];
    Y_test1 = T*[(P')*Xt;ones(1,size(Xt,2))];
    X_train1 = X_train1./repmat(sqrt(sum(X_train1.^2)),[size(X_train1,1) 1]);
    Y_test1 = Y_test1./repmat(sqrt(sum(Y_test1.^2)),[size(Y_test1,1) 1]);
    knn_model1 = fitcknn(X_train1',Y_train,'NumNeighbors',1);
    result1 = predict(knn_model1,Y_test1');
    acc1 = length(find(result1==Xt_label))/length(Xt_label)*100;
    accc = [accc;acc1];
    
    
end
%er(1)=[];plot(er); obj1(1)=[];plot(obj1); accc(1)=[];plot(accc);
end



