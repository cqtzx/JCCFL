load(['dslr_SURF_L10.mat']);     % source domain
fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
Xs = zscore(fts,1);    clear fts
Xs_label = labels;           clear labels

load(['webcam_SURF_L10.mat']);     % target domain
fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
Xt = zscore(fts,1);     clear fts
Xt_label = labels;            clear labels
Xs =Xs';
Xt =Xt';
Xs = Xs./repmat(sqrt(sum(Xs.^2)),[size(Xs,1) 1]);
Xt = Xt./repmat(sqrt(sum(Xt.^2)),[size(Xt,1) 1]);

alpha=1;
lambda=0.001;
gamma=0.001;
delta=100;
sigma=0.1;
[P,T] = JCCFL(Xs,Xs_label,Xt,Xt_label,alpha,lambda,delta,gamma,sigma);
X_train = T*[(P')*Xs;ones(1,size(Xs,2))];
Y_test = T*[(P')*Xt;ones(1,size(Xt,2))];
X_train = X_train./repmat(sqrt(sum(X_train.^2)),[size(X_train,1) 1]);
Y_test = Y_test./repmat(sqrt(sum(Y_test.^2)),[size(Y_test,1) 1]);
knn_model = fitcknn(X_train',Xs_label,'NumNeighbors',1);
result = predict(knn_model,Y_test');
acc = length(find(result==Xt_label))/length(Xt_label)*100