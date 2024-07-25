function [Ww, Wb] = MFAAffinityMatrix(K, id, NNw,NNb)
% compute distance in the kernel space using kernel matrix
temp = repmat(diag(K), 1, size(K,1));
dis = temp + temp' - 2*K;
dis(sub2ind(size(dis), [1:size(dis,1)], [1:size(dis,1)]))=inf;
temp = repmat(id.^2, 1, length(id));
idm = temp + temp' - 2*id*id';

disw = dis;
disw(idm~=0) = inf;
[temp, ixw]= sort(disw);
ixw(isinf(temp))=0;

if NNw==0 % Use the maximum possible number of within class
    NNw = max(sum(~isinf(temp)));
end

ixw = ixw(1:NNw, :);
ixtmp = repmat([1:size(K,1)], NNw, 1);
ixtmp= ixtmp(ixw(:)>0);
ixw = ixw(ixw(:)>0);
ixtmp = sub2ind(size(K), ixtmp(:), ixw(:));
Ww = zeros(size(K));
Ww(ixtmp) = 1;
Ww = Ww+ Ww';
Ww = double(Ww>0);

disb = dis;
disb(idm==0) = inf;
[temp, ixb]= sort(disb);
ixb(isinf(temp))=0;
if NNb == 0 % use the maximum possible number of between classes
    NNb = max(sum(~isinf(temp),1));
end
ixb = ixb(1:NNb, :);
ixtmp = repmat([1:size(K,1)], NNb, 1);
ixtmp= ixtmp(ixb(:)>0);
ixb = ixb(ixb(:)>0);
ixtmp = sub2ind(size(K), ixtmp(:), ixb(:));
ixtmp= ixtmp(ixtmp>0);
Wb = zeros(size(K));
Wb(ixtmp) =1;
Wb = Wb+ Wb';
Wb = double(Wb>0);
return;