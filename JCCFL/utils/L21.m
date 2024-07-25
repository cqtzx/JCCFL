function [va]=L21(M)
[m,n] = size(M);
s = zeros(n,1);
for i=1:n
    s(i) = sum(M(:,i).^2);
end
va = sum(sqrt(s));
end