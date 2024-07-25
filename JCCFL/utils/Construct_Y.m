function Y = Construct_Y(gnd,num_l)
%%
% gnd:标签向量；
% num_l:表示有标签样本的数目；
% Y:生成的标签矩阵；
nClass = length(unique(gnd));
Y = zeros(nClass,length(gnd));
for i = 1:num_l
    for j = 1:nClass
        if j == gnd(i)
            Y(j,i) = 1;
        end  
    end
end
end