function B = Construct_B(Y)
%%
B = Y;
B(Y==0) = -1;
end