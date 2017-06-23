% zerocross zero crossings in x
% [n] = zc(x) calculates the number of zero crossings in x

function [z] = zerocross(x)
s=sign(x);
t=filter([1 1],1,s);
z = t==0;