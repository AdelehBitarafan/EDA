function V = normalized(V,dim)
for i=1:dim
    V(:,i) = V(:,i)./norm(V(:,i));
end
 
