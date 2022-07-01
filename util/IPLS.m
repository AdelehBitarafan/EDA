function [v1,C,W,mean_x_old,mean_y_old,N] =...
    IPLS(X,Y,k,mean_x_old,mean_y_old,v1,C,N)

[~,n] = size(X);

% Calculate v1
for i=1:n
    N = N+1;
    NN_1= ((N-1)./N);
    N_1 = 1./N;
    mean_x_new = NN_1.*mean_x_old + N_1.*X(:,i);
    mean_y_new = NN_1.*mean_y_old + N_1.*Y(1,i);
    xnn = X(:,i)- mean_x_new;
    delta = mean_x_new - mean_x_old;
    
    v1  = v1 - (N-1).*mean_y_old*delta + Y(1,i).*xnn;
   
    C = NN_1.*C + NN_1.*(delta*delta') + N_1.*(xnn*xnn');
        
    mean_x_old = mean_x_new;
    mean_y_old = mean_y_new;
end

V(:,1) = v1;
w = v1;

for i=2:k
    w = C*w;
    V(:,i) = w;
    for j=1:i-1       
        V(:,i) = V(:,i) - V(:,i)'*(V(:,j)/norm(V(:,j))).*(V(:,j)/norm(V(:,j)));
    end
end
W = normalized(V,k);
%W = NormalizeData(V, 'l2_zscore');
