function [v1,C,W,mean_x_new,mean_y_new,N] =...
    IPLS_remove(X,Y,k,mean_x_old,mean_y_old,v1,C,N)

n = N;
nn_1 = n./(n - 1);
n_1 = 1./(n-1);
mean_x_new = nn_1.*mean_x_old - n_1.*X;
mean_y_new = nn_1.*mean_y_old - n_1.*Y;
xnn = X- mean_x_new;
delta = mean_x_new - mean_x_old;

v1  = v1 - n.*mean_y_old*delta - Y.*xnn;
C = nn_1.*C + nn_1.*(delta*delta') -n_1.*(xnn*xnn');
      
N = N-1;

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