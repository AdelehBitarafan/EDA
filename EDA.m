% Unsupervised Evolving Domain Adaptation (EDA) 
function [measures,time_so_far2] = EDA(Data, expt)

% Usage: measures = EDA(BatchSize, expt, Data)
%
% Implements the algorithm from:
% Incremental Evolving Domain Adaptation.
% Adeleh Bitarafan, Mahdieh Soleymani Baghshah, Marzieh Gheisari.

% Required Parameters
nLabeled = 50; % the number of labeled training data
BatchSize = expt.BatchSize;
k = expt.k;
gamma = expt.gamma;
sigma = expt.sigma;
sig = expt.sig;                                         

Xs = Data.Xs;
Ys = Data.Ys;
Xt = Data.Xt;
Yt = Data.Yt;                 
T = size(Xt, 1);
if size(Xs,2)<100
    dim = fix(size(Xs,2)/2);  % the number of subspace dimension 
else
    dim = 100;
end
                         
% Initialize
pred = zeros(T,1);
maxT=ceil(T/BatchSize);
temp_Xt = Xs;
temp_Yt = Ys;
 
% Compute the source subspace (PLS)
[M,N] = size(Xs');
mean_x_old = Xs(1,:)';
mean_y_old = Ys(1)';
v1 = zeros(M,1);
C = zeros(M,M);
[v1, C, Ps,mean_x_old, mean_y_old, N] = IPLS(Xs(2:end,:)',...
    Ys(2:end)', dim, mean_x_old, mean_y_old, v1, C, 1);

% fprintf(' Running Unsupervised Evolving Domain Adaptation (EDA) ');
% t_osvd = tic;
% wfig = waitbar(0, 'Unsupervised EDA');

X = (Xs)';
X = X*diag(sparse(1./sqrt(sum(X.^2))));
clear bins g lambdas
% Compute approximate numerical values of eigenfunctions
for ddim=1:size(X,1)
    [bins(:,ddim),g(:,:,ddim),lambdas(:,ddim),pp]=numericalEigenFunctions(X(ddim,:)',sig);
end
% Compute approximate eigenvectors of graph Laplacian of train set
[ddtrain,uutrain,ii,jj] = eigenfunctionsIncremental(X',g,lambdas,k,bins);
uutest = []; 

Y = [Ys;Yt];
YYtotal = [];
for c = 1:max(Yt)
    YYtotal = [YYtotal,Y==c];
end
YYtotal = sparse(YYtotal);
dd2 = ddtrain;
term1term2 = sparse(gamma.*dd2 + sigma.*eye(k));
n = nLabeled; 
E = diag(sparse([ones(n,1);zeros(T,1)]));
ytotal = E*YYtotal;
  
lambda = zeros(n+T,1);
lambda(1:nLabeled) = 1;
Lambda_total = diag(sparse(lambda));
%Ps = princomp(Xs);
   
t_osvd2 = tic;
for t=1:maxT
    batchIndex =(t-1).*BatchSize+1:min(t.*BatchSize,T);
    %% Pase1: Finding a linear transformation 
   [~,~,Pt ] = svd(Xt(batchIndex,:));  % Compute the target subspace (PCA)
  %  Pt = princomp(Xt(batchIndex,:));     
    G = fastGFK([Ps,null(Ps')], Pt(:,1:dim));
    Xtt = (Xt(batchIndex,:)*G)';
    Xtt = Xtt*diag(sparse(1./sqrt(sum(Xtt.^2))));
    
    
    %% Pase3: Prediction by Incremental Semi-Supervised Learning (ISSL)
    m = (t-1).*BatchSize + length(batchIndex); % The number of test data
    Lambda = Lambda_total(1:n+m,1:n+m);
    % Updating to compute eigenvectors of graph Laplacian
    uu_t = Inceign(Xtt',k,bins,g,ii,jj);
    uutest = [uutest;uu_t];
    uu2 = [uutrain;uutest];
    
% nEmpty = (t-100)+n+1;
% if nEmpty<0
%     nEmpty = n+1;
% end
% uu2(n+1:nEmpty,:)=[];
% y(n+1:nEmpty,:)=[];
% Lambda(n+1:nEmpty,n+1:nEmpty)=[];

    Lambda = uu2'*Lambda;
    alpha2=(term1term2 +Lambda*uu2)\(Lambda*ytotal(1:n+m,:));

    alpha2 =uu2*alpha2;
    [~,Cls2] = max(alpha2,[],2);
    pred(batchIndex) = Cls2(end-(length(batchIndex)-1):end);
    
    %% Pase2: Updating the source domain subspace
    temp_Xt = [temp_Xt ;Xt(batchIndex,:)];
    temp_Yt = [temp_Yt ;pred(batchIndex,1)];
    [v1, C, Ps, mean_x_old, mean_y_old, N] = IPLS(Xt(batchIndex,:)',...
        pred(batchIndex,1)',dim, mean_x_old, mean_y_old, v1, C, N);
    
%     time_so_far = toc(t_osvd);
%     iter = t*BatchSize/BatchSize;
%     max_iter = T / BatchSize;
%     expected_time = time_so_far / iter * max_iter;
%     waitbar(t*BatchSize/T, wfig, sprintf('Unsupervised EDA %2.1f/%2.1f (s)', time_so_far, expected_time));
% 
%     if mod(t*BatchSize/BatchSize, 50) == 0
%         fprintf('.');
%     end
end
%close(wfig);
time_so_far2 = toc(t_osvd2);
measures=evaluateResult(Yt, pred, BatchSize);
