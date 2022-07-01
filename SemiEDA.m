% Semisupervised Evolving Domain Adaptation (SemiEDA) 
function [measures,time_so_far2] = SemiEDA(Data, expt, percentage)
% Usage: measures = SemiEDA(BatchSize, expt, Data)
%
% Implements the algorithm from:
% Incremental Evolving Domain Adaptation.
% Adeleh Bitarafan, Mahdieh Soleymani Baghshah, Marzieh Gheisari.

% Required Parameters
BatchSize = expt.BatchSize;
k = expt.k;
l = expt.l;
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
% index of the labeled data in the each batch
%Labeled = randperm(BatchSize,fix(BatchSize*percentage)); 
Labeled = [4,11,19,21,26,27,30,36,48,49];  
% Compute the source subspace (PLS)
[M,~] = size(Xs');
mean_x_old = Xs(1,:)';
mean_y_old = Ys(1)';
v1 = zeros(M,1);
C = zeros(M,M);
[v1, C, Ps,mean_x_old, mean_y_old, N] = IPLS(Xs(2:end,:)',...
    Ys(2:end)', dim, mean_x_old, mean_y_old, v1, C, 1);

% fprintf(' Running Semi Supervised Evolving Domain Adaptation (SemiEDA) ');
% t_osvd = tic;
% wfig = waitbar(0, 'Semi Supervised EDA');

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
idx_Lableled = 1:BatchSize;

Y = [Ys;Yt];
YYtotal = [];
for c = 1:max(Yt)
    YYtotal = [YYtotal,Y==c];
end
YYtotal = sparse(YYtotal);
dd2 = ddtrain;
term1term2 = sparse(gamma.*dd2 + sigma.*eye(k));
    weight = exp(-l.*(1:maxT));
    weight(1)=1;
    weight(find(weight<0.6))=0;
    weight = repmat(weight,BatchSize,1);
    weight = weight(:);


t_osvd2 = tic;
for t=1:maxT
    batchIndex =(t-1).*BatchSize+1:min(t.*BatchSize,T);
    %% Pase1: Finding a linear transformation
    [~,~,Pt ] = svd(Xt(batchIndex,:)); % Compute the target subspace (PCA) 
    G = GFK([Ps,null(Ps')], Pt(:,1:dim));
    Xtt = (Xt(batchIndex,:)*G)';
    Xtt = Xtt*diag(sparse(1./sqrt(sum(Xtt.^2))));
    
    %% Pase3: Prediction by Incremental Semi-Supervised Learning (ISSL)
    nm = t.*BatchSize + length(batchIndex);
    % Compute the weight of the instances
    A = fliplr(weight(1:nm-length(batchIndex))')';
    A = [A;zeros(length(batchIndex),1)];

    E = zeros(nm,1);
    E(idx_Lableled)=1;
    E = diag(sparse(E));
    y = E*YYtotal(1:length(E),:);
    
    % Updating to compute eigenvectors of graph Laplacian
    uu_t = Inceign(Xtt',k,bins,g,ii,jj);
    uutest = [uutest;uu_t];
    uu2 = [uutrain;uutest];
    
    lambda = zeros(nm,1);
    lambda(idx_Lableled) = 1;
    lambda = lambda.*A;
    
nEmpty = length(find(A==0));
uu2(1:nEmpty,:)=[];
y(1:nEmpty,:)=[];
lambda(1:nEmpty)=[];
    
    lambda = uu2'*diag(lambda);
    alpha2=(term1term2 +lambda*uu2)\(lambda*y);
    alpha2 =uu2*alpha2;
    [~,Cls2] = max(alpha2,[],2);
    pred(batchIndex) = Cls2(end-(length(batchIndex)-1):end);
    
    %% Pase2: Updating the source domain subspace
    temp_Xt = [temp_Xt ;Xt(batchIndex,:)];
    YPLS = pred(batchIndex,1);
    
    if t~=maxT
        YPLS(Labeled)=Yt((t-1).*BatchSize+Labeled);
    end
    temp_Yt = [temp_Yt ;YPLS];
    [v1, C, Ps, mean_x_old, mean_y_old, N] = IPLS(Xt(batchIndex,:)',...
        YPLS',dim, mean_x_old, mean_y_old, v1, C, N);
%     for i=1:5%fix(BatchSize/10);
%         [v1, C, Ps, mean_x_old, mean_y_old, N] = IPLS_remove(temp_Xt(1,:)',...
%             temp_Yt(1,:)',dim, mean_x_old, mean_y_old, v1, C, N);
%         temp_Xt = temp_Xt(2:end,:);
%         temp_Yt = temp_Yt(2:end,:);
%     end
    idx_Lableled = [idx_Lableled,t.*BatchSize+Labeled];
    
%     time_so_far = toc(t_osvd);
%     iter = t*BatchSize/BatchSize;
%     max_iter = T / BatchSize;
%     expected_time = time_so_far / iter * max_iter;
%     waitbar(t*BatchSize/T, wfig, sprintf('Semi Supervised EDA %2.1f/%2.1f (s)', time_so_far, expected_time));
% 
%     if mod(t*BatchSize/BatchSize, 5) == 0
%         fprintf('.');
%     end
end
%close(wfig);
time_so_far2 = toc(t_osvd2);
measures=evaluateResult(Yt, pred, BatchSize);

