clc;
clear all;
startup;

% Set parameters for experimental setup
expt.BatchSize = 50;           % how many images to consider per time step                              
expt.k = 100;                  % the number of eigenvectors
% spam=1000  other = 100
expt.sig = 0.2;                % the scaling term
expt.l=0.01;                   % forgetting factor
expt.gamma = 1;                % regularization parameter
expt.sigma = 0.1;              % regularization parameter         

% Load Data
load('Datasets/car.mat');
Data =LoadSpamData(expt.BatchSize,dataset);
Xs = Data.Xs;
Ys = Data.Ys;
Xt = Data.Xt;
Yt = Data.Yt; 
% Supervised Evolving Domain Adaptation (SEDA) 
%  [measures,time_so_far]  = SEDA(Data, expt);
% fprintf('\n Accuracy= %2.2f\n Precision= %2.2f\n Kappa= %2.2f\n',...
%     measures.totalAccuracy, measures.totalPrecision, measures.totalKappa);
% time_so_far*1000
% return
% % % Semisupervised Evolving Domain Adaptation (SemiEDA) 
% percentage = 0.2; %percentage of labeled instances
% [measures,time_so_far] = SemiEDA(Data, expt, percentage);
% fprintf('\n Accuracy= %2.2f\n Precision= %2.2f\n Kappa= %2.2f\n',...
%     measures.totalAccuracy, measures.totalPrecision, measures.totalKappa);
% time_so_far*1000
% return
% Un Supervised Evolving Domain Adaptation (EDA) 
 expt.BatchSize = 2;
 [measures,time_so_far] = EDA(Data, expt);
fprintf('\n Accuracy= %2.2f\n Precision= %2.2f\n Kappa= %2.2f\n',...
    measures.totalAccuracy, measures.totalPrecision, measures.totalKappa);
time_so_far*1000