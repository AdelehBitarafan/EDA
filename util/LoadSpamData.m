function Data =LoadSpamData(BatchSize,dataset)

X = dataset.data;
Y = dataset.labels;

fts = X;
fts = zscore(fts,1);

Data.Xs = fts(1:BatchSize,:);
Data.Ys = Y(1:BatchSize,:);
Data.Xt = fts(BatchSize+1:end,:);
Data.Yt = Y(BatchSize+1:end,:);
  