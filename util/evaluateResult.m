function measures=evaluateResult(Labels,PredictedLabels, batchSize)
n=size(Labels,1);
T=ceil(n/batchSize);
measures.accuracy=zeros(T-1,1);
totalNumber=0;
trueNumber=0;
alpha=1;

for t=1:T-1
index=(t)*batchSize+1:min((t+1)*batchSize,n);
totalNumber=totalNumber*alpha+length(index);
trueNumber=trueNumber*alpha+sum(Labels(index)==PredictedLabels(index));
measures.accuracy(t)=trueNumber/totalNumber*100;
end
measures.totalAccuracy=mean(Labels==PredictedLabels)*100;
measures.totalPrecision= sum(PredictedLabels==2 & Labels==PredictedLabels)/sum(PredictedLabels==2)*100;
q=sum(Labels==Labels(1))/length(Labels);
pStar=q^2+(1-q)^2;
measures.totalKappa=evaluate_kappa(PredictedLabels, Labels);
end
