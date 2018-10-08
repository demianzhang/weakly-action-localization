function [rec,prec,ap]=TH14eventclspr(conf,labels)


[so,sortind]=sort(-conf);
tp=labels(sortind)==1;
fp=labels(sortind)~=1;
npos=length(find(labels==1));

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);

% compute average precision

ap=0;
recallpoints=0:0.1:1;
for t=recallpoints
    p=max(prec(rec>=t));
    if isempty(p)
        p=0;
    end
    ap=ap+p/length(recallpoints);
end
