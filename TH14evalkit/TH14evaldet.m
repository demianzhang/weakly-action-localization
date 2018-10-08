function [pr_all,ap_all,map]=TH14evaldet(detfilename,gtpath,subset)

% [pr_all,ap_all,map]=TH14evaldet(detfilename,gtpath,subset)
%
% Evaluation of the temporal detection for 20 classes in the THUMOS 2014 
% action detection challenge http://crcv.ucf.edu/THUMOS14/
%
% The function produces precision-recall curves and average precision
% values for each action class and five values of thresholds for
% the overlap  between ground-truth action intervals and detected action
% intervals. Mean average precision values over classes are also returned.
%
% Example:
% 
%  [pr_all,ap_all,map]=TH14evaldet('results/Run-2-det.txt','groundtruth','val');
%  
% Plotting precision-recall results:
% 
%  overlapthresh=0.1;
%  ind=find([pr_all.overlapthresh]==overlapthresh);
%  clf
%  for i=1:length(ind)
%    subplot(4,5,i)
%    pr=pr_all(ind(i));
%    plot(pr.rec,pr.prec)
%    axis([0 1 0 1])
%    title(sprintf('%s AP:%1.3f',pr.class,pr.ap))
%  end 
%
  

  
% THUMOS14 detection classes
%
  
[th14classids,th14classnames]=textread([gtpath '/detclasslist.txt'],'%d%s');
  
% read ground truth
%

clear gtevents
gteventscount=0;
th14classnamesamb=cat(1,th14classnames,'Ambiguous');
for i=1:length(th14classnamesamb)
  class=th14classnamesamb{i};
  gtfilename=[gtpath '/' class '_' subset '.txt'];
  if exist(gtfilename,'file')~=2
    error(['TH14evaldet: Could not find GT file ' gtfilename])
  end
  [videonames,t1,t2]=textread(gtfilename,'%s%f%f');
  for j=1:length(videonames)
    gteventscount=gteventscount+1;
    gtevents(gteventscount).videoname=videonames{j};
    gtevents(gteventscount).timeinterval=[t1(j) t2(j)];
    gtevents(gteventscount).class=class;
    gtevents(gteventscount).conf=1;
  end
end


% parse detection results
%

if exist(detfilename,'file')~=2
  error(['TH14evaldet: Could not find file ' detfilename])
end

[videonames,t1,t2,clsid,conf]=textread(detfilename,'%s%f%f%d%f');
videonames=regexprep(videonames,'\.mpeg','');

clear detevents
for i=1:length(videonames)
  ind=find(clsid(i)==th14classids);
  if length(ind)
    detevents(i).videoname=videonames{i};
    detevents(i).timeinterval=[t1(i) t2(i)];
    detevents(i).class=th14classnames{ind};
    detevents(i).conf=conf(i);
  else
    fprintf('WARNING: Reported class ID %d is not among THUMOS14 detection classes.\n')
  end
end

% Evaluate per-class PR for multiple overlap thresholds
%

ap_all=[];
clear pr_all
overlapthreshall=.1:.1:.5;

for i=1:length(th14classnames)
  class=th14classnames{i};
  classid=strmatch(class,th14classnames,'exact');
  assert(length(classid)==1);

  for j=1:length(overlapthreshall)
    overlapthresh=overlapthreshall(j);
    [rec,prec,ap]=TH14eventdetpr(detevents,gtevents,class,overlapthresh);
    pr_all(i,j).class=class;
    pr_all(i,j).classid=classid;
    pr_all(i,j).overlapthresh=overlapthresh;
    pr_all(i,j).prec=prec;
    pr_all(i,j).rec=rec;
    pr_all(i,j).ap=ap;
    ap_all(i,j)=ap;
    
    fprintf('AP:%1.3f at overlap %1.1f for %s\n',ap,overlapthresh,class)
  end
end

map=mean(ap_all,1);


