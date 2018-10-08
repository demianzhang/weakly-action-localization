function [rec,prec,ap]=TH14eventdetpr(detevents,gtevents,class,overlapthresh)

  
gtvideonames={gtevents.videoname};
detvideonames={detevents(:).videoname};
videonames=unique(cat(2,gtvideonames,detvideonames));

tpconf=[];
fpconf=[];
npos=length(strmatch(class,{gtevents.class},'exact'));
assert(npos>0)

indgtclass=strmatch(class,{gtevents.class},'exact');
indambclass=strmatch('Ambiguous',{gtevents.class},'exact');
inddetclass=strmatch(class,{detevents.class},'exact');

for i=1:length(videonames)
  gt=gtevents(intersect(strmatch(videonames{i},gtvideonames,'exact'),indgtclass));
  amb=gtevents(intersect(strmatch(videonames{i},gtvideonames,'exact'),indambclass)); 
  det=detevents(intersect(strmatch(videonames{i},detvideonames,'exact'),inddetclass));
  
  if length(det)
  
    [vs,is]=sort(-[det(:).conf]);
    det=det(is);
    conf=[det(:).conf];
    indfree=ones(1,length(det));
    indamb=zeros(1,length(det));

    % interesct event detection intervals with GT
    if length(gt)
      ov=intervaloverlapvalseconds(cat(1,gt(:).timeinterval),cat(1,det(:).timeinterval));
      for k=1:size(ov,1)
	ind=find(indfree);
	[vm,im]=max(ov(k,ind));
	if vm>overlapthresh
	  indfree(ind(im))=0;
	end
      end
    end
    
    % respect ambiguous events (overlapping detections will be removed from the FP list)
    if length(amb)
      ovamb=intervaloverlapvalseconds(cat(1,amb(:).timeinterval),cat(1,det(:).timeinterval));
      indamb=sum(ovamb,1);
    end
    
    tpconf=[tpconf conf(find(indfree==0))];
    %fpconf=[fpconf conf(find(indfree==1))];
    fpconf=[fpconf conf(find(indfree==1 & indamb==0))];
  end
end

%%disp(size(tpconf))
conf=[tpconf fpconf; 1*ones(size(tpconf)) 2*ones(size(fpconf))];
if size(tpconf,2)==0
    ap = 0;
    rec = 0;
    prec = 0;
    return;
end
[vs,is]=sort(-conf(1,:));
tp=cumsum(conf(2,is)==1);
fp=cumsum(conf(2,is)==2);
rec=tp/npos;
prec=tp./(fp+tp);
ap=prap(rec,prec);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ap=prap(rec,prec)

  
ap=0;
recallpoints=0:0.1:1;
for t=recallpoints
    p=max(prec(rec>=t));
    if isempty(p)
        p=0;
    end
    ap=ap+p/length(recallpoints);
end


