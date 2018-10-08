THUMOS CHALLENGE 2014
http://crcv.ucf.edu/THUMOS14/index.html

-------------------------------------------------------------------

This directory contains Matlab functions designed for the evaluation
of temporal action detection in the THUMOS 2014 action detection challenge.

The data and the description of the THUMOS 2014 is available from
http://crcv.ucf.edu/THUMOS14/ In particular, the current package assumes
detection results in a format specified in the Evaluation Setup document
available from http://crcv.ucf.edu/THUMOS14/THUMOS14_Evaluation.pdf

Example:

% running evaluation (simulated detection results)
[pr_all,ap_all,map]=TH14evaldet('results/Run-2-det.txt','groundtruth','val');

% plotting precision-recall results
overlapthresh=0.1;
ind=find([pr_all.overlapthresh]==overlapthresh);
clf
for i=1:length(ind)
  subplot(4,5,i)
  pr=pr_all(ind(i));
  plot(pr.rec,pr.prec)
  axis([0 1 0 1])
  title(sprintf('%s AP:%1.3f',pr.class,pr.ap))
end 
