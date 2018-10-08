import numpy as np
import torch
from scipy.ndimage import gaussian_filter

def gen_prop(v,scores):

    # v:[int(line[i]), cam[i]]
    #    start_frame,  score
    frm_duration = len(v)
    scores = np.array(scores)

    topk_labels = label_frame_by_threshold(scores, bw=None, thresh=[.6], multicrop=False)

    bboxes = []
    #tol_lst = [0.05, .1, .2, .3, .4, .5, .6, 0.8, 1.0]
    tol_lst = [.5]

    bboxes.extend(build_box_by_search(topk_labels, np.array(tol_lst)))

    bboxes = np.array(bboxes)
    v = np.array(v)

    bboxes = temporal_nms(bboxes, 0.9)

    #pr_box = [(x[0] / float(frm_duration) * v.duration, x[1] / float(frm_duration) * v.duration) for x in bboxes]

    # filter out too short proposals
    #pr_box = list(filter(lambda b: b[1] - b[0] > args.minimum_len, pr_box))
    pr_box  = []

    for i in range(len(bboxes)):
       pr_box.append([v[int(max(bboxes[i][0]-2,0))],v[int(bboxes[i][1]-2)],bboxes[i][2]])

    return pr_box, [x[2] for x in bboxes]

def label_frame_by_threshold(score_mat, bw=None, thresh=list([0.05]), multicrop=True):
    """
    Build frame labels by thresholding the foreground class responses
    :param score_mat:
    #:param cls_lst:
    :param bw:
    :param thresh:
    :param multicrop:
    :return:
    """
    if multicrop:
        f_score = score_mat.mean(axis=1)
    else:
        f_score = score_mat

    ss = np.array(f_score)

    #ss = softmax(f_score)

    rst = []

    cls_score = ss[:] if bw is None else gaussian_filter(ss[:], bw)
    for th in thresh:
        rst.append((cls_score > th, f_score[:]))

    return rst

def build_box_by_search(frm_label_lst, tol, min=1):
    boxes = []
    for frm_labels, frm_scores in frm_label_lst:
        length = len(frm_labels)
        diff = np.empty(length+1)
        diff[1:-1] = frm_labels[1:].astype(int) - frm_labels[:-1].astype(int)
        diff[0] = float(frm_labels[0])
        diff[length] = 0 - float(frm_labels[-1])
        cs = np.cumsum(1 - frm_labels)
        offset = np.arange(0, length, 1)

        up = np.nonzero(diff == 1)[0]
        down = np.nonzero(diff == -1)[0]

        assert len(up) == len(down), "{} != {}".format(len(up), len(down))
        for i, t in enumerate(tol):
            signal = cs - t * offset
            for x in range(len(up)):
                s = signal[up[x]]
                for y in range(x + 1, len(up)):
                    if y < len(down) and signal[up[y]] > s:
                        boxes.append((up[x], down[y-1]+1, sum(frm_scores[up[x]:down[y-1]+1])))
                        break
                else:
                    boxes.append((up[x], down[-1] + 1, sum(frm_scores[up[x]:down[-1] + 1])))

            for x in range(len(down) - 1, -1, -1):
                s = signal[down[x]] if down[x] < length else signal[-1] - t
                for y in range(x - 1, -1, -1):
                    if y >= 0 and signal[down[y]] < s:
                        boxes.append((up[y+1], down[x] + 1, sum(frm_scores[up[y+1]:down[x] + 1])))
                        break
                else:
                    boxes.append((up[0], down[x] + 1, sum(frm_scores[0:down[x]+1 + 1])))

    print("len:{}".format(len(boxes)))


    return boxes



def temporal_nms(bboxes, thresh):
    """
    One-dimensional non-maximal suppression
    :param bboxes: [[st, ed, score, ...], ...]
    :param thresh:
    :return:
    """
    if len(bboxes) == 0:
        return []
    t1 = bboxes[:, 0]
    t2 = bboxes[:, 1]
    scores = bboxes[:, 2]

    durations = t2 - t1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(t1[i], t1[order[1:]])
        tt2 = np.minimum(t2[i], t2[order[1:]])
        intersection = tt2 - tt1
        IoU = intersection / (durations[i] + durations[order[1:]] - intersection).astype(float)

        inds = np.where(IoU <= thresh)[0]
        order = order[inds + 1]

    return bboxes[keep, :]

def softmax(raw_score, T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]

def temporal_iou(span_A, span_B):
    """
    Calculates the intersection over union of two temporal "bounding boxes"
    span_A: (start, end)
    span_B: (start, end)
    """
    union = min(span_A[0], span_B[0]), max(span_A[1], span_B[1])
    inter = max(span_A[0], span_B[0]), min(span_A[1], span_B[1])

    if inter[0] >= inter[1]:
        return 0
    else:
        return float(inter[1] - inter[0]) / float(union[1] - union[0])