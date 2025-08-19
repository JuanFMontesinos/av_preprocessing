# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Juan Montesinos to adapt to Numpy 2.0
# --------------------------------------------------------

import numpy as np
cimport numpy as cnp

# ensure NumPy C-API is initialized (in your setup.py/build.py add include_dirs=[np.get_include()])
# and at module init time call cnp.import_array() if you split into .pxd; not needed in simple .pyx

cdef inline cnp.float32_t fmax(cnp.float32_t a, cnp.float32_t b):
    return a if a >= b else b

cdef inline cnp.float32_t fmin(cnp.float32_t a, cnp.float32_t b):
    return a if a <= b else b

def cpu_nms(cnp.ndarray[cnp.float32_t, ndim=2] dets, float thresh):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] x1 = dets[:, 0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] y1 = dets[:, 1]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] x2 = dets[:, 2]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] y2 = dets[:, 3]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] scores = dets[:, 4]

    cdef cnp.ndarray[cnp.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # argsort returns pointer-sized ints -> intp_t
    cdef cnp.ndarray[cnp.intp_t, ndim=1] order = scores.argsort()[::-1]

    cdef Py_ssize_t ndets = dets.shape[0]
    cdef cnp.ndarray[cnp.intp_t, ndim=1] suppressed = np.zeros((ndets,), dtype=np.intp)

    # nominal indices
    cdef Py_ssize_t _i, _j
    # sorted indices
    cdef Py_ssize_t i, j
    # temp variables for box i (the box currently under consideration)
    cdef cnp.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef cnp.float32_t xx1, yy1, xx2, yy2
    cdef cnp.float32_t w, h
    cdef cnp.float32_t inter, ovr

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]; iy1 = y1[i]; ix2 = x2[i]; iy2 = y2[i]; iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = fmax(ix1, x1[j])
            yy1 = fmax(iy1, y1[j])
            xx2 = fmin(ix2, x2[j])
            yy2 = fmin(iy2, y2[j])
            w = fmax(0.0, xx2 - xx1 + 1)
            h = fmax(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return keep

def cpu_soft_nms(cnp.ndarray[cnp.float32_t, ndim=2] boxes,
                 float sigma=0.5, float Nt=0.3, float threshold=0.001,
                 unsigned int method=0):
    cdef unsigned int N = <unsigned int>boxes.shape[0]
    cdef float iw, ih, box_area
    cdef float ua
    cdef unsigned int pos = 0
    cdef float maxscore = 0
    cdef unsigned int maxpos = 0
    cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov
    cdef unsigned int i

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]; ty1 = boxes[i,1]; tx2 = boxes[i,2]; ty2 = boxes[i,3]; ts = boxes[i,4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        tx1 = boxes[i,0]; ty1 = boxes[i,1]; tx2 = boxes[i,2]; ty2 = boxes[i,3]; ts = boxes[i,4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]; y1 = boxes[pos, 1]; x2 = boxes[pos, 2]; y2 = boxes[pos, 3]; s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # IoU between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight * boxes[pos, 4]

                    # if box score falls below threshold, discard by swapping with last box and shrinking N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    return [i for i in range(N)]
