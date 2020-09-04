import os
import cv2
from pts_loader import load
import numpy as np

def _read_landmark_pts(fn):
    '''
    read landmark pts file
    param:
        fn: input filename
    return: list of landmark coordinate [x1, x2,..., x68, y1, y2,..., y_68]
    '''
    points = load(fn)
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    flat = x + y
    return flat

def _get_bbox(xy):
    x = xy[:68]
    y = xy[-68:]
    x2 = np.max([abs(cor) for cor in x if abs(cor) != 1])
    x1 = np.min([abs(cor) for cor in x if abs(cor) != 1])
    y2 = np.max([abs(cor) for cor in y if abs(cor) != 1])
    y1 = np.min([abs(cor) for cor in y if abs(cor) != 1])
    bbox = [int(x1), int(y1), int(x2), int(y2)]
    return bbox

def _get_bbox_landmark(pts_fn, img_fn, ratio=1.2):
    '''
    get bbox and landmark
    '''
    landmark=load(pts_fn)
    landmark=np.array(landmark)
    landmark=np.abs(landmark)
    landmark[landmark == 1] = 9999
    # Lowest x, y
    xy = np.min(abs(landmark), axis=0).astype(np.int32) 
    landmark[abs(landmark) == 9999]=1
    # Highest x, yload_data
    zz = np.max(landmark, axis=0).astype(np.int32)
    # width, height of facelandmark
    wh = zz - xy + 1

    # Center of landmark
    center = (xy + wh/2).astype(np.int32)
    img = cv2.imread(img_fn)
    boxsize = int(np.max(wh)*ratio)

    # Take bbox
    xy = center - boxsize//2
    x1, y1 = xy
    x2, y2 = xy + boxsize
    height, width, _ = img.shape

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    landmark = landmark.astype(np.int32)
    # for l in landmark:
    #     cv2.circle(img, (l[0], l[1]), 2, (255, 0, 0), 0)
        
    # cv2.imshow('image landmark', img)
    # key = cv2.waitKey(0) & 0xFF
    # if key==27:
    #     cv2.destroyAllWindows()
    return (x1, y1, x2, y2), landmark
    

# _get_bbox_expand(pts_fn='left/trainset/image17036.pts', img_fn='left/trainset/image17036.jpg')
# _read_landmark_pts('frontal/trainset/image00895.pts')
