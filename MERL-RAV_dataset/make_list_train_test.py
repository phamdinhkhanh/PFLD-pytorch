import cv2
import os
import numpy as np
import glob2
import numpy as np
from pts_loader import load

GROUP_POSE = {
    'frontal': 0,
    'left': 1,
    'lefthalf': 2,
    'right': 3,
    'righthalf': 4
}

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

def load_data(outFile, isTrain=True):
    all_files = glob2.glob('./**/*.jpg')
    if isTrain:
        all_files = [fn for fn in all_files if 'trainset' in fn]
    else:
        all_files = [fn for fn in all_files if 'testset' in fn]
    for fn in all_files:
        pts_fn = fn[:-4]+'.pts'
        points = _read_landmark_pts(pts_fn)
        # Get bbox
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
        img = cv2.imread(fn)
        boxsize = int(np.max(wh)*1.2)

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

        bbox = [x1, y1, x2, y2]
        # Get pose embedding
        pose = fn.split('/')[2]
        idx_pose  = GROUP_POSE[pose]
        embed_pose = np.zeros(5, dtype=np.int8)
        embed_pose[idx_pose] = 1
        embed_pose = list(embed_pose)
        # concatenate
        # landmark, bbox, one-hot pose, file_path
        landmark_points = landmark.reshape(1, -1)[0]
        landmark_points = list(landmark_points)
        all_attr = []
        all_attr += landmark_points
        all_attr += bbox
        all_attr += embed_pose
        all_attr += [fn]
        all_attr = [str(attr) for attr in all_attr]
        line = ' '.join(all_attr)

        # write line
        with open(outFile, 'a+') as f:
            f.write(line+'\n')

if not os.path.exists('aflw_annotations/list_68pt_rect_attr_train_test'):
    os.makedirs('aflw_annotations/list_68pt_rect_attr_train_test', exist_ok=True)

# load_data(outFile='./aflw_annotations/list_68pt_rect_attr_train_test/list_68pt_rect_attr_test.txt', isTrain=False)
load_data(outFile='./aflw_annotations/list_68pt_rect_attr_train_test/list_68pt_rect_attr_train.txt', isTrain=True)
