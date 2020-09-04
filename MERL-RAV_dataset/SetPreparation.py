#-*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from pfld.utils import calculate_pitch_yaw_roll
debug = False

def rotate(angle, center, landmark):
    '''
    param: 
        angle: rotation angle
        center: center of image as pin point to rotate
    return: landmark and matrix rotation
    '''
    # Randomly rotate landmark
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    # Rotation matrix
    M = np.zeros((2,3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1-alpha)*center[0] - beta*center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta*center[0] + (1-alpha)*center[1]

    landmark_ = np.asarray([(M[0,0]*x+M[0,1]*y+M[0,2],
                             M[1,0]*x+M[1,1]*y+M[1,2]) for (x,y) in landmark])
    return M, landmark_

class ImageDate():
    def __init__(self, line, imgDir, image_size=112):
        '''
        param:
            line: line of annotation as description bellow
            imgDir: folder image save file
        '''
        self.image_size = image_size
        line = line.strip().split()
        #0-135: landmark 坐标点  136-139: bbox 坐标点;
        #140: (frontal): 1 yes, 0 no
        #141: (left)   : 1 yes, 0 no
        #142: (lefthalf): 1 yes, 0 no
        #143: (right)   : 1 yes, 0 no
        #144: (righthalf): 1 yes, 0 no
        #145: (image name)
        assert(len(line) == 146)
        self.list = line
        self.landmark = np.asarray(list(map(float, line[:136])), dtype=np.float32).reshape(-1, 2)
        self.box = np.asarray(list(map(int, line[136:140])),dtype=np.int32)
        flag = list(map(int, line[140:145]))
        flag = list(map(bool, flag))
        self.frontal = flag[0]
        self.left = flag[1]
        self.lefthalf = flag[2]
        self.right = flag[3]
        self.righthalf = flag[4]
        self.path = os.path.join(imgDir, line[145])
        self.img = None
        self.self_occluded = np.where(self.landmark[:, 0] == 1)
        self.imgs = []
        self.landmarks = []
        self.boxes = []

    def change_brightness(self, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v,value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
        
    def load_data(self, is_train, repeat, mirror=None):
        # if (mirror is not None):
            # with open(mirror, 'r') as f:
            #     lines = f.readlines()
            #     assert len(lines) == 1
                # mirror_idx = lines[0].strip().split(',')
                # mirror_idx = list(map(int, mirror_idx))
        # Lowest x, y
        xy = np.min(self.landmark, axis=0).astype(np.int32) 
        # Highest x, yload_data
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        # width, height of facelandmark
        wh = zz - xy + 1

        # Center of landmark
        center = (xy + wh/2).astype(np.int32)
        img = cv2.imread(self.path)
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
        
        # Show landmark
        imgT = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for x, y in (self.landmark+0.5).astype(np.int32):
                cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
            cv2.imshow('0', imgTT)
            if cv2.waitKey(0) == 27:
                exit()
        # Resize face crop
        imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        # Normalize landmark according to boxsize (value between 0 and 1)
        landmark = (self.landmark - xy)/boxsize
        assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        self.imgs.append(imgT)
        self.landmarks.append(landmark)

        if is_train:
            while len(self.imgs) < repeat:
                # Randomly rotate 30 degree
                angle = np.random.randint(-30, 30)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize*0.1, boxsize*0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                M, landmark = rotate(angle, (cx,cy), self.landmark)

                # Rotate Image and scale 1.1 times
                imgT = cv2.warpAffine(img, M, (int(img.shape[1]*1.1), int(img.shape[0]*1.1)))

                # Change Image bright randomly
                imgT = self.change_brightness(imgT, value=np.random.randint(-80, 80))

                # Width and heigth of facelandmark
                wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
                # Randomly pick up size
                size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                # Take x, y leftlower point
                xy = np.asarray((cx - size // 2, cy - size//2), dtype=np.int32)
                # Normalize landmark
                landmark = (landmark - xy) / size
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if (dx > 0 or dy > 0 or edx >0 or edy > 0):
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
                
                # Rescale image into standard image size
                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                # Randomly flip according to horizon
                if np.random.choice((True, False)):
                    landmark[:,0] = 1 - landmark[:,0]
                    # landmark = landmark[mirror_idx]
                    imgT = cv2.flip(imgT, 1)
                
                # Update self occluded position
                landmark[self.self_occluded, 0] = 0
                landmark[self.self_occluded, 1] = 0
                self.imgs.append(imgT)
                self.landmarks.append(landmark)

    def save_data(self, path, prefix):
        # Save attribute
        attributes = [self.frontal, self.left, self.lefthalf, self.right, self.righthalf]
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))
        labels = []
        # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
        for i, (img, lanmark) in enumerate(zip(self.imgs, self.landmarks)):
            assert lanmark.shape == (68, 2)
            save_path = os.path.join(path, prefix+'_'+str(i)+'.png')
            assert not os.path.exists(save_path), save_path
            # Save image
            cv2.imwrite(save_path, img)

            # save TRACKED_POINTS
            euler_angles_landmark = []
            for index in TRACKED_POINTS:
                euler_angles_landmark.append(lanmark[index])
            euler_angles_landmark = np.asarray(euler_angles_landmark).reshape((-1, 28))
            pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0])
            euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)
            euler_angles_str = ' '.join(list(map(str, euler_angles)))

            landmark_str = ' '.join(list(map(str,lanmark.reshape(-1).tolist())))

            label = '{} {} {} {}\n'.format(save_path, landmark_str, attributes_str, euler_angles_str)

            labels.append(label)
        return labels
def get_dataset_list(imgDir, outDir, landmarkDir, is_train):
    '''
    Write image and landmark annotation to folder
    param:
        imgDir: string, input image directory
        outDir: string, output image directory
        landmarkDir: string, landmark annotation file
        is_train: boolean, is train or test dataset
    '''
    with open(landmarkDir,'r') as f:
        lines = f.readlines()
        labels = []
        save_img = os.path.join(outDir, 'imgs')
        if not os.path.exists(save_img):
            os.mkdir(save_img)

        if debug:
            lines = lines[:100]
        for i, line in enumerate(lines):
            Img = ImageDate(line, imgDir)
            img_name = Img.path
            Img.load_data(is_train, 10)
            _, filename = os.path.split(img_name)
            filename, _ = os.path.splitext(filename)
            label_txt = Img.save_data(save_img, str(i)+'_' + filename)
            labels.append(label_txt)
            if ((i + 1) % 100) == 0:
                print('file: {}/{}'.format(i+1, len(lines)))

    with open(os.path.join(outDir, 'list.txt'),'w') as f:
        for label in labels:
            f.writelines(label)

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    imageDirs = '.'
    # Mirror_file = 'WFLW/WFLW_annotations/Mirror98.txt'

    landmarkDirs = ['aflw_annotations/list_68pt_rect_attr_train_test/list_68pt_rect_attr_test.txt',
                    'aflw_annotations/list_68pt_rect_attr_train_test/list_68pt_rect_attr_train.txt']

    outDirs = ['test_data', 'train_data']
    for landmarkDir, outDir in zip(landmarkDirs, outDirs):
        outDir = os.path.join(root_dir, outDir)
        print(outDir)
        if os.path.exists(outDir):
            shutil.rmtree(outDir)
        os.mkdir(outDir)
        if 'list_98pt_rect_attr_test.txt' in landmarkDir:
            is_train = False
        else:
            is_train = True
        imgs = get_dataset_list(imageDirs, outDir, landmarkDir, is_train)
    print('end')