import os
import numpy as np
import cv2

with open("./train_data/list.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

i = np.random.randint(100)
line = lines[i]
line = line.split()

img_path = line[0]
landmark = line[1:137]
attr = line[137:142]
angle = line[142:145]
print(img_path)
print(landmark)
print(attr)
print(angle)

img = cv2.imread(img_path)
img = cv2.resize(img, (112*5, 112*5), interpolation=cv2.INTER_LINEAR)
width, heigh, _ = img.shape
landmark = [float(l) for l in landmark]
landmark = np.array(landmark)*width
landmark = landmark.astype(np.int32)
landmark = landmark.reshape(-1, 2)
for l in landmark:
    img=cv2.circle(img, (l[0], l[1]), 1, (0, 255, 0), 0)

cv2.imshow('image', img)
key = cv2.waitKey(0) & 0xFF
if key == 27:
    cv2.destroyAllWindow()