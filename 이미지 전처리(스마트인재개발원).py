#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt 
import os


# In[2]:


# 파일 경로 설정 및 이미지 이름만 가져오는 배열 생성
dir_path = "./first_img"
img_file =[]
for (root, directories, files) in os.walk(dir_path):
    for file in files:
        file_path = os.path.join(root, file)
        img_file.append(file)


# In[3]:


img_file


# In[16]:


# 이미지 불러오기
for i in img_file:
    img = cv2.imread(f'./first_img/{i}', cv2.IMREAD_COLOR)

# 블러처리를 위한 마스크 생성
    k = np.array([[1,1,1],[1,1,1],[1,1,1]]) * (1/9)
# 미디언 블러 처리
    blur = cv2.filter2D(img, -1, k)

#merged = np.hstack((img,blur))  # 기존 이미지와 미디언 블러 처리한 이미지 둘다 보여 줌 // 이부분은 주석 처리하면 될듯 

# Edge Dectect Canny
    canny = cv2.Canny(img, 30, 100)
    cv2.imwrite(f'./first_img/canny/{i}', canny)


# # 캐니가 가장 높아 여기까지의 코드만 사용하면 될 것 같습니다 !

# In[17]:


# 회색조 처리
for i in img_file:
    img = cv2.imread(f'./first_img/canny/{i}', cv2.IMREAD_GRAYSCALE)

    # 임계값 처리
    ret, thresh = cv2.threshold(img,127,255, cv2.THRESH_BINARY)
    img = thresh

    # 흑백 반전
    inverted_image = cv2.bitwise_not(img)
    img = inverted_image

    # 사이즈 변환 512 * 512
    img = cv2.resize(img,(512,512))

    #정규화
    img = cv2.normalize(img, None, alpha=0,beta=230, norm_type=cv2.NORM_MINMAX)


    # RGB로 다시 변경
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
    
    cv2.imwrite(f'./first_img/cvtColor//{i}', img)


# In[18]:


# 회색조 처리
for i in img_file:
    img = cv2.imread(f'./first_img/canny/{i}', cv2.IMREAD_GRAYSCALE)

    # 임계값 처리
    ret, thresh = cv2.threshold(img,127,255, cv2.THRESH_BINARY)
    img = thresh

    # 흑백 반전
    inverted_image = cv2.bitwise_not(img)
    img = inverted_image

    # 사이즈 변환 512 * 512
    img = cv2.resize(img,(512,512))

    #정규화
    img = cv2.normalize(img, None, alpha=0,beta=230, norm_type=cv2.NORM_MINMAX)


    # RGB로 다시 변경
    
    cv2.imwrite(f'./first_img/nomalization/{i}', img)


# In[1]:


# img_label = ["36.0","35.5","35.0","34.5","34.0","33.5","33.0","32.5","32.0","31.5","31.0","30.5","30.0",
#                 "29.5","29.0","28.5","28.0","27.5","27.0","26.5","26.0","25.5","25.0","24.5","24.0",
#                 "23.5","23.0","22.5","22.0","21.5","21.0","20.5","20.0","19.5","19.0","18.5",
#                 "18.0","17.5","17.0","16.5","16.0","15.5","15.0","14.5","14.0","13.5","13.0",
#                 "12.5","12.0","11.5","11.0","10.5","10.0","9.5","9.0","8.5","8.0","7.5","7.0",
#                 "6.5","6.0","5.5","5.0","4.5","4.0","3.5","3.0","2.5","2.0","1.5","1.0","0.5",]


# In[21]:


# for i in range(len(img_label)):
#     img_dir = os.path.join('./image_list', f'{img_label[i]}')
#     os.makedirs(img_dir)

