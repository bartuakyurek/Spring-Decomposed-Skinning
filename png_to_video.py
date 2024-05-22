# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:50:18 2020

@author: Haoran6
"""

import cv2
import os
from os.path import isfile, join


def png2video(pathIn, fps=30):
    pathOut = 'video.avi'
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    
    if files[0] == '.DS_Store':
        files = files[1:-1]
        
    files.sort(key = lambda x: int(x[0:-4]))
    for i in range(len(files)):
        filename=pathIn + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

if __name__ == '__main__':
    print(">> verts_animation.py tests are not implemented yet.")
    png2video('./rendered_jpgs/')