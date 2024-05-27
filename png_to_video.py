# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:50:18 2020

@author: Haoran6
"""

import cv2
import os
import numpy as np
from os.path import isfile, join


def combine_two_videos(first_video_path, second_video_path, 
                       new_video_path="combined.mp4", im_resolution=None):
    
    video1 = cv2.VideoCapture(first_video_path)
    video2 = cv2.VideoCapture(second_video_path)
    
    frame_rate = int(video1.get(cv2.CAP_PROP_FPS))
    frame_width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height= int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if im_resolution:
        frame_width, frame_height = im_resolution   
    else:
        im_resolution = (frame_width, frame_height)
    
    resolution = (frame_width*2, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(new_video_path, fourcc, frame_rate, resolution)
    
    while True:
       
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        
        if not ret1 or not ret2:
            break
        
        frame1 = center_crop(frame1, im_resolution)
        frame2 = center_crop(frame2, im_resolution) 
        canvas = np.zeros((frame_height, frame_width * 2, 3), dtype=np.uint8)
        canvas[:, :frame_width] = frame1
        canvas[:, frame_width:] = frame2
        out.write(canvas)

    video1.release() 
    video2.release()
    out.release()
    
"""
DOESNT WORK IDK WHY
def concatenate_videos(new_video_path, *videos):
   
    video_list = [cv2.VideoCapture(v) for v in videos]
        
    frame_width = 1024 #int(video_list[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height= 1024 #int(video_list[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(video_list[0].get(cv2.CAP_PROP_FPS))
    im_resolution = (frame_width, frame_height)
    resolution = (frame_width*2, frame_height)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(new_video_path, fourcc, frame_rate, resolution)

  
    while True:
        is_end = False
        canvas = np.zeros((frame_height, frame_width * len(videos), 3), dtype=np.uint8)
        
        for i, v in enumerate(video_list):
            
            r, frame = v.read()
            if not r:
                is_end = True
                break #reached end of a video
            
            frame = center_crop(frame, im_resolution)
            canvas[:, frame_width*i : frame_width*(i+1)] = frame
            
        if is_end:
            break
        else:
            out.write(canvas)
        
    [v.release() for v in video_list]
    out.release()
    
"""
        
def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def extract_image_files(files):
    image_extensions = ["png", "jpg", "jpeg"]
    image_files = []
    for file in files:
        if file.split('.')[-1] in image_extensions:
            image_files.append(file)
            
    return image_files

def png2video(input_path, output_path=None, fps=24):
    if output_path is None:
        output_path = 'result.avi'
        
    frame_array = []
    
    all_files = [f for f in os.listdir(input_path) if isfile(join(input_path, f))]
    image_files = extract_image_files(all_files)
        
    image_files.sort(key = lambda x: int(x[0:-4]))
    for i in range(len(image_files)):
        filename = input_path + image_files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        frame_array.append(img)
        
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

if __name__ == '__main__':
    print(">> Testing: ", os.path.basename(__file__))
    #png2video('./results/rendered_jpgs/', './results/result.avi')
    
    combine_two_videos(first_video_path = './results/smpl_rigid.avi', 
                       second_video_path = './results/smpl_jiggle.avi',
                       new_video_path = './results/combined.mp4',
                       im_resolution = (1024, 1024))

    #concatenate_videos('./results/combined.mp4', './results/smpl_rigid.avi', './results/smpl_jiggle.avi', './results/smpl_rigid.avi')
    
    
    