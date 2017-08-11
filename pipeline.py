from colourfiltering import get_combined_binary, filter_colors_hsv, filter_white_yellow_hls2, analyse_histogram, filter_hls

from moviepy.editor import VideoFileClip

from perspective_transform import birdseye, warp_back

import cv2
import numpy as np
import matplotlib.image as mpimg
#from util import plot_figure
import matplotlib.pyplot as plt
# from conv_lane_finder import find_the_lanes
from basic_sliding_lane_finder_refactored import find_lanes_blind, find_lanes_near, fit_poly_lines


from calibration import load_calibration, undistort


# loap calibration
mtx, dist= load_calibration()



image = mpimg.imread('test_images/straight_lines1.jpg')
image = mpimg.imread('test_images/straight_lines2.jpg')
image = mpimg.imread('test_images/test1.jpg')
#image = mpimg.imread('test_images/test2.jpg')
#image = mpimg.imread('test_images/challenge.png')
#image = mpimg.imread('test_images/challenge2.png')
#image = mpimg.imread('test_images/challenge4.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



def process_image(image):
    image_wip = undistort(image, mtx, dist)
    image_wip, M, Minv = birdseye(image_wip)

    #TODO reinstate
    # plt.imshow(image_wip)
    # plt.title('pipeline: birdseye')
    image_wip1 = filter_colors_hsv(image_wip)
    analyse_histogram(image_wip1)
    image_wip2 = filter_white_yellow_hls2(image_wip)
    analyse_histogram(image_wip2)
    image_wip3 = filter_hls(image_wip)
    analyse_histogram(image_wip3)
    # image = get_combined_binary(image)
    # plt.imshow(image)
    # plt.title('pipeline: combined')
    # plt.show()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.show()
    # find_the_lanes(binary)
    find_lanes_blind(image_wip3)
    left_fitx, right_fitx, ploty = fit_poly_lines(image_wip1)
    return warp_back(image, left_fitx, right_fitx, ploty, Minv)

def process_video():
    white_output = 'output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)

# process_image(image)
process_video()




