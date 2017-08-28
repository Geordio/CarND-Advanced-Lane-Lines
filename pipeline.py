from colourfiltering import get_combined_binary, filter_white_yellow_hsv, filter_white_yellow_hls2, analyse_histogram, filter_hls

from moviepy.editor import VideoFileClip


from Frame import Frame
from Line import Line


import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from Line import LaneSide

from calibration import load_calibration, undistort


# load calibration
mtx, dist= load_calibration()

l_line = Line(LaneSide.Left )
r_line = Line(LaneSide.Right)

# image = mpimg.imread('test_images/straight_lines1.jpg')
#image = mpimg.imread('test_images/straight_lines2.jpg')
# image = mpimg.imread('test_images/test1.jpg')
#image = mpimg.imread('test_images/test2.jpg')
image = mpimg.imread('test_images/challenge.jpg')
# image = mpimg.imread('test_images/challenge2.png')
# image = mpimg.imread('test_images/challenge4.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

frames = []


# prosses a single image
# is also called when processing a video, each frame is handled as a seperate image
def process_image(image):

    lanes_found = True
    leftx_pixel_pos = []
    lefty_pixel_pos = []
    rightx_pixel_pos = []
    righty_pixel_pos = []

    frame = Frame( l_line, r_line, mtx, dist )

# call method to return lane line pixels
    lanes_found, leftx_pixel_pos, lefty_pixel_pos, \
    rightx_pixel_pos, righty_pixel_pos = frame.get_raw_line_pixels(image)
    l_line.set_ploty(frame.ploty)
    r_line.set_ploty(frame.ploty)


# if a lane was found....
    if lanes_found:
        left_fit = l_line.fit_poly_lines(leftx_pixel_pos, lefty_pixel_pos )
        right_fit = r_line.fit_poly_lines(rightx_pixel_pos, righty_pixel_pos)
        left_fitx = l_line.get_line_poly_pix(left_fit)
        right_fitx = r_line.get_line_poly_pix( right_fit)

        l_line.addline(left_fit, left_fitx)
        r_line.addline(right_fit, right_fitx)
        # frame.visualise_poly()
    else:
        print('no lines found')

    image_with_marking = frame.create_output_drive_view(frame.image_undistorted, \
                                                        l_line, r_line, frame.Minv)


    return image_with_marking




def process_video():
    output_video = 'output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    # clip1 = VideoFileClip("challenge_video.mp4")
    # clip1 = VideoFileClip("harder_challenge_video.mp4")

    output_clip = clip1.fl_image(process_image)
    output_clip.write_videofile(output_video, audio=False)

# process_image(image)
process_video()




