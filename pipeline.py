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


## need to determine her how to find them, blind or abased on previous

    lanes_found, leftx_pixel_pos, lefty_pixel_pos, \
    rightx_pixel_pos, righty_pixel_pos = frame.get_raw_line_pixels(image)
    l_line.set_ploty(frame.ploty)
    r_line.set_ploty(frame.ploty)


    if lanes_found:
        left_fit = l_line.fit_poly_lines(leftx_pixel_pos, lefty_pixel_pos )
        right_fit = r_line.fit_poly_lines(rightx_pixel_pos, righty_pixel_pos)
        left_fitx = l_line.get_line_poly_pix(left_fit)
        right_fitx = r_line.get_line_poly_pix( right_fit)
        l_line.addline(left_fit, left_fitx)
        r_line.addline(right_fit, right_fitx)
    else:
        print('no lines found')




    # left_best = l_line.get_best_fit_x(frame.ploty)
    # right_best = r_line.get_best_fit_x(frame.ploty)

    # frame.get_car_position(frame, image, l_line, r_line)


    # left_rad, right_rad = frame.calculate_radius_real(left_fit, right_fit)
    # frame.visualise_poly()


    image_with_marking = frame.create_output_drive_view(frame.image_undistorted, \
                                                        l_line, r_line, frame.Minv)
        # plt.imshow(self.image_with_marking)
        # plt.show()

    # else:

    return image_with_marking



    # image_wip = undistort(image, mtx, dist)
    # image_wip, M, Minv = birdseye(image_wip)
    #
    # #TODO reinstate
    # # plt.imshow(image_wip)
    # # plt.title('pipeline: birdseye')
    # image_wip1 = filter_colors_hsv(image_wip)
    # analyse_histogram(image_wip1)
    # image_wip2 = filter_white_yellow_hls2(image_wip)
    # analyse_histogram(image_wip2)
    # image_wip3 = filter_hls(image_wip)
    # analyse_histogram(image_wip3)
    # # image = get_combined_binary(image)
    # # plt.imshow(image)
    # # plt.title('pipeline: combined')
    # # plt.show()
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # plt.show()
    # # find_the_lanes(binary)
    # find_lanes_blind(image_wip3)
    # left_fitx, right_fitx, ploty = fit_poly_lines(image_wip1)
    # return warp_back(image, left_fitx, right_fitx, ploty, Minv)

def process_video():
    output_video = 'output.mp4'
    # clip1 = VideoFileClip("project_video.mp4")
    # clip1 = VideoFileClip("challenge_video.mp4")
    clip1 = VideoFileClip("harder_challenge_video.mp4")

    output_clip = clip1.fl_image(process_image)
    output_clip.write_videofile(output_video, audio=False)

# process_image(image)
process_video()




