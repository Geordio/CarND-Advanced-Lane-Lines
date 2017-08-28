import numpy as np
# from colourfiltering import get_combined_binary, filter_hsv, filter_white_yellow_hls2, analyse_histogram, filter_hls

from moviepy.editor import VideoFileClip

# from perspective_transform import birdseye, warp_back

import cv2
import numpy as np
import matplotlib.image as mpimg
from util import plot_figure
from Line import Line
import matplotlib.pyplot as plt

from colourfiltering import get_combined_binary, filter_white_yellow_hls2, filter_white_yellow_hsv, filter_hls, analyse_histogram, v_mean



from calibration import load_calibration, undistort

class Frame():

    def __init__(self,  l_line, r_line, c_mtx=None, c_dist=None ):
        # was the line detected in the last iteration?
        self.image_raw = False
        self.image_undistorted = False
        self.image_windowed = None
        self.image_with_marking = None
        self.image_birdseye = None
        self.image_output = None
        self.image_hsv_bin = None
        self.image_hls_bin = None
        self.image_flt_hsv_bin = None
        self.image_hsv_histo = None
        self.image_yellow_white_hls_bin = None
        self.image_birdseye_debug = None

        # self.last_valid_frame = Frame()
        self.frame_count = 0

        self.mtx = c_mtx
        self.dist = c_dist
        self.M = None
        self.Minv = None
        self.leftx_base = 0
        self.rightx_base = 0

        self.nonzero = []
        self.nonzeroy = []
        self.nonzerox = []
        self.ym_per_pix = 3. / 160  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 560  # meteres per pixel in x dimension



# left and right x and y are the pixel positions of each line
#         self.leftx_pixel_pos = []
#         self.lefty_pixel_pos = []
#         self.rightx_pixel_pos = []
#         self.righty_pixel_pos = []

        self.left_lane_pix_found = False
        self.right_lane_pix_found = False
        self.both_lanes_pix_found = False
        self.left_lane_inds = []
        self.right_lane_inds = []
        self.margin = 100
        self.ploty = None


        # self.left_rad = 0
        # self.right_rad = 0
        self.left_lane_pts = []
        self.right_lane_pts = []
        # Polyfit represnetation of each Lane
        self.left_fitx = None
        self.right_fitx = None
        self.left_fit = None
        self.right_fit = None

        self.l_line = l_line
        self.r_line = r_line



    def set_input_image(self, image):
        self.image_raw = image

    def set_camera_params(self, c_mtx, c_dist ):
        self.mtx = c_mtx
        self.dist = c_dist




# prcoess an image and return the pixels that have been detected as potential lines
    def get_raw_line_pixels(self, image):
        self.image_undistorted = undistort(image, self.mtx, self.dist)
        self.ploty = np.linspace(0, self.image_undistorted.shape[0] - 1, self.image_undistorted.shape[0])

        self.image_birdseye, self.M, self.Minv = self.get_birdseye(self.image_undistorted)

        # TODO reinstate
        # plt.imshow(image_wip)
        # plt.title('pipeline: birdseye')

        # self.image_hls_bin = get_combined_binary(self.image_birdseye)

        self.image_hsv_bin = filter_white_yellow_hsv(self.image_birdseye)
        self.image_hls_bin = filter_hls(self.image_birdseye)


        analyse_histogram(self.image_hsv_bin)
        self.image_yellow_white_hls_bin = image_wip2 = filter_white_yellow_hls2(self.image_birdseye)
        # filter_white_yellow_hls2
        # analyse_histogram(image_wip2)
        self.image_flt_hls_bin = filter_hls(self.image_birdseye)




        # image = get_combined_binary(image)
        # plt.imshow(image)
        # plt.title('pipeline: combined')
        # plt.show()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.show()
        # find_the_lanes(binary)

        # first try to find the lines blindly.

        if (self.l_line.plausible == True) & (self.r_line.plausible == True) :
            self.lanes_found, self.leftx_pixel_pos, self.lefty_pixel_pos, self.rightx_pixel_pos, self.righty_pixel_pos = self.find_lanes_near(
                self.image_hsv_bin)
        else :
            self.lanes_found, self.leftx_pixel_pos, self.lefty_pixel_pos, self.rightx_pixel_pos, self.righty_pixel_pos = self.find_lanes_blind(self.image_hsv_bin)

        print('self.left_fitx: {}, self.right_fitx: {}' .format(self.left_fitx, self.right_fitx))


        # self.calculate_radius_real(self.left_fit, self.right_fit)
        return self.lanes_found, self.leftx_pixel_pos, self.lefty_pixel_pos, self.rightx_pixel_pos, self.righty_pixel_pos



    #
    # def split_and_thres(self, img, colourspace, thres1=(70, 100), thres2=(70, 100), thres3=(70, 100)):
    #     image_array = []
    #     image_titles = []
    #
    #     chn1 = img[:, :, 0]
    #     chn2 = img[:, :, 1]
    #     chn3 = img[:, :, 2]
    #     # get the text for each channel
    #     chn1_txt = colourspace[0]
    #     chn2_txt = colourspace[1]
    #     chn3_txt = colourspace[2]
    #
    #     image_array.append(chn1)
    #     image_array.append(chn2)
    #     image_array.append(chn3)
    #     image_titles.append(chn1_txt)
    #     image_titles.append(chn2_txt)
    #     image_titles.append(chn3_txt)
    #
    #     print('chn1')
    #     image_array.append(self.threshold(chn1, thres1))
    #     print('chn2')
    #     image_array.append(self.threshold(chn2, thres2))
    #     print('chn3')
    #     image_array.append(self.threshold(chn3, thres3))
    #     image_titles.append(chn1_txt + ' binary')
    #     image_titles.append(chn2_txt + ' binary')
    #     image_titles.append(chn3_txt + ' binary')
    #
    #     plot_figure(image_array, image_titles, 2, 3, (64, 64), 'gray')
    #
    # def threshold(self, chn, thresh=(15, 100)):
    #     binary = np.zeros_like(chn)
    #     print('thres: {}, {}'.format(thresh[0], thresh[1]))
    #     binary[(chn > thresh[0]) & (chn <= thresh[1])] = 1
    #
    #     print(chn)
    #     return binary

# ## Filter the colours based on a range on the HSV colour space
#     def filter_colors_hsv(self, img):
#
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#
#         # TODO reinstate
#         # plt.imshow(img)
#         # plt.show()
#         upper_thres = np.uint8([10, 50, 100])
#         lower_thres = np.uint8([100, 255, 255])
#         yellows = cv2.inRange(img, upper_thres, lower_thres)
#
#         upper_thres = np.uint8([[0, 0, 200]])
#         lower_thres = np.uint8([255, 255, 255])
#         whites = cv2.inRange(img, upper_thres, lower_thres)
#         yellows_or_whites = yellows | whites
#         img = cv2.bitwise_and(img, img, mask=yellows | whites)
#         # ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
#
#         img[(img > 0)] = 1
#
#         return img[:, :, 0]




    # generate the points used as source and destination of the warping
    # the points are based on the relative position rather than absolute pixel coordinates
    def calc_warp_points(self, img):

        # h, w = img.shape[:2]
        height = img.shape[0]
        width = img.shape[1]
        print('height: {}, width:{}'.format(height, width))

        tl_x_scaling = 0.41
        tl_y_scaling = 0.68
        bl_x_scaling = 0.170
        bl_y_scaling = 0.94
        br_x_scaling = 1 - bl_x_scaling
        br_y_scaling = bl_y_scaling
        tr_x_scaling = 1 - tl_x_scaling
        tr_y_scaling = tl_y_scaling

        src = np.float32([[width * tr_x_scaling, height * tr_y_scaling], \
                          [width * br_x_scaling, height * bl_y_scaling], \
                          [width * bl_x_scaling, height * bl_y_scaling], \
                          [width * tl_x_scaling, height * tl_y_scaling]])
        # print(src)

        # TODO REINSTATE
        # plt.imshow(img)

        # for pt in src:
        #     plt.plot(pt[0], pt[1], '.', color='m')

        l_dst_scaling = 0.25
        r_dst_scaling = 1 - 0.25

        dst = np.float32([[width * r_dst_scaling, 0], \
                          [width * r_dst_scaling, height - 1], \
                          [width * l_dst_scaling, height - 1], \
                          [width * l_dst_scaling, 0]])
        # for pt in dst:
        #     plt.plot(pt[0], pt[1], '.', color='r')

        # TODO REINSTATE
        # plt.show()
        # print(dst)
        return src, dst


# does warp to create the birdseye view
    def get_birdseye(self, img):

        src, dst = self.calc_warp_points(img)
        height = img.shape[0]
        width = img.shape[1]

        print('height: {}, width:{}' .format(height,width))
        # get the transformation matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # get the inverse matrix
        Minv = cv2.getPerspectiveTransform(dst, src)

        warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)

        # images = []
        # titles = []
        # images.append(img)
        # images.append(warped)
        # titles.append('normal view')
        # titles.append('birdseye view')
        # plot_figure(images, titles,1,2)
        return warped, M, Minv

# reverses the birdeye view warp.
# Plots the polylines and fills that area between
    def create_output_drive_view(self, image, l_line, r_line, Minv):
        # Create an image to draw the lines on
        left_fitx = l_line.get_best_fit_x()
        right_fitx = r_line.get_best_fit_x()

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

        warp_zero = np.zeros_like(image[:,:,0]).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = []
        pts_right = []
        if (left_fitx is not None):
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        if (right_fitx is not None):
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

        print('pts_left, pts_right: {}, {}' .format(pts_left, pts_right))

        # handle the cases where left or right are empty
        if (len(pts_left)> 0) & (len(pts_right) > 0):
            pts = np.hstack((pts_left, pts_right))
        elif (len(pts_left) == 0):
            pts = pts_right
        else:
            pts = pts_left


        # Draw the lane onto the warped blank image
        if len(pts)>0 :
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
            cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0, 0, 255), thickness=20)
            cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=20)

# store this as a image for bebug use
        self.image_birdseye_lanes = color_warp

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.4, 0)
        self.image_output = result
        #TODO REINSTATE
        # plt.imshow(result)
        # plt.show()
        # l_rad = l_line.calculate_radius_real(l_line, self.ym_per_pix, self.xm_per_pix)
        # r_rad = r_line.calculate_radius_real(r_line, self.ym_per_pix, self.xm_per_pix)
        rad, rad_txt = self.get_avg_rad(l_line, r_line)
        distance_from_centre_px, distance_from_centre_m, distance_from_centre_text = self.get_car_position(result, l_line, r_line)
        self.annotate_image(result, distance_from_centre_text, rad_txt)

        # return self.create_debug_output()
        return result


# creates an output that includes multiple images taken throughout the pipeline.
# This creates a picture in picture output that allows the performance of things like colour thresholding to be evaluated.
    def create_debug_output(self):
        import sys
        from PIL import Image
        import PIL

        # images = map(Image.open, ['Test1.jpg', 'Test2.jpg', 'Test3.jpg'])
        # widths, heights = zip(*(i.size for i in images))

        height = self.image_undistorted.shape[0]
        width = self.image_undistorted.shape[1]

        total_width = int(width * 5/4)
        max_height = height

        debug_main = PIL.Image.fromarray(self.image_output)
        new_im = Image.new('RGB', (total_width, max_height))

        new_im.paste(debug_main ,(0,0))
        image_birdseye_resized = []
        image_hsv_bin_resized = []
        # cv2.resize(self.image_birdseye , image_birdseye_resized, image_birdseye_resized.size(), 0.25, 0.25, cv2.interpolation);

        image_undistorted_resized = PIL.Image.fromarray(cv2.resize(self.image_undistorted, (0, 0), fx=0.25, fy=0.25))
        image_birdseye_resized = PIL.Image.fromarray(cv2.resize(self.image_birdseye, (0, 0), fx=0.25, fy=0.25))

        image_hsv_bin_temp = np.dstack((self.image_hsv_bin, self.image_hsv_bin, self.image_hsv_bin))* 255
        image_hsv_bin_resized= cv2.resize(image_hsv_bin_temp, (0, 0), fx=0.25, fy=0.25)
        # image_hsv_bin_resized = np.dstack((image_hsv_bin_resized, image_hsv_bin_resized, image_hsv_bin_resized))* 255

        image_hls_bin_resized= cv2.resize(self.image_flt_hls_bin, (0, 0), fx=0.25, fy=0.25)
        image_hls_bin_resized = np.dstack((image_hls_bin_resized, image_hls_bin_resized, image_hls_bin_resized))* 255

        image_yell_wht_bin_resized= cv2.resize(self.image_yellow_white_hls_bin, (0, 0), fx=0.25, fy=0.25)
        image_yell_wht_resized = np.dstack((image_yell_wht_bin_resized, image_yell_wht_bin_resized, image_yell_wht_bin_resized))* 255
        # image_hsv_bin_resized = PIL.Image.fromarray(image_hsv_bin_resized)

        # plt.imshow(image_hsv_bin_resized)
        # plt.show()
        # image_hsv_bin_resized = np.dstack((image_hsv_bin_resized, image_hsv_bin_resized, image_hsv_bin_resized))
        # cv2.resize(self.image_birdseye , image_birdseye_resized, image_birdseye_resized.size(), 0.25, 0.25, cv2.interpolation);

        new_im.paste(image_undistorted_resized ,(width,0))
        new_im.paste(image_birdseye_resized ,(width,int(height*0.25)))
        new_im.paste(PIL.Image.fromarray(self.annotate_image(image_hsv_bin_resized, 'hsv_y&w',  str(v_mean(self.image_birdseye)))) ,(width,int(height*0.25*2)))
        new_im.paste(PIL.Image.fromarray(self.annotate_image(image_hls_bin_resized, 'hls')) ,(width,int(height*0.25*3)))
        # new_im.paste(PIL.Image.fromarray(self.annotate_image(image_yell_wht_resized, 'yell_wht')), (width, int(height * 0.25 * 3)))

        # new_im.save('test.jpg')
        return  np.asarray( new_im, dtype="int32" )



# find the pixels that are associated with a lane line
# finds the starting point by finding the peaks a historgram of the number of non zero pixels
# uses the sliding window method from the course notes
# each window is recentred on the mean poistion of the previous window.
    def find_lanes_blind(self, binary_warped):


        #TODO REINSTATE
        # plt.imshow(binary_warped)
        # plt.title('find lanes blind')
        # plt.show()
        self.leftx_base, self.rightx_base = self.get_starting_points(binary_warped)

        # Choose the number of sliding windows
        nwindows = 10
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        self.nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])
        # Current positions to be updated for each window
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base

        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 20

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xleft_low) & (
                self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xright_low) & (
                self.nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)


            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)


        # Extract left and right line pixel positions
        leftx_pixel_pos = self.nonzerox[left_lane_inds]
        lefty_pixel_pos = self.nonzeroy[left_lane_inds]
        rightx_pixel_pos = self.nonzerox[right_lane_inds]
        righty_pixel_pos = self.nonzeroy[right_lane_inds]

        lanes_found = self.check_lane_pixels_found(leftx_pixel_pos, lefty_pixel_pos, rightx_pixel_pos, righty_pixel_pos)

        return lanes_found, leftx_pixel_pos, lefty_pixel_pos, rightx_pixel_pos, righty_pixel_pos



# check is any potential lane pixels were found
    def check_lane_pixels_found(self, leftx_pixel_pos, lefty_pixel_pos, rightx_pixel_pos, righty_pixel_pos):
        # Fit a second order polynomial to each
        # print some debug
        print('len left x: {}, {}'.format(len(lefty_pixel_pos), len(leftx_pixel_pos)))
        print('len right x: {}, {}'.format(len(righty_pixel_pos), len(rightx_pixel_pos)))
        self.left_lane_pix_found = False
        self.right_lane_pix_found = False
        if len(rightx_pixel_pos) != 0:
            self.right_lane_pix_found = True
        if len(leftx_pixel_pos) != 0:
            self.left_lane_pix_found = True

        if (self.right_lane_pix_found & self.left_lane_pix_found ):
            self.both_lanes_pix_found = True
        else:
            self.both_lanes_pix_found = False
        return self.both_lanes_pix_found


# not used
    def visualise_poly(self):
            ##########################
            # Visualise
            # Generate x and y values for plotting
            ploty = np.linspace(0, self.image_birdseye.shape[0] - 1, self.image_birdseye.shape[0])
            # left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
            # right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

            # color the non zero pixels found in the windows to red and blue
            # out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
            # out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]


            # if len(self.l_line.currentx) > 0:
            #     # cv2.fillPoly(color_warp, np.int_([self.l_line.currentx]), (0, 255, 0))
            #     cv2.polylines(out_img, np.int32([self.l_line.currentx]), isClosed=False, color=(0, 0, 255), thickness=20)
            # if len(self.r_line.currentx) > 0:
            #     cv2.polylines(out_img, np.int32([self.r_line.currentx]), isClosed=False, color=(0, 0, 255), thickness=20)
            lanes_found = True
            plt.imshow(out_img)
            plt.title('out_img')
            plt.show()
        # TODO REINSTATE
            plt.imshow(out_img)
            plt.plot(self.l_line.currentx, ploty, color='red')
            plt.plot(self.r_line.currentx, ploty, color='red')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.title('fitx')
            plt.show()

            # self.lanes_found = lanes_found

            return

# find the starting points to be used by the find lanes blind function
# analyses the histogram of the birdseye view , and uses the peaks as a starting point.
    def get_starting_points(self, binary_warped):
        global out_img
        print('binary_warped.shape[0]: {}'.format(binary_warped.shape[0]))
        # take histrogram (of bottom of image)
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        #TODO reinistate        #TODO reinistate
        # plt.plot(histogram)
        # plt.show()


        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        print('leftx_base: {}, {}, {}'.format(leftx_base, np.max(histogram[:midpoint]), np.mean(histogram[:midpoint])))
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        print('rightx_base: {}, {}, {}'.format(rightx_base, np.max(histogram[midpoint:]), np.mean(histogram[midpoint:])))

        return leftx_base, rightx_base



# finds lanes using the previously found poly line as a starting point
    def find_lanes_near(self, binary_warped):

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])


        # create a margin round the previous polyline constant value
        left_lane_inds = ((nonzerox > (self.l_line.current_fit[0] * (nonzeroy ** 2) + self.l_line.current_fit[1] * nonzeroy + self.l_line.current_fit[2] - self.margin)) & (
        nonzerox < (self.l_line.current_fit[0] * (nonzeroy ** 2) + self.l_line.current_fit[1] * nonzeroy + self.l_line.current_fit[2] + self.margin)))
        right_lane_inds = (
        (nonzerox > (self.r_line.current_fit[0] * (nonzeroy ** 2) + self.r_line.current_fit[1] * nonzeroy + self.r_line.current_fit[2] - self.margin)) & (
        nonzerox < (self.r_line.current_fit[0] * (nonzeroy ** 2) + self.r_line.current_fit[1] * nonzeroy + self.r_line.current_fit[2] + self.margin)))
        # Again, extract left and right line pixel positions
        leftx_pixel_pos = nonzerox[left_lane_inds]
        lefty_pixel_pos = nonzeroy[left_lane_inds]
        rightx_pixel_pos = nonzerox[right_lane_inds]
        righty_pixel_pos = nonzeroy[right_lane_inds]

        lanes_found = self.check_lane_pixels_found(leftx_pixel_pos, lefty_pixel_pos, rightx_pixel_pos, righty_pixel_pos)
        return lanes_found, leftx_pixel_pos, lefty_pixel_pos, rightx_pixel_pos, righty_pixel_pos



    # visualise the ploy lines on the birdeys
    def visualise_poly_lines(self, binary_warped):
        global out_img
        ####################################
        # fit polynomial lines
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        print('self.left_fitx: {}, self.right_fitx: {}' .format(self.left_fitx, self.right_fitx))
        left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx - self.margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx + self.margin, self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        print('self.left_fitx: {}, self.right_fitx: {}' .format(self.left_fitx, self.right_fitx))

        right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx - self.margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx + self.margin, self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        plt.imshow(result)
        plt.title('result')
        plt.show()

        plt.imshow(window_img)
        plt.title('window_img')
        plt.show()
        plt.imshow(out_img)
        plt.title('out_img')
        plt.show()

        #TODO REINSTATE FOR VISUALISATION
        # plt.imshow(result)
        # plt.title('polylines')
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # plt.show()

        return left_line_pts, right_line_pts



# write test onto the top left of the image
    def annotate_image(self, image, line1, line2="", line3 =""):

        text_x = 20
        text_y = 25
        text_line_sp = 25


        # image_annotated = np.copy(image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image , line1, (text_x,text_y), font, 1, (0,255,255), 2)
        cv2.putText(image , line2, (text_x,text_y + text_line_sp), font, 1, (0,255,255), 2)
        cv2.putText(image , line3, (text_x,text_y + 2* text_line_sp), font, 1, (0,255,255), 2)

        return image

    # calculate the distance from the centre of the car to the centre of the lane
    # centre of the image is assumed to be the centre of the car
    # converted to metres using the pixels per metre defined in class
    def get_car_position(self, image, l_line, r_line):
        car_position = image.shape[1] / 2

        lx = l_line.get_near_line_point()
        rx = r_line.get_near_line_point()
        lane_centre = (lx + rx) / 2

        distance_from_centre_px = car_position - lane_centre

        print('distance_from_centre: {}'.format(distance_from_centre_px))

        distance_from_centre_m = distance_from_centre_px * self.xm_per_pix

        distance_from_centre_text = 'Distance from centre: {:03.2f}' .format(distance_from_centre_m)+'m'

        return distance_from_centre_px, distance_from_centre_m, distance_from_centre_text



# gets the average radius from the 2 lane line objects radius.
    def get_avg_rad(self, l_line, r_line):
        l_rad = l_line.radius_of_curvature
        r_rad = r_line.radius_of_curvature

        avg_rad = (l_rad + r_rad) /2
        avg_rad_text = 'Radius of curve: {:03.2f}' .format(avg_rad)+'m'
        return avg_rad, avg_rad_text
