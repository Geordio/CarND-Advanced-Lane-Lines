import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from util import plot_figure
from util import mag_thresh, abs_sobel_thresh


def split_and_thres(img, colourspace, thres1= (70,100), thres2= (70,100),thres3= (70,100)):
    image_array = []
    image_titles = []

    chn1 = img[:, :, 0]
    chn2 = img[:, :, 1]
    chn3 = img[:, :, 2]
    #get the text for each channel
    chn1_txt = colourspace[0]
    chn2_txt = colourspace[1]
    chn3_txt = colourspace[2]

    image_array.append(chn1)
    image_array.append(chn2)
    image_array.append(chn3)
    image_titles.append(chn1_txt)
    image_titles.append(chn2_txt)
    image_titles.append(chn3_txt)

    print('chn1')
    image_array.append(threshold(chn1,thres1))
    print('chn2')
    image_array.append(threshold(chn2,thres2))
    print('chn3')
    image_array.append(threshold(chn3,thres3))
    image_titles.append(chn1_txt + ' binary')
    image_titles.append(chn2_txt + ' binary')
    image_titles.append(chn3_txt + ' binary')

    plot_figure(image_array, image_titles, 2, 3,(64,64), 'gray')


def threshold(chn, thresh=(15, 100)):
    binary = np.zeros_like(chn)
    print('thres: {}, {}' .format(thresh[0], thresh[1]))
    binary[(chn > thresh[0]) & (chn <= thresh[1])] = 1

    print (chn)
    return binary


# trial different colour space manipulations
def colourmap_sandbox(image):
    filtered_img = filter_white_yellow_hls2(image)
    plt.imshow(filtered_img)
    plt.show()
    filtered_img = filter_white_yellow_hsv(image)
    plt.imshow(filtered_img)
    plt.show()

    print('colour sandbox')
    global image_array, image_titles
    thresh = (180, 255)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

    split_and_thres(image, 'RGB', (150, 200), (150, 200), (150, 200))
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    split_and_thres(hls, 'HLS', (0, 20), (200, 255), (180, 255))
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #50, 170
    split_and_thres(hsv, 'HSV', (0, 10), (0, 170), (150, 255))
    # H = hsv[:, :, 0]
    # S = hsv[:, :, 1]
    V = hsv[:, :, 2]


    plt.imshow(V)
    plt.show()
    temp = threshold(V, thresh=(200, 255))
    plt.imshow(temp)
    plt.show()

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    split_and_thres(lab, 'LAB', (150, 255), (0, 255), (20, 140))

    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    split_and_thres(yuv, 'YUV', (0, 20), (20, 255), (0, 5))





image_array = []
image_titles = []
##########

# create a binary representation by filtering of the S channel of HLS colourspace
def filter_hls(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # plt.imshow(image)
    # plt.show()
    S = hls[:, :, 2]
    binary = threshold(S, thresh=(180, 255))
    # plt.imshow(binary)
    # plt.show()
    return binary

def v_mean(img):
    temp = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    V = temp[:, :, 2]
    v_mean_val = V.mean()

    # plt.imshow(V)
    # plt.show()
    print('MEAN_V: {}' .format(v_mean_val))
    return v_mean_val

def s_mean(img):
    temp = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    S = temp[:, :, 1]
    s_mean_val = S.mean()
    print('MEAN_S: {}' .format(s_mean_val))
    #
    # plt.imshow(S)
    # plt.show()
    return s_mean_val

# create a binary representation by filtering of the V channel of HSV colourspace
def filter_white_yellow_hsv(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_mean(img)
    s_mean(img)

    if v_mean(img) < 150:
        print('MEAN TRUE')
        upper_thres_yell = np.uint8([10,50,100])
        lower_thres_yell = np.uint8([100,255,255])
        upper_thres_white = np.uint8([[0, 0, 200]])
        lower_thres_white = np.uint8([255, 255, 255])
    else:
        print('MEAN FALSE')
        upper_thres_yell = np.uint8([10,50,200])
        lower_thres_yell = np.uint8([100,255,255])
        upper_thres_white = np.uint8([[0, 0, 220]])
        lower_thres_white = np.uint8([255, 255, 255])
    yellows = cv2.inRange(img, upper_thres_yell, lower_thres_yell)
    whites = cv2.inRange(img, upper_thres_white, lower_thres_white)

    img = cv2.bitwise_and(img, img, mask=yellows | whites)
    # ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    img[(img > 0)] = 1
    return img[:, :, 0]


# create a binary representation by filtering of the HLS colourspace on the
# ranges of yellows and whites.
# yellows and white ranges are handled independnatly, with an aggregated image created by ORing the
# 2 seperate filters together
def filter_white_yellow_hls2(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    #TODO REINSTATE
    # plt.imshow(img)
    # plt.show()
    # upper_thres = np.uint8([10,50,100])
    # lower_thres = np.uint8([100,255,255])
    upper_thres = np.uint8([10,50,100])
    lower_thres = np.uint8([100,255,255])
    yellows = cv2.inRange(img, upper_thres, lower_thres)
    # plt.imshow(yellows)
    # plt.title('yellow')
    # plt.show()

    upper_thres = np.uint8([[15,180,80]])
    lower_thres = np.uint8([255,255,255])


    whites = cv2.inRange(img, upper_thres, lower_thres)
    # plt.imshow(whites)
    # plt.title('whites')
    # plt.show()

    yellows_or_whites = yellows | whites

    img = cv2.bitwise_and(img, img, mask=yellows | whites)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    image_array = []
    image_titles = []
    image_array.append(img[:, :, 0])
    image_array.append(img[:, :, 1])
    image_array.append(img[:, :, 2])
    image_titles.append('b H')
    image_titles.append('b L')
    image_titles.append('b S')
    # plot_figure(image_array, image_titles, 1,3,(64, 64), 'gnuplot')
    # img[(img > 0) & (img <= 255)] = 1
    return img#[:, :, 0]


# create a binary image based on colour filtering and sobel
def get_combined_binary(image):

    binary_S = filter_hls(image)

    sobel_mag = mag_thresh(image, 3, (30, 100))
    # plt.imshow(sobel_mag, cmap ='gray')
    # plt.show()
    sobel_x = abs_sobel_thresh(image, 'x', 3, (30, 100))
    print('sobel: {} ' .format(sobel_x))
    # plt.imshow(sobel_x, cmap ='gray')
    # plt.show()
    sobel_y = abs_sobel_thresh(image, 'y', 3, (30, 100))
    # plt.imshow(sobel_y, cmap ='gray')
    # plt.show()
    image_array = []
    image_titles = []
    image_array.append(image)
    image_array.append(sobel_mag)
    image_array.append(sobel_x)
    image_array.append(sobel_y)
    image_titles.append('original')
    image_titles.append('sobel_mag')
    image_titles.append('sobel_x')
    image_titles.append('sobel_y')
    # plot_figure(image_array, image_titles, 2, 2, (64, 64), 'gray')
    color_binary = np.dstack((np.zeros_like(sobel_x), sobel_x, binary_S))
    # plt.imshow(binary_S, cmap ='gray')
    # plt.title('binary')
    # plt.show()
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(binary_S)
    # combined_binary[(binary_S == 1) ] = 1
    combined_binary[(sobel_y == 1) | (sobel_x == 1)] = 1
    combined_binary[(binary_S > 0) | (sobel_x == 1)] = 1

    #TODO REINSTATE
    # plt.imshow(combined_binary, cmap ='gray')
    # plt.title('test')
    # plt.show()

    image_array = []
    image_titles = []
    image_array.append(sobel_x)
    image_array.append(binary_S)
    image_array.append(combined_binary)
    # image_array.append(cv2.cvtColor(combined_binary, cv2.COLOR_BGR2RGB))
    image_titles.append('sobel_x')
    image_titles.append('binary_S')
    image_titles.append('combined_binary')
    # plot_figure(image_array, image_titles, 1, 3, (64, 64), 'gnuplot')
    # cv2.imshow("thingy",combined_binary )
    # cv2.waitKey(0)
    return combined_binary

# get_combined_binary(image)

# colourmap_sandbox()

# analyse the histogram of image and return the poits from where to start searching for lanes (as indicated by the peaks)
def analyse_histogram(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    # print('leftx_base: {}, {}, {}'.format(leftx_base, np.max(histogram[:midpoint]), (np.mean(histogram[:midpoint]))))
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # print('rightx_base: {}, {}, {}'.format(rightx_base, np.max(histogram[midpoint:]), (np.mean(histogram[midpoint:]))))

# check is the peak is above a threshold, and the ratio of peak to mean is greater than another threshold
    peak_thesh = 5000
    peak2mean_thresh = 2
    left_mean = np.mean(histogram[:midpoint])
    left_peak = np.max(histogram[:midpoint])
    right_mean = np.mean(histogram[midpoint:])
    right_peak = np.max(histogram[midpoint:])
    print('leftx_base: {}, {}, {}'.format(leftx_base, left_peak, left_mean, left_peak/left_mean))
    # rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    print('rightx_base: {}, {}, {}'.format(rightx_base,right_peak, right_mean, right_peak/right_mean))


    #TODO reinstate
    # plt.plot(histogram)
    # plt.show()

    return leftx_base, rightx_base

#
# ## filter the inpt image based on the HLS colour space,
# def filter_hls(image):
#     hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
#     S = hls[:, :, 2]
#
#     binary_S = threshold(S, thresh=(180, 255))
#     return binary_S



def main():
    image = mpimg.imread('test_images/straight_lines1.jpg')
    image = mpimg.imread('test_images/challenge.jpg')
    # image = mpimg.imread('test_images/test2.jpg')
    # image = mpimg.imread('test_images/test1.jpg')
    #TODO REINSTATE
    # plt.imshow(image)
    # plt.show()
    # colourmap_sandbox(image)
    get_combined_binary(image)


if __name__ == "__main__":
    main()