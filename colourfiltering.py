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



def colourmap_sandbox(image):
    filtered_img = filter_white_yellow_hls2(image)
    plt.imshow(filtered_img)
    plt.show()
    filtered_img = filter_colors_hsv(image)
    plt.imshow(filtered_img)
    plt.show()

    print('colour sandbox')
    global image_array, image_titles
    thresh = (180, 255)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
    # plt.imshow(binary,cmap='gray')
    # plt.show()
    # cv2.imshow('binary',binary)
    # cv2.waitKey(0)
    # image_array = []
    # image_titles = []
    # R = image[:, :, 0]
    # G = image[:, :, 1]
    # B = image[:, :, 2]
    split_and_thres(image, 'RGB', (150, 200), (150, 200), (150, 200))
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    split_and_thres(hls, 'HLS', (0, 20), (200, 255), (180, 255))
    # H = hls[:, :, 0]
    # L = hls[:, :, 1]
    # S = hls[:, :, 2]
    # image_array.append(H)
    # image_array.append(L)
    # image_array.append(S)
    # image_titles.append('HLS H')
    # image_titles.append('HLS L')
    # image_titles.append('HLS S')
    # plt.imshow(H,cmap='gray')
    # plt.show()
    # plt.imshow(L,cmap='gray')
    # plt.show()
    # plt.imshow(S,cmap='gray'    )
    # plt.show()
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    split_and_thres(hsv, 'HSV', (0, 10), (0, 170), (50, 170))
    # H = hsv[:, :, 0]
    # S = hsv[:, :, 1]
    V = hsv[:, :, 2]


    plt.imshow(V)
    plt.show()
    temp = threshold(V, thresh=(200, 255))
    # temp = cv2.cvtColor(V, cv2.bi)
    plt.imshow(temp)
    plt.show()
    # image_array.append(H)
    # image_array.append(S)
    # image_array.append(V)
    # image_titles.append('HLS H')
    # image_titles.append('HLS S')
    # image_titles.append('HLS V')
    # binary_V = threshold(V, thresh=(230, 255))
    # plt.imshow(H,cmap='gray')
    # plt.title('HSV H')
    # plt.show()
    # plt.imshow(S,cmap='gray')
    # plt.title('HSV S')
    # plt.show()
    # plt.imshow(V,cmap='gray'    )
    # plt.title('HSV V')
    # plt.show()
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    split_and_thres(lab, 'LAB', (150, 255), (0, 255), (20, 140))
    # L = lab[:, :, 0]
    # A = lab[:, :, 1]
    # B = lab[:, :, 2]
    # image_array.append(L)
    # image_array.append(A)
    # image_array.append(B)
    # image_titles.append('LAB L')
    # image_titles.append('LAB A')
    # image_titles.append('LAB B')
    # plt.imshow(L,cmap='gray')
    # plt.title('LAB L')
    # plt.show()
    # plt.imshow(A,cmap='gray')
    # plt.title('LAB A')
    # plt.show()
    # plt.imshow(B,cmap='gray'    )
    # plt.title('LAB B')
    # plt.show()
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    split_and_thres(yuv, 'YUV', (0, 20), (20, 255), (0, 5))
    # Y = yuv[:, :, 0]
    # U = yuv[:, :, 1]
    # V = yuv[:, :, 2]
    # image_array.append(Y)
    # image_array.append(U)
    # image_array.append(V)
    # image_titles.append('YUV Y')
    # image_titles.append('YUV U')
    # image_titles.append('YUV V')
    # plt.imshow(Y,cmap='gray')
    # plt.title('YUV Y')
    # plt.show()
    # plt.imshow(U,cmap='gray')
    # plt.title('YUV U')
    # plt.show()
    # plt.imshow(V,cmap='gray')
    # plt.title('YUV V')
    # plt.show()
    # plot_figure(image_array, image_titles, 4, 3,(64,64), 'gray')


# colourmap_sandbox()


image_array = []
image_titles = []
##########

def filter_colors_hsv(img):
    """
    Convert image to HSV color space and suppress any colors
    outside of the defined color ranges
    """

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #TODO reinstate
    # plt.imshow(img)
    # plt.show()
    upper_thres = np.uint8([10,50,100])
    lower_thres = np.uint8([100,255,255])
    yellows = cv2.inRange(img, upper_thres, lower_thres)

    upper_thres = np.uint8([[0,0,200]])
    lower_thres = np.uint8([255,255,255])
    whites = cv2.inRange(img, upper_thres, lower_thres)
    yellows_or_whites = yellows | whites
    img = cv2.bitwise_and(img, img, mask=yellows | whites)
    # ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    img[(img > 0)] = 1

    return img[:, :, 0]

def filter_white_yellow_hls2(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    #TODO REINSTATE
    # plt.imshow(img)
    # plt.show()
    upper_thres = np.uint8([10,50,100])
    lower_thres = np.uint8([100,255,255])
    yellows = cv2.inRange(img, upper_thres, lower_thres)

    upper_thres = np.uint8([[15,180,0]])
    lower_thres = np.uint8([255,255,255])
    whites = cv2.inRange(img, upper_thres, lower_thres)
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
    return img[:, :, 0]



def get_combined_binary(image):
    # hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # split_and_thres(hsv, 'HSV', (0, 10), (0, 25), (0, 255))
    # H = hsv[:, :, 0]
    # L = hsv[:, :, 1]
    # S = hsv[:, :, 2]
    # global image_array, image_titles
    # image_array.append(H)
    # image_array.append(L)
    # image_array.append(S)
    # image_titles.append('HLS H')
    # image_titles.append('HLS L')
    # image_titles.append('HLS S')
    # binary_S = threshold(S, thresh=(0, 255))
    binary_S = filter_white_yellow_hls2(image)

    #TODO REINSTATE
    # plt.imshow(binary_S)
    # print('binary_S: {} ' .format(binary_S))
    # plt.title("binary s")
    # plt.show()


    sobel_mag = mag_thresh(image, 3, (30, 100))
    # plt.imshow(sobel_mag, cmap ='gray')
    # plt.show()
    sobel_x = abs_sobel_thresh(image, 'x', 3, (30, 80))
    print('sobel: {} ' .format(sobel_x))
    # plt.imshow(sobel_x, cmap ='gray')
    # plt.show()
    sobel_y = abs_sobel_thresh(image, 'y', 3, (30, 100))
    # plt.imshow(sobel_y, cmap ='gray')
    # plt.show()
    image_array = []
    image_titles = []
    image_array.append(sobel_mag)
    image_array.append(sobel_x)
    image_array.append(sobel_y)
    image_titles.append('sobel_mag')
    image_titles.append('sobel_x')
    image_titles.append('sobel_y')
    plot_figure(image_array, image_titles, 1, 3, (64, 64), 'gray')
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
    plot_figure(image_array, image_titles, 1, 3, (64, 64), 'gnuplot')
    # cv2.imshow("thingy",combined_binary )
    # cv2.waitKey(0)
    return combined_binary

# get_combined_binary(image)

# colourmap_sandbox()


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

def filter_hls(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # H = hls[:, :, 0]
    # L = hls[:, :, 1]
    S = hls[:, :, 2]
    # global image_array, image_titles
    # image_array.append(H)
    # image_array.append(L)
    # image_array.append(S)
    # image_titles.append('HLS H')
    # image_titles.append('HLS L')
    # image_titles.append('HLS S')
    binary_S = threshold(S, thresh=(180, 255))
    return binary_S

    # split_and_thres(image, 'RGB', (150, 200), (150, 200), (150, 200))
    # hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # return split_and_thres(hls, 'HLS', (0, 20), (200, 255), (180, 255))



def main():
    image = mpimg.imread('test_images/straight_lines1.jpg')
    image = mpimg.imread('test_images/challenge4.jpg')
    image = mpimg.imread('test_images/test2.jpg')

    #TODO REINSTATE
    # plt.imshow(image)
    # plt.show()
    colourmap_sandbox(image)


if __name__ == "__main__":
    main()