import cv2
import numpy as np
import matplotlib.image as mpimg
#from util import plot_figure
import matplotlib.pyplot as plt

# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                     [img_size[0]-offset, img_size[1]-offset],
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M


def calc_points(img):

    # h, w = img.shape[:2]
    height = img.shape[0]
    width = img.shape[1]
    print('height: {}, width:{}' .format(height,width))
    tl_x_scaling =  0.41
    tl_y_scaling = 0.68
    bl_x_scaling = 0.170
    bl_y_scaling = 0.94
    br_x_scaling =  1 - bl_x_scaling
    br_y_scaling = bl_y_scaling
    tr_x_scaling = 1 - tl_x_scaling
    tr_y_scaling = tl_y_scaling


    src = np.float32([[width *tr_x_scaling, height * tr_y_scaling ], \
                    [width * br_x_scaling, height * bl_y_scaling],\
                    [width *bl_x_scaling, height * bl_y_scaling], \
                    [width * tl_x_scaling, height * tl_y_scaling]])
    print(src)

    #TODO REINSTATE
    # plt.imshow(img)
    for pt in src:
        plt.plot(pt[0], pt[1],'.')

    l_dst_scaling = 0.25
    r_dst_scaling = 1- 0.25


    dst = np.float32([[width *r_dst_scaling, 0 ], \
                    [width * r_dst_scaling, height-1],\
                    [width *l_dst_scaling, height-1], \
                    [width * l_dst_scaling, 0]])
    for pt in dst:
        plt.plot(pt[0], pt[1],'.')

    #TODO REINSTATE
    # plt.show()


    print(dst)
    return src, dst

def birdseye(img):

    src, dst = calc_points(img)
    height = img.shape[0]
    width = img.shape[1]

    print('height: {}, width:{}' .format(height,width))
    # get the transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # get the inverse
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)
    return warped, M, Minv


def warp_back( image, left_fitx, right_fitx, ploty, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # # Draw the lane onto the warped blank image
    # # cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    #
    # cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0, 0, 255), thickness=40)


    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0, 0, 255), thickness=20)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=20)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    #TODO REINSTATE
    # plt.imshow(result)
    # plt.show()
    return result



def perpective_sandbox():
    global image, warped, Minv
    image = mpimg.imread('test_images/straight_lines2.jpg')
    image = mpimg.imread('test_images/test1.jpg')
    image = mpimg.imread('test_images/test2.jpg')
    src, dst = calc_points(image)
    warped, M, Minv = birdseye(image)

    #TODO REINSTATE
    # plt.imshow(warped)
    # plt.show()


# perpective_sandbox()

def main():
    perpective_sandbox()


if __name__ == "__main__":
    main()