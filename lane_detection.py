# ======================================================================================================================================================================= #
#-------------> Project 02 <---------------#
# ======================================================================================================================================================================= #
# Course    :-> ENPM673 - Perception for Autonomous Robots
# Date      :-> 13 March 2019
# Authors   :-> Niket Shah(UID: 116345156), Siddhesh(UID: 116147286), Sudharsan(UID: 116298636)
# ======================================================================================================================================================================= #

# ======================================================================================================================================================================= #
# Import Section for Importing library
# ======================================================================================================================================================================= #

import time
import sys
import copy
import numpy as np
import cv2 as cv

# ======================================================================================================================================================================= #
# Function Inverse warp
# ======================================================================================================================================================================= #
def inverse_warp(undist, warped_img, left_fit, right_fit, M):
    # get image height. So we get y values from 0 to img height-1
    img_height = warped_img.shape[0]
    # get list of all y coordinates
    y_coord = np.linspace(0, img_height-1, img_height)
    
    # Get x coordinates for left and right lanes using polyfit values
    left_x = left_fit[0]*y_coord**2 + left_fit[1]*y_coord + left_fit[2]
    right_x = right_fit[0]*y_coord**2 + right_fit[1]*y_coord + right_fit[2]

    # create a numpy array of warped_img size so we can draw on it
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    # make it three channel
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # use y and y coordinates of each line to get point format
    pts_left = np.array([np.transpose(np.vstack([left_x, y_coord]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, y_coord])))])
    points = np.hstack((pts_left, pts_right))

    # Draw the lane on warped image
    cv.fillPoly(color_warp, np.int_([points]), (0,255, 0))
    # Calculate M inverse so we can warp this region on world frame
    Minv = np.linalg.inv(M)
    new_warp = cv.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine with original frame
    unwarped = cv.addWeighted(undist, 1, new_warp, 0.4, 0)
    return unwarped
# ======================================================================================================================================================================= #
# Function Definition Section
# ======================================================================================================================================================================= #
def adjust_lane(x, y, img):
    new_x = 0
    i = 0
    x_dot = 0
    centroid = +10
    while i < 50:
        rect_right = img[y-10:y+10, x+i-10:x+i+10]
        hist_right = np.sum(rect_right[:, :], axis=0)
        try:
            if hist_right.mean() > 50:
                new_x = x + i
                x_dot = np.where(hist_right == np.amax(hist_right[:]))
                centroid = x_dot[0][0]
                return new_x, centroid
            rect_left = img[y-10:y+10, x-i-10:x-i+10]
            hist_left = np.sum(rect_left[:, :], axis=0)
            if hist_left.mean() > 50:
                x_dot = np.where(hist_left == np.amax(hist_left[:]))
                new_x = x - i
                centroid = x_dot[0][0]
                return new_x, centroid
        except:
            new_x = None

        i += 1
    return new_x, centroid


def fit_curve(left_lanex, left_laney, right_lanex, right_laney, img):

    try:
        left_coeffs = np.polyfit(left_laney, left_lanex, 2)
        right_coeffs = np.polyfit(right_laney, right_lanex, 2)

        plot_xy = np.linspace(0, img.shape[0] - 1, img.shape[0])
        fit_left = left_coeffs[0]*plot_xy**2 + left_coeffs[1]*plot_xy + left_coeffs[2]
        fit_right = right_coeffs[0]*plot_xy**2 + right_coeffs[1]*plot_xy + right_coeffs[2]


        left_line_window1 = np.array([np.transpose(np.vstack([fit_left-5, plot_xy]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_left+5, plot_xy])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([fit_right-5, plot_xy]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_right+5, plot_xy])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv.fillPoly(img, np.int_([left_line_pts]), (255, 255, 255))
        cv.fillPoly(img, np.int_([right_line_pts]), (255, 255, 255))
    except:
        pass
    return left_coeffs, right_coeffs

# ======================================================================================================================================================================= #
# Lane Detection codes
# ======================================================================================================================================================================= #
if __name__ == '__main__':

    # Importing the Video
    # =================================================================================================================================================================== #
    cap = cv.VideoCapture('Data/project_video.mp4')
    # cap = cv.VideoCapture('Data/challenge_video.mp4')
    count = 0
    # ============================================================================================================================================================= #
    # Camera Parameters
    # ============================================================================================================================================================= #
    K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
                  [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([-2.42565104e-01, - 4.77893070e-02, -
                     1.31388084e-03, - 8.79107779e-05, 2.20573263e-02])

    key_frame = cv.imread('Frames/frame331.jpg', 1)
    shawdow1 = cv.imread('Frames/frame538.jpg', 1)
    shawdow2 = cv.imread('Frames/frame549.jpg', 1)
    shawdow3 = cv.imread('Frames/frame573.jpg', 1)
    h,  w = key_frame.shape[:2]
    newmatrix, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 0)

    reference = np.array([[600, 450], [715, 450], [1280, 670], [185, 670]])
    target = np.array([[300, 0], [600, 0], [600, 600], [300, 600]])

    H_mat, status = cv.findHomography(reference, target)

    while True:
        # Capture frame-by-frame
        ret, key_frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # ============================================================================================================================================================= #
        # Removing Distortion in the Image
        # ============================================================================================================================================================= #
        key_frame = cv.undistort(key_frame, K, dist, None, newmatrix)

        # ============================================================================================================================================================= #
        # Denoising the Image
        # ============================================================================================================================================================= #
        # denoised_kf = cv.fastNlMeansDenoisingColored(key_frame, None, 10, 10, 7, 15)
        denoised_kf = cv.bilateralFilter(key_frame, 16, 100, 100)
        
        # ============================================================================================================================================================= #
        # Unwarpping the Image
        # ============================================================================================================================================================= #
        bird_view = cv.warpPerspective(denoised_kf, H_mat, (900, 600))
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # bird_view = cv.filter2D(bird_view, -1, kernel)
        # =================================================================================================================================================================== #
        # Improving contrast
        # =================================================================================================================================================================== #
        hls = cv.cvtColor(bird_view, cv.COLOR_BGR2HLS)
        h, l1, s = cv.split(hls)
        l1_flag = l1 > 180
        l1[l1_flag] = 255
        n_hls = cv.merge((h, l1, s))
        bird_view = cv.cvtColor(n_hls, cv.COLOR_HLS2BGR)
        # ============================================================================================================================================================= #
        # Thresholding the Unwarpped Image
        # ============================================================================================================================================================= #
        lab = cv.cvtColor(bird_view, cv.COLOR_BGR2LAB)
        l2, a, b = cv.split(lab)  
        clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))        
        cl = clahe.apply(l2)
        ca = clahe.apply(a)
        cb = clahe.apply(b)
        l2_flag = cb > 185
        cb[l2_flag] = 255  
        cl[l2_flag] = 255
        limg = cv.merge((cl, ca, cb))
        bird_view = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
        #bird_view = cv.bilateralFilter(bird_view, 16, 100, 100)
        # ============================================================================================================================================================= #
        # Thresholding the Unwarpped Image
        # ============================================================================================================================================================= #
        warped_gray = cv.cvtColor(bird_view, cv.COLOR_BGR2GRAY)
        ret, thresh_warped = cv.threshold(warped_gray, 210, 255, cv.THRESH_BINARY)
        cv.imshow("xc,", thresh_warped)
        
        histogram = np.sum(thresh_warped[350:, 250:650], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        left_lane_base = np.where(histogram == np.amax(histogram[:midpoint]))
        right_lane_base = np.where(histogram == np.amax(histogram[midpoint:]))
        left_lane_basex = left_lane_base[0][0] + 250
        right_lane_basex = right_lane_base[0][0] + 250
        if abs(left_lane_basex - right_lane_basex) < 100 or abs(left_lane_basex - right_lane_basex) > 220:
            right_lane_basex = left_lane_basex +  200
        mask_copy = copy.copy(thresh_warped)
        cv.circle(mask_copy, (left_lane_basex, 580), 5, (255, 0, 255), -1)
        cv.circle(mask_copy, (right_lane_basex, 580), 5, (255, 0, 255), -1)

        # # Get boxes on the lane points and fit curve
        y = mask_copy.shape[1] - 30
        
        left_lanex = []
        left_laney = []
        right_lanex = []
        right_laney = []

        
        while(y > 10):

            left_lane_currentx, centroid_l = adjust_lane(left_lane_basex, y, mask_copy)
            right_lane_currentx, centroid_r = adjust_lane(right_lane_basex, y, mask_copy)
            
            if left_lane_currentx != 0:
                cv.rectangle(mask_copy, (left_lane_currentx-10, y-10),
                            (left_lane_currentx+10, y+10), (0, 255, 0), 3)
                left_lanex.append(centroid_l + left_lane_currentx-10)
                left_laney.append(y)
                cv.circle(mask_copy, (centroid_l + left_lane_currentx-10, y),
                        5, (255, 0, 0), -1)

            if right_lane_currentx != 0:
                cv.rectangle(mask_copy, (right_lane_currentx-10, y-10),
                            (right_lane_currentx+10, y+10), (0, 255, 0), 3)
                right_lanex.append(centroid_r + right_lane_currentx-10)
                right_laney.append(y)
                cv.circle(mask_copy, (centroid_r + right_lane_currentx-10, y),
                        5, (255, 0, 0), -1)
            y -= 20
        
        left, right = fit_curve(left_lanex, left_laney, right_lanex, right_laney, mask_copy)
        result = inverse_warp(key_frame, mask_copy, left, right, H_mat)
        #cv.imshow("Masked", mask_copy)
        #cv.imshow("Original", key_frame)
        cv.imshow("unwarped", result)
        # ============================================================================================================================================================= #
        # Thresholding the Unwarpped Image
        # ============================================================================================================================================================= #

        if cv.waitKey(2) == ord('q'):
            break

    # ================================================================================================================================================================= #
    # Computation Section
    # ================================================================================================================================================================= #

    cap.release()
    cv.destroyAllWindows()
