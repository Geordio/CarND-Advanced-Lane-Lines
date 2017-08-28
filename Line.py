import numpy as np
import cv2
from enum import Enum

class LaneSide(Enum):
    Left = 0
    Right = 1

# Define a class to receive the characteristics of each line detection
# implements methods to perform calculations of preoperties of the line
class Line():


    def __init__(self, laneSide):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.best_fitx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        self.currentx = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        self.lines = []
        self.count_of_lines_total = 0
        self.count_of_lines_plaus = 0
        self.count_of_lines_not_plaus = 0

        self.lane_side = laneSide
        self.no_of_lines_to_avg = 5
        # self.x_pixel_pos = None
        # self.y_pixel_pos = None

        self.plausible = False



        self.ym_per_pix = 3. / 160  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 560  # meteres per pixel in x dimension

        self.ploty = None



    # if this line is similar to the previously detected lines, and it falls within some basic checks then return true, else false
    def check_plausible_line(self, line):
        print('checking line plausibility')
        # TODO  need to tune these thresholds
        quadratic_high_thresh = 1
        quadratic_low_thresh = 0
        linear_high_thresh = 10
        linear_low_thresh = 0
        const_high_thresh = 2000
        const_low_thresh = -1000

# define thresholds for checking the new line against the last averages
        quadratic_diff_thresh = 0.005
        linear_diff_thresh = 0.5
        const_diff_thresh = 300 # was 50

        plausible = True
        if ((abs(line[0]) > quadratic_low_thresh) & (abs(line[0]) < quadratic_high_thresh)) != True:
            print('plausibility failure - quad: {}'.format(line[0]))
            plausible = False
        if ((abs(line[1]) > linear_low_thresh) & (abs(line[1]) < linear_high_thresh)) != True:
            print('plausibility failure - linear: {}'.format(line[1]))
            plausible = False

        if ((abs(line[2]) > const_low_thresh) & (abs(line[2]) < const_high_thresh)) != True:
            print('plausibility failure - const: {}'.format(line[2]))
            plausible = False

        if ((self.count_of_lines_total > 1) & (plausible == True) & (self.best_fit != None)):
            #some lines were previously found
            self.diffs = abs(self.best_fit - line)

            if (abs(self.diffs[0]) > quadratic_diff_thresh) :
                print('plausibility diff failure - quad: {}'.format(self.diffs[0]))
                plausible = False
            if (abs(self.diffs[1]) > linear_diff_thresh):
                print('plausibility diff failure - linear: {}'.format(self.diffs[1]))
                plausible = False

            if (abs(self.diffs[2]) > const_diff_thresh):
                print('plausibility diff failure - const: {}'.format(self.diffs[2]))
                plausible = False
        self.plausible = plausible

        return plausible

# return the pixels that represent the polyline
    def get_best_fit_x(self):
        return self.get_line_poly_pix(self.best_fit)



# add a candidate for a lane line to the line object.
# checks that the line is plausible against some basic rules
# creates a average fit based on the last 5 good lines
    def addline(self, newline, newline_x):

        self.current_fit = newline
        self.currentx = newline_x
        # increment the total line count
        self.count_of_lines_total = self.count_of_lines_total+ 1


        if self.check_plausible_line(newline):
            self.count_of_lines_plaus +=1
            self.lines.append(newline)

            if (len(self.lines) > self.no_of_lines_to_avg) :
                del self.lines[0]

            self.best_fit = np.average(self.lines, axis=0)
            self.best_fitx = self.get_best_fit_x()

            # self.best_fit_cr
        else:
            self.count_of_lines_not_plaus += 1

        print ('line summary: t: {}, p: {}, np: {}' .format(self.count_of_lines_total, self.count_of_lines_plaus, self.count_of_lines_not_plaus))
        max_quad = abs(np.max(self.lines, axis=0)[0])
        max_lin = abs(np.max(self.lines, axis=0)[1])
        max_const = abs(np.max(self.lines, axis=0)[2])
        self.calculate_radius_real()
        print ('fit summary: q: {}, l: {}, c: {}' .format(max_quad, max_lin, max_const))


    def set_ploty(self, ploty):
        self.ploty = ploty



    # calculates the radius in world space
    # uses best_fitx which is the pixel representation of the average polyfit
    def calculate_radius_real(self):
        y_eval = np.max(self.ploty)
        fit_cr = np.polyfit(self.ploty * self.ym_per_pix, self.best_fitx * self.xm_per_pix, 2)
        # right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        rad = ((1 + (2 * fit_cr[0] * y_eval * self.ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
        # Now our radius of curvature is in meters
        print(rad, 'm')
        rad_text = 'Radius of Curve: {:0.0f}' .format(rad)+'m'
        self.radius_of_curvature = rad
        return rad


# gets a polyline that best fits the raw pixels that were found from the image processing
    def fit_poly_lines(self, x_pixel_pos, y_pixel_pos):
        self.x_pixel_pos = x_pixel_pos
        self.y_pixel_pos = y_pixel_pos

        fit = np.polyfit(y_pixel_pos, x_pixel_pos, 2)
        return fit


# gets the pixel representation of the polyline
    def get_line_poly_pix(self, fit):
        print ('Get the pixels from the polyline')
        fitx = None
        if fit is not None:
            fitx = fit[0] * self.ploty ** 2 + fit[1] * self.ploty + fit[2]
        return fitx


# calcualte the x cordinate of the nearest point of the line to the car.
    def get_near_line_point(self):

        y_pos = 0
        nearest_x  = self.best_fit[0] * y_pos ** 2 + self.best_fit[1] * y_pos + self.best_fit[2]
        print ('nearest_x: {} ' .format(nearest_x))
        return nearest_x