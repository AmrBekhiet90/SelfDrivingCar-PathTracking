#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._conv_rad_to_degree  = 180.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi
        self.kp                  = 3.4
        self.ki                  = 0.7
        self.kd                  = 0.5
        self.L                   = 3
        self._Kvf                = 3

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def get_distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def normalize_angle(self,angle):
        """
        Normalize an angle to [-pi, pi].
        :param angle: (float)
        :return: (float) Angle in radian in [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def get_lookahead_dis(self, v):
        return self._Kvf * v

    def set_steering_angle_stanley(self,waypoints,x,y,v,yaw):

        k_e = 0.5

        print('yaw', yaw)
        path_yaw = np.arctan2(waypoints[10][1] - waypoints[0][1], waypoints[10][0] - waypoints[0][0])
        print('yaw_path', path_yaw)
        yaw_diff = path_yaw - yaw
        yaw_diff = self.normalize_angle(yaw_diff)

        heading_error = yaw_diff
        heading_error = heading_error *(1 + 0.2 * v)
        print('heading_error', heading_error)

        current_xy = np.array([x, y])
        crosstrack_error = np.min(np.sum((current_xy - np.array(waypoints)[:, :2]) ** 2, axis=1))

        cross_error = np.sqrt(crosstrack_error)
        print('cross_error', cross_error)

        yaw_cross_track = np.arctan2(y - waypoints[0][1], x - waypoints[0][0])
        yaw_path2ct = path_yaw - yaw_cross_track

        yaw_path2ct = self.normalize_angle(yaw_path2ct)

        if yaw_path2ct > 0:
            crosstrack_error = abs(cross_error)
        else:
            crosstrack_error = - abs(cross_error)

        yaw_diff_crosstrack = np.arctan2(k_e * crosstrack_error, v)

        yaw_diff_crosstrack = yaw_diff_crosstrack * (1 + 0.5 * v)
        # Change the steer output with the lateral controller.
        steer_output = heading_error + yaw_diff_crosstrack
        steer_output = np.fmax(np.fmin(steer_output, 1.22), -1.22)
        print('steer_output', steer_output)

        return steer_output

    def set_steering_angle_pure_pursuit(self,waypoints,x,y,v,yaw,v_desired,steer_last):
        k_dd = 20.2
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(waypoints)):
            if np.abs(self.get_distance(waypoints[i][0], waypoints[i][1] , x, y)) > 0.6 * v_desired:
                min_idx = i
                break

        print('x_i',waypoints[min_idx][0])
        print('y_i',waypoints[min_idx][1])
        print('x',x)
        print('y',y)

        path_alpha = np.arctan2(waypoints[min_idx][1] - y, waypoints[min_idx][0] - x)
        path_alpha = path_alpha - yaw
        path_alpha = self.normalize_angle(path_alpha)
        print('path_alpha', path_alpha)
        steer_output = np.arctan(2 * self.L * np.sin(path_alpha)/ k_dd * v)

        # Change the steer output with the lateral controller.
        steer_output = np.fmax(np.fmin(steer_output, 1.22), -1.22)

        if np.abs(steer_output) - np.abs(steer_last) > 1.22/4:
            if steer_output > 0:
                steer_output -= 1.22/8
            elif steer_output < 0:
                steer_output += 1.22/8

        print('steer_output', steer_output)

        return steer_output

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################
        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.

            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)

            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0

            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('t_previous', 0.0)

        self.vars.create_var('error_previous', 0.0)
        self.vars.create_var('v_Integral', 0.0)

        self.vars.create_var('x_near_last', self._waypoints[0][0])
        self.vars.create_var('y_near_last', self._waypoints[0][1])

        self.vars.create_var('steer_last', steer_output)
        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position

                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a longitudinal controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """

            error = v_desired - v
            dt = self._current_timestamp - self.vars.t_previous
            v_Integral = self.vars.v_Integral + (error * dt)
            v_Derivative = (error - self.vars.error_previous) / dt

            PID_p = self.kp * error
            PID_i = self.ki * v_Integral
            PID_d = self.kd * v_Derivative

            Long_Controller_output = PID_p + PID_i + PID_d

            ffwd = v_desired
            
            # Change these outputs with the longitudinal controller. Note that
            # brake_output is optional and is not required to pass the
            # assignment, as the car will naturally slow down over time.
            throttle_output = Long_Controller_output + (ffwd * 0.0)
            brake_output    = 0

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a lateral controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """

            steer_output = self.set_steering_angle_stanley(waypoints,x,y,v,yaw)

            #steer_output = self.set_steering_angle_pure_pursuit(waypoints,x,y,v,yaw,v_desired,self.vars.steer_last)

            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)

        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.vars.v_previous = v  # Store forward speed to be used in next step
        self.vars.t_previous = self._current_timestamp  # Store forward speed to be used in next step
        self.vars.v_integral = v_Integral
        self.vars.error_previous = error
        self.vars.steer_last = steer_output

