#!/usr/bin/env python3
# encoding: utf-8
# Soccer Match Robot Implementation

import cv2
import time
import math
import queue
import rclpy
import threading
import numpy as np
import sdk.fps as fps
import sdk.common as common
import sdk.pid as pid
from rclpy.node import Node
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from kinematics import transform
from kinematics_msgs.srv import SetRobotPose
from rclpy.executors import MultiThreadedExecutor
from interfaces.msg import ColorsInfo, ColorInfo, ColorDetect
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from interfaces.srv import SetColorDetectParam, SetString, SetCircleROI
from kinematics.kinematics_control import set_pose_target
from servo_controller.bus_servo_control import set_servo_position

class SoccerMatchNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        
        # Robot state variables
        self.running = True
        self.match_started = False
        self.match_time = 150  # 2.5 minutes in seconds
        self.match_timer = None
        self.score = 0
        self.name = name
        
        # Robot control parameters
        self.max_linear_speed = 0.3  # m/s
        self.max_angular_speed = 1.0  # rad/s
        self.z_dis = 0.36
        self.y_dis = 500
        self.x_init = transform.link3 + transform.tool_link
        
        # PID controllers for different control aspects
        self.pid_forward = pid.PID(0.005, 0.001, 0.0005)  # Control forward/backward movement
        self.pid_rotation = pid.PID(0.008, 0.001, 0.0005)  # Control rotation
        self.pid_kicker = pid.PID(0.05, 0.0, 0.0)         # Control kicker mechanism
        
        # Objects detected by vision system
        self.ball = None
        self.goal = None
        self.opponent = None
        
        # Image processing
        self.image = None
        self.image_queue = queue.Queue(maxsize=2)
        self.fps = fps.FPS()
        self.bridge = CvBridge()
        self.camera_type = 'Stereo'
        self.camera = 'depth_cam'
        self.display = self.get_parameter('enable_display').value
        
        # Configuration data
        self.lab_data = common.get_yaml_data("/home/ubuntu/share/lab_tool/lab_config.yaml")
        
        # Define ROIs for detection
        self.ball_roi = {'x_min': 0, 'x_max': 640, 'y_min': 0, 'y_max': 480}
        self.goal_roi = {'x_min': 0, 'x_max': 640, 'y_min': 0, 'y_max': 240}  # Upper half for goal
        
        # Publishers and subscribers
        self.image_sub = self.create_subscription(Image, f'/{self.camera}/rgb/image_raw', self.image_callback, 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)  # Base control
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)  # Servo control
        
        # Services
        self.create_service(Trigger, '~/start_match', self.start_match_callback)
        self.create_service(Trigger, '~/stop_match', self.stop_match_callback)
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        
        # Soccer-specific color configuration
        self.ball_color = "orange"  # Assuming orange ball
        self.goal_color = "blue"    # Assuming blue is opponent's goal
        
        # Service clients
        self.kinematics_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
        self.set_roi_client = self.create_client(SetCircleROI, '/color_detect/set_circle_roi')
        
        # Wait for services
        self.kinematics_client.wait_for_service()
        self.set_roi_client.wait_for_service()
        
        # Initialize threads
        threading.Thread(target=self.vision_processing, daemon=True).start()
        threading.Thread(target=self.robot_control, daemon=True).start()
        
        self.get_logger().info('\033[1;32m%s\033[0m' % 'Soccer Match Node Initialized')

    def get_node_state(self, request, response):
        response.success = True
        return response
        
    def image_callback(self, ros_image):
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if self.image_queue.full():
            self.image_queue.get()  # Discard oldest image if queue is full
        self.image_queue.put(bgr_image)
        
    def start_match_callback(self, request, response):
        if not self.match_started:
            self.get_logger().info('\033[1;32m%s\033[0m' % "Starting Soccer Match")
            self.match_started = True
            self.score = 0
            
            # Configure color detection for ball and goal
            self.configure_color_detection()
            
            # Start match timer
            self.match_timer = threading.Timer(self.match_time, self.end_match)
            self.match_timer.start()
            
            # Initialize robot position (center of field, facing opponent's goal)
            self.initialize_position()
            
            response.success = True
            response.message = "Soccer match started"
        else:
            response.success = False
            response.message = "Match already in progress"
        return response
        
    def stop_match_callback(self, request, response):
        if self.match_started:
            self.get_logger().info('\033[1;32m%s\033[0m' % "Stopping Soccer Match")
            self.end_match()
            response.success = True
            response.message = "Soccer match stopped"
        else:
            response.success = False
            response.message = "No match in progress"
        return response
        
    def end_match(self):
        if self.match_timer is not None:
            self.match_timer.cancel()
        self.match_started = False
        self.stop_robot()
        self.get_logger().info(f'\033[1;32mMatch ended. Final score: {self.score}\033[0m')
        
    def configure_color_detection(self):
        # Set ROI for ball detection
        ball_roi_request = SetCircleROI.Request()
        ball_roi_request.data.x_min = self.ball_roi['x_min']
        ball_roi_request.data.x_max = self.ball_roi['x_max']
        ball_roi_request.data.y_min = self.ball_roi['y_min']
        ball_roi_request.data.y_max = self.ball_roi['y_max']
        self.set_roi_client.call_async(ball_roi_request)
        
        # Set ROI for goal detection
        goal_roi_request = SetCircleROI.Request()
        goal_roi_request.data.x_min = self.goal_roi['x_min']
        goal_roi_request.data.x_max = self.goal_roi['x_max']
        goal_roi_request.data.y_min = self.goal_roi['y_min']
        goal_roi_request.data.y_max = self.goal_roi['y_max']
        self.set_roi_client.call_async(goal_roi_request)
        
    def initialize_position(self):
        # Move to center position and face opponent's goal
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.angular.z = 0.0
        self.mecanum_pub.publish(twist)
        
        # Initialize arm position for optimal ball handling
        msg = set_pose_target([self.x_init, 0.0, self.z_dis], 0.0, [-180.0, 180.0], 1.0)
        res = self.send_request(self.kinematics_client, msg)
        if res.pulse:
            servo_data = res.pulse
            set_servo_position(
                self.joints_pub, 
                1.5, 
                ((10, 500), (5, 500), (4, servo_data[3]), (3, servo_data[2]), (2, servo_data[1]), (1, servo_data[0]))
            )
            time.sleep(1.0)
        
    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()
        
    def vision_processing(self):
        """Process camera images to detect ball and goal"""
        while self.running:
            if not self.match_started:
                time.sleep(0.1)
                continue
                
            try:
                image = self.image_queue.get(block=True, timeout=0.5)
            except queue.Empty:
                continue
                
            result_image = image.copy()
            h, w = image.shape[:2]
            
            # Convert image to LAB color space for better color detection
            img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            img_blur = cv2.GaussianBlur(img_lab, (3, 3), 3)
            
            # Detect ball (orange color)
            ball_mask = self.detect_color(img_blur, self.ball_color, self.ball_roi)
            ball_contours = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
            if ball_contours:
                max_contour = common.get_area_max_contour(ball_contours, 100)[0]
                if max_contour is not None:
                    ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
                    if radius > 5:  # Minimum size threshold
                        self.ball = {
                            'x': int(x) + self.ball_roi['x_min'],
                            'y': int(y) + self.ball_roi['y_min'],
                            'radius': int(radius)
                        }
                        # Draw ball on result image
                        cv2.circle(result_image, (self.ball['x'], self.ball['y']), 
                                   self.ball['radius'], common.range_rgb[self.ball_color], 2)
                    else:
                        self.ball = None
                else:
                    self.ball = None
            else:
                self.ball = None
                
            # Detect goal (blue color)
            goal_mask = self.detect_color(img_blur, self.goal_color, self.goal_roi)
            goal_contours = cv2.findContours(goal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
            if goal_contours:
                max_contour = common.get_area_max_contour(goal_contours, 500)[0]
                if max_contour is not None:
                    x, y, w, h = cv2.boundingRect(max_contour)
                    if w > 20 and h > 20:  # Minimum size threshold
                        self.goal = {
                            'x': int(x + w/2) + self.goal_roi['x_min'],
                            'y': int(y + h/2) + self.goal_roi['y_min'],
                            'width': w,
                            'height': h
                        }
                        # Draw goal on result image
                        cv2.rectangle(result_image, 
                                     (self.goal['x'] - w//2, self.goal['y'] - h//2),
                                     (self.goal['x'] + w//2, self.goal['y'] + h//2),
                                     common.range_rgb[self.goal_color], 2)
                    else:
                        self.goal = None
                else:
                    self.goal = None
            else:
                self.goal = None
                
            # Display robot state information on image
            cv2.putText(result_image, f"Match: {'Active' if self.match_started else 'Inactive'}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_image, f"Score: {self.score}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            # Show result image and publish it
            if self.display:
                cv2.imshow("Soccer Match", result_image)
                cv2.waitKey(1)
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(result_image, "bgr8"))
            
    def detect_color(self, img, color_name, roi):
        """Detect specific color in the given ROI"""
        blob = img[roi['y_min']:roi['y_max'], roi['x_min']:roi['x_max']]
        mask = cv2.inRange(blob, 
                          tuple(self.lab_data['lab'][self.camera_type][color_name]['min']), 
                          tuple(self.lab_data['lab'][self.camera_type][color_name]['max']))
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.erode(mask, kernel)
        mask = cv2.dilate(mask, kernel)
        return mask
        
    def robot_control(self):
        """Main control loop for soccer play strategy"""
        while self.running:
            if not self.match_started:
                time.sleep(0.1)
                continue
                
            # Soccer playing strategy
            if self.ball is not None:
                # Calculate ball position relative to image center
                image_center_x = (self.ball_roi['x_max'] - self.ball_roi['x_min']) / 2
                image_center_y = (self.ball_roi['y_max'] - self.ball_roi['y_min']) / 2
                
                # Ball is in view, align with ball and move toward it
                ball_x_error = self.ball['x'] - image_center_x
                ball_y_error = self.ball['y'] - image_center_y
                
                # Check if we're close enough to the ball to kick it
                if self.ball['radius'] > 30 and abs(ball_x_error) < 30:
                    # We're close to the ball, now aim for the goal if visible
                    if self.goal is not None:
                        # Align with goal and kick ball
                        goal_x_error = self.goal['x'] - image_center_x
                        
                        # Adjust orientation to face goal
                        self.pid_rotation.SetPoint = 0
                        rotation_output = self.pid_rotation.update(goal_x_error)
                        
                        # Move forward while aiming at goal
                        twist = Twist()
                        twist.linear.x = 0.2  # Move forward
                        twist.angular.z = -rotation_output  # Adjust orientation
                        self.mecanum_pub.publish(twist)
                        
                        # Activate kicker if well aligned with goal
                        if abs(goal_x_error) < 20 and self.ball['radius'] > 40:
                            self.kick_ball()
                    else:
                        # Goal not visible, search for it by rotating
                        twist = Twist()
                        twist.linear.x = 0.0
                        twist.angular.z = 0.3  # Rotate to find goal
                        self.mecanum_pub.publish(twist)
                else:
                    # Not close enough to ball, move toward it
                    self.pid_forward.SetPoint = 0
                    self.pid_rotation.SetPoint = 0
                    
                    forward_output = self.pid_forward.update(ball_y_error)
                    rotation_output = self.pid_rotation.update(ball_x_error)
                    
                    # Create velocity command
                    twist = Twist()
                    twist.linear.x = -forward_output  # Move forward/backward
                    twist.angular.z = -rotation_output  # Adjust orientation
                    
                    # Limit speeds
                    twist.linear.x = max(min(twist.linear.x, self.max_linear_speed), -self.max_linear_speed)
                    twist.angular.z = max(min(twist.angular.z, self.max_angular_speed), -self.max_angular_speed)
                    
                    self.mecanum_pub.publish(twist)
            else:
                # Ball not visible, search for it by rotating
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.3  # Rotate to find ball
                self.mecanum_pub.publish(twist)
                
            # Sleep to maintain control loop rate
            time.sleep(0.05)
            
    def kick_ball(self):
        """Activate kicker mechanism to kick the ball"""
        # Extend kicker servo
        set_servo_position(self.joints_pub, 0.1, ((5, 800),))  # Extend kicker
        time.sleep(0.1)
        # Retract kicker servo
        set_servo_position(self.joints_pub, 0.1, ((5, 500),))  # Retract kicker
        
        # Check if we scored a goal
        # In a real implementation, this would require a more sophisticated detection
        # For now, we'll assume if the ball was close to the goal and we kicked, there's a chance we scored
        if self.ball is not None and self.goal is not None:
            # Calculate distance between ball and goal center
            dx = self.ball['x'] - self.goal['x']
            dy = self.ball['y'] - self.goal['y']
            distance = math.sqrt(dx*dx + dy*dy)
            
            # If ball is close to goal and we kicked, assume a successful goal with 30% probability
            # This is just a simple simulation - in real life you'd need actual goal detection
            if distance < 100 and np.random.random() < 0.3:
                self.score += 1
                self.get_logger().info(f'\033[1;32mGOAL! Score: {self.score}\033[0m')
                
    def stop_robot(self):
        """Stop all robot movement"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.angular.z = 0.0
        self.mecanum_pub.publish(twist)
        
        # Return to safe position
        self.initialize_position()

def main():
    node = SoccerMatchNode('soccer_match')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()
