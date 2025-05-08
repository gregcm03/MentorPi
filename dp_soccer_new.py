#!/usr/bin/env python3
import time
import threading
import random
import math
import numpy as np
import cv2
import atexit
import sys
import signal

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

class SoccerStrategy:
    """Enumeration of different soccer strategies."""
    SEARCH = 0        # Search for the ball
    ATTACK_BALL = 1   # Direct approach to the ball
    DRIBBLE = 2       # Control ball while moving toward goal
    SHOOT = 3         # Fast kick toward goal
    STRIKE = 4        # Powerful hit on the ball
    RETREAT = 5       # Return to defensive position
    DEFEND = 6  
class RobotSoccerPlayer(Node):
    def __init__(self):
        super().__init__('robot_soccer_player')
        
        ##################################################
        # Publishers
        ##################################################
        self.publisher_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        
        ##################################################
        # CV Bridge and Frame Handling
        ##################################################
        self.bridge = CvBridge()
        self.frame = None
        self.frame_lock = threading.Lock()
        
        ##################################################
        # Game Object Detection States
        ##################################################
        # Ball detection
        self.ball_detected = False
        self.ball_center = None
        self.ball_area = 0
        
        # Goal detection
        self.goal_detected = False
        self.goal_center = None
        self.goal_area = 0
        
        # Opponent detection
        self.opponent_detected = False
        self.opponent_center = None
        self.opponent_area = 0
        
            # Variables for strike and retreat strategy
        self.home_position = (320, 240)  # Default center position, adjust based on your setup
        self.strike_executed = False     # Flag to track if a strike was performed
        self.strike_cooldown = 0         # Cooldown timer for strikes
        self.retreat_time = 0            # Timer for retreat phase

        ##################################################
        # Game State
        ##################################################
        self.has_ball = False
        self.current_strategy = SoccerStrategy.SEARCH
        self.goals_scored = 0
        
        ##################################################
        # Control Variables
        ##################################################
        self.running = True  # Flag to control the main loop
        
        ##################################################
        # ROS Subscribers
        ##################################################
        self.create_subscription(
            Image, 
            'ascamera/camera_publisher/rgb0/image',
            self.image_callback,
            1
        )
        
        # Start display thread
        self.display_thread = threading.Thread(target=self.display_camera_feed, daemon=True)
        self.display_thread.start()
        
        # Add debug flags to help troubleshoot
        self.debug_mode = True
        self.last_image_time = time.time()

    ##################################################
    # Image Processing Callback
    ##################################################
    def image_callback(self, msg):
        try:
            self.last_image_time = time.time()
            
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Log image received for debugging
            if self.debug_mode:
                h, w = cv_image.shape[:2]
                self.get_logger().info(f"Image received: {w}x{h}")
            
            # Create a copy for visualization
            detection_image = cv_image.copy()
            
            # Convert to HSV for better color detection
            hsv_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Add simple image stats overlay
            cv2.putText(detection_image, f"Frame: {w}x{h}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Process game objects
            self.detect_ball(hsv_frame, detection_image)
            self.detect_goal(hsv_frame, detection_image)
            self.detect_opponent(hsv_frame, detection_image)
            
            # Update ball possession
            self.update_ball_possession(detection_image)
            
            # Add strategy info
            strategy_names = {
                SoccerStrategy.SEARCH: "SEARCH",
                SoccerStrategy.ATTACK_BALL: "ATTACK BALL",
                SoccerStrategy.DRIBBLE: "DRIBBLE",
                SoccerStrategy.SHOOT: "SHOOT"
            }
            cv2.putText(detection_image, f"Strategy: {strategy_names.get(self.current_strategy, 'UNKNOWN')}", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Update the frame
            with self.frame_lock:
                self.frame = detection_image
        
        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    ##################################################
    # Ball Detection (Red)
    ##################################################
    def detect_ball(self, hsv_frame, detection_image):
        # Red color often wraps around in HSV, so we need two ranges
        # Lower red range
        lower_red1 = np.array([0, 120, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        
        # Upper red range
        lower_red2 = np.array([160, 120, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        
        # Combine masks
        ball_mask = cv2.bitwise_or(mask1, mask2)
        
        # Noise reduction
        ball_mask = cv2.GaussianBlur(ball_mask, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        ball_mask = cv2.erode(ball_mask, kernel, iterations=1)
        ball_mask = cv2.dilate(ball_mask, kernel, iterations=2)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reset ball detection
        prev_detected = self.ball_detected
        self.ball_detected = False
        
        # Draw the ball mask in top-left corner
        h, w = detection_image.shape[:2]
        small_mask = cv2.resize(ball_mask, (80, 60))
        detection_image[10:70, 10:90] = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(detection_image, "Ball Mask", (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if contours:
            # Sort by area (largest first)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Check if we have a contour large enough
            if sorted_contours and cv2.contourArea(sorted_contours[0]) > 100:
                largest_contour = sorted_contours[0]
                area = cv2.contourArea(largest_contour)
                
                # Compute circularity
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)
                    
                    # Check if it's circular enough to be a ball
                    if circularity > 0.6:  # Relaxed circularity constraint
                        M = cv2.moments(largest_contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            self.ball_detected = True
                            self.ball_center = (cx, cy)
                            self.ball_area = area
                            
                            # Draw ball visualization
                            cv2.drawContours(detection_image, [largest_contour], -1, (0, 0, 255), 2)
                            cv2.circle(detection_image, (cx, cy), 5, (0, 255, 255), -1)
                            cv2.putText(detection_image, f"BALL ({area:.0f})", (cx - 40, cy - 15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            if not prev_detected:
                                self.get_logger().info(f"Ball detected! Area: {area:.0f} at ({cx}, {cy})")
    
    ##################################################
    # Goal Detection (Green and Purple)
    ##################################################
    def detect_goal(self, hsv_frame, detection_image):
        self.goal_detected = False
        self.goal_center = None
        self.goal_area = 0

        h, w = detection_image.shape[:2]

        # --- GREEN GOAL ---
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        green_mask = cv2.GaussianBlur(green_mask, (5, 5), 0)
        green_mask = cv2.erode(green_mask, None, iterations=1)
        green_mask = cv2.dilate(green_mask, None, iterations=2)

        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if green_contours:
            largest = max(green_contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 200:
                x, y, w_g, h_g = cv2.boundingRect(largest)
                self.goal_detected = True
                self.goal_center = (x + w_g // 2, y + h_g // 2)
                self.goal_area = cv2.contourArea(largest)
                cv2.rectangle(detection_image, (x, y), (x + w_g, y + h_g), (0, 255, 0), 2)
                cv2.putText(detection_image, "GREEN GOAL", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- PURPLE GOAL ---
        lower_purple = np.array([125, 50, 50])
        upper_purple = np.array([150, 255, 255])
        purple_mask = cv2.inRange(hsv_frame, lower_purple, upper_purple)
        purple_mask = cv2.GaussianBlur(purple_mask, (5, 5), 0)
        purple_mask = cv2.erode(purple_mask, None, iterations=1)
        purple_mask = cv2.dilate(purple_mask, None, iterations=2)

        purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if purple_contours:
            largest = max(purple_contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 200:
                x, y, w_p, h_p = cv2.boundingRect(largest)
                self.goal_detected = True
                self.goal_center = (x + w_p // 2, y + h_p // 2)
                self.goal_area = cv2.contourArea(largest)
                cv2.rectangle(detection_image, (x, y), (x + w_p, y + h_p), (255, 0, 255), 2)
                cv2.putText(detection_image, "PURPLE GOAL", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    
    ##################################################
    # Opponent Detection (Orange)
    ##################################################
    def detect_opponent(self, hsv_frame, detection_image):
        # Orange color range
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([25, 255, 255])
        opponent_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)
        
        # Noise reduction
        opponent_mask = cv2.GaussianBlur(opponent_mask, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        opponent_mask = cv2.erode(opponent_mask, kernel, iterations=1)
        opponent_mask = cv2.dilate(opponent_mask, kernel, iterations=2)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(opponent_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reset opponent detection
        prev_detected = self.opponent_detected
        self.opponent_detected = False
        
        # Draw the opponent mask in top-right corner
        h, w = detection_image.shape[:2]
        small_mask = cv2.resize(opponent_mask, (80, 60))
        detection_image[10:70, 190:270] = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(detection_image, "Opponent", (190, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        if contours:
            # Sort contours by area (largest first)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            if sorted_contours and cv2.contourArea(sorted_contours[0]) > 150:
                largest_contour = sorted_contours[0]
                area = cv2.contourArea(largest_contour)
                
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    self.opponent_detected = True
                    self.opponent_center = (cx, cy)
                    self.opponent_area = area
                    
                    # Draw opponent visualization
                    cv2.drawContours(detection_image, [largest_contour], -1, (0, 165, 255), 2)
                    cv2.putText(detection_image, "OPPONENT", (cx - 50, cy - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    
                    if not prev_detected:
                        self.get_logger().info(f"Opponent detected! Area: {area:.0f} at ({cx}, {cy})")
    
    ##################################################
    # Ball Possession Logic
    ##################################################
    def update_ball_possession(self, image):
        h, w = image.shape[:2]
        
        if not self.ball_detected:
            self.has_ball = False
            return
            
        # Ball is considered "possessed" if it's large and in the bottom portion of the frame
        if (self.ball_area > 1500 and 
            self.ball_center[1] > h * 0.7):
            
            self.has_ball = True
            cv2.putText(image, "BALL POSSESSED!", (w//2 - 80, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            self.has_ball = False
    
    ##################################################
    # Display Thread: Shows camera feed continuously
    ##################################################
    def display_camera_feed(self):
        cv2.namedWindow("Robot Soccer Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Robot Soccer Vision", 800, 600)
        
        while self.running and not self.is_shutdown:
            with self.frame_lock:
                current_frame = self.frame
            
            if current_frame is not None:
                cv2.imshow("Robot Soccer Vision", current_frame)
            else:
                # Display a blank frame with text if no image is available
                blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
                cv2.putText(blank_frame, "Waiting for camera feed...", (150, 240), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Robot Soccer Vision", blank_frame)
                
                # Debug info for camera issues
                time_since_last = time.time() - self.last_image_time
                if time_since_last > 5.0 and self.debug_mode:
                    self.get_logger().warn(f"No camera image for {time_since_last:.1f} seconds")
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' key
                self.running = False
                print("Exit key pressed. Stopping robot and exiting program.")
                self.stop_robot()
                # Signal the main thread to exit
                if not self.is_shutdown:
                    self.get_logger().info("User requested exit. Shutting down...")
                    rclpy.shutdown()
                    
            time.sleep(0.01)
    
    @property
    def is_shutdown(self):
        return not rclpy.ok()

    ##################################################
    # Movement Functions
    ##################################################
    def move_robot(self, linear_x, linear_y, angular_z, duration):
        if not self.running:
            return
            
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.angular.z = float(angular_z)
        
        # Log significant movements
        if abs(linear_x) > 0.1 or abs(angular_z) > 0.1:
            self.get_logger().debug(f"Moving: x={linear_x:.2f}, ang={angular_z:.2f}")
        
        start_time = time.time()
        rate = 0.01  # 100Hz control rate
        while (time.time() - start_time < duration) and rclpy.ok() and self.running:
            self.publisher_cmd_vel.publish(msg)
            time.sleep(rate)

    def stop_robot(self):
        if hasattr(self, 'publisher_cmd_vel') and self.publisher_cmd_vel:
            msg = Twist()
            for _ in range(5):  # Send multiple stop commands
                self.publisher_cmd_vel.publish(msg)
                time.sleep(0.01)
    
    ##################################################
    # Soccer Strategy Implementation
    ##################################################
    def determine_strategy(self):
        if self.strike_cooldown > 0:
            self.strike_cooldown -= 1
    
    # Check if we need to retreat after a strike
        if self.strike_executed:
            return SoccerStrategy.RETREAT
        
    # If we're at home position after retreat, switch to defend
        if (self.current_strategy == SoccerStrategy.RETREAT and self.is_near_home_position()):
            return SoccerStrategy.DEFEND
        
    # If ball detected while defending, consider striking
        if (self.current_strategy == SoccerStrategy.DEFEND and self.ball_detected and self.strike_cooldown == 0):
            if self.is_ball_in_strike_zone():
                return SoccerStrategy.STRIKE
    
    # If no special conditions, use the default strategy logic
        if not self.ball_detected:
             SoccerStrategy.SEARCH
        
        if self.has_ball:
            if self.goal_detected:
                return SoccerStrategy.SHOOT
            else:
                return SoccerStrategy.DRIBBLE
        else:
        # Only attack if we're not defending or the ball is very close
            if (self.current_strategy != SoccerStrategy.DEFEND or self.is_ball_close()):
                return SoccerStrategy.ATTACK_BALL
            else:
                return SoccerStrategy.DEFEND

    def is_near_home_position(self):
        """Check if robot is near its home/defensive position."""
        if not hasattr(self, 'robot_position'):
            # Estimate robot position at frame center bottom
            h, w = 480, 640  # Default values
            with self.frame_lock:
                if self.frame is not None:
                    h, w = self.frame.shape[:2]
            self.robot_position = (w//2, h-50)
    
        # Calculate distance to home position (simple estimate)
        dx = self.robot_position[0] - self.home_position[0]
        dy = self.robot_position[1] - self.home_position[1]
        distance = math.sqrt(dx*dx + dy*dy)
    
        return distance < 50  # Threshold for "near enough"

    def is_ball_in_strike_zone(self):
        """Check if the ball is in a good position to strike."""
        if not self.ball_detected:
            return False
        
        # Get frame dimensions
        h, w = 480, 640  # Default values
        with self.frame_lock:
            if self.frame is not None:
                h, w = self.frame.shape[:2]
    
        # Define strike zone (middle third of field, front half)
        x_min = w // 3
        x_max = 2 * w // 3
        y_min = h // 2
        y_max = h
    
        # Check if ball is in strike zone
        return (x_min < self.ball_center[0] < x_max and y_min < self.ball_center[1] < y_max)

    def is_ball_close(self):
        """Check if the ball is very close to the robot."""
        return self.ball_detected and self.ball_area > 2000  # Adjust threshold as needed

    def execute_strategy(self):
        """Execute the current soccer strategy."""
        # Update current strategy
        new_strategy = self.determine_strategy()
        if new_strategy != self.current_strategy:
            self.get_logger().info(f"Changing strategy to: {new_strategy}")
            self.current_strategy = new_strategy
    
        # Execute the appropriate strategy
        if self.current_strategy == SoccerStrategy.SEARCH:
            self.execute_search()
        elif self.current_strategy == SoccerStrategy.ATTACK_BALL:
            self.execute_attack_ball()
        elif self.current_strategy == SoccerStrategy.DRIBBLE:
            self.execute_dribble()
        elif self.current_strategy == SoccerStrategy.SHOOT:
            self.execute_shoot()
        elif self.current_strategy == SoccerStrategy.STRIKE:
            self.execute_strike()
        elif self.current_strategy == SoccerStrategy.RETREAT:
            self.execute_retreat()
        elif self.current_strategy == SoccerStrategy.DEFEND:
            self.execute_defend()
    
    def execute_search(self):
        """Search for the ball when it's not visible."""
        # Simple spin search
        self.get_logger().info("Searching for ball - spinning")
        self.move_robot(0.0, 0.0, 1.5, 0.1)  # Spin in place
    
    def execute_attack_ball(self):
        """Move directly to the ball."""
        if not self.ball_detected:
            self.execute_search()
            return
            
        # Calculate error (how far off-center the ball is)
        frame_width = 640  # Default
        with self.frame_lock:
            if self.frame is not None:
                frame_width = self.frame.shape[1]
        
        center_x = frame_width // 2
        error_x = self.ball_center[0] - center_x
        
        # Normalize error to range [-1, 1]
        max_error = center_x
        normalized_error = -float(error_x) / max_error
        
        # Set speed based on ball area (distance)
        # Larger area = closer = slower approach
        max_speed = 1.5  # m/s
        if self.ball_area > 3000:
            speed = max_speed * 0.5  # Slow down when close
        else:
            speed = max_speed  # Full speed when far
            
        # Execute movement: forward motion with turning to center the ball
        turn_rate = normalized_error * 3.0
        
        self.get_logger().info(f"ATTACK_BALL: ball at ({self.ball_center[0]}, {self.ball_center[1]}), "
                              f"area={self.ball_area:.0f}, error={normalized_error:.2f}")
        self.move_robot(speed, 0.0, turn_rate, 0.1)
    
    def execute_dribble(self):
        """Control the ball while moving toward the goal."""
        # If we lost the ball, go back to attack
        if not self.has_ball:
            self.current_strategy = SoccerStrategy.ATTACK_BALL
            return
            
        # Simple dribbling - move forward with gentle turns to find goal
        self.get_logger().info("DRIBBLE: searching for goal")
        # Oscillating direction search for goal
        search_direction = 1.0 * math.sin(time.time() * 2)
        self.move_robot(0.8, 0.0, search_direction, 0.1)
    
    def execute_shoot(self):
        """Shoot the ball toward the goal with high speed."""
        if not (self.has_ball and self.goal_detected):
            # If we lost the ball or can't see the goal, go back to appropriate strategy
            if self.has_ball:
                self.current_strategy = SoccerStrategy.DRIBBLE
            else:
                self.current_strategy = SoccerStrategy.ATTACK_BALL
            return
            
        # Calculate direction to goal
        frame_width = 640  # Default
        with self.frame_lock:
            if self.frame is not None:
                frame_width = self.frame.shape[1]
        
        center_x = frame_width // 2
        error_x = self.goal_center[0] - center_x
        
        # Normalize error to range [-1, 1]
        max_error = center_x
        normalized_error = -float(error_x) / max_error
        
        # Check if we're aligned with the goal
        if abs(normalized_error) < 0.2:  # Well aligned
            # Execute powerful shot: maximum forward speed, minimal turning
            self.get_logger().info("SHOOT: GOAL!!!")
            shot_speed = 3.0  # m/s - high speed
            self.move_robot(shot_speed, 0.0, normalized_error * 0.5, 0.3)
            
            # Check if we might have scored
            if self.goal_area > 5000 and self.has_ball:
                # Large goal area and we have the ball = likely goal
                self.goals_scored += 1
                self.get_logger().info(f"GOAL SCORED! Total: {self.goals_scored}")
        else:
            # Not well aligned, adjust position
            self.get_logger().info(f"SHOOT: aligning with goal, error={normalized_error:.2f}")
            align_speed = 0.5  # Slow for precise alignment
            self.move_robot(align_speed, 0.0, normalized_error * 2.0, 0.1)
    def execute_strike(self):
        """Execute a powerful strike on the ball."""
        if not self.ball_detected:
            self.current_strategy = SoccerStrategy.SEARCH
            return
        
        # Calculate direction to ball
        frame_width = 640  # Default
        with self.frame_lock:
            if self.frame is not None:
                frame_width = self.frame.shape[1]
    
        center_x = frame_width // 2
        error_x = self.ball_center[0] - center_x
    
        # Normalize error to range [-1, 1]
        max_error = center_x
        normalized_error = -float(error_x) / max_error
    
        # If well aligned with ball, execute strike
        if abs(normalized_error) < 0.2:
            # Execute strike: maximum forward speed, minimal turning
            self.get_logger().info("STRIKE: Hitting ball with force!")
            strike_speed = 2.5  # m/s - high speed strike
            self.move_robot(strike_speed, 0.0, normalized_error * 0.5, 0.5)  # Longer duration for momentum
        
            # Mark strike as executed and set cooldown
            self.strike_executed = True
            self.strike_cooldown = 50  # Approx 5 seconds at 10Hz loop
            self.retreat_time = time.time()
        else:
            # Not well aligned, adjust position
            self.get_logger().info(f"STRIKE: aligning with ball, error={normalized_error:.2f}")
            align_speed = 0.3  # Slow for precise alignment
            self.move_robot(align_speed, 0.0, normalized_error * 2.0, 0.1)

    def execute_retreat(self):
        """Retreat to defensive position after striking."""
        # If we've been retreating too long, force transition to defend
        if time.time() - self.retreat_time > 5.0:  # 5 second timeout
            self.strike_executed = False
            self.current_strategy = SoccerStrategy.DEFEND
            return
        
        # Calculate direction to home position
        frame_width, frame_height = 640, 480  # Default
        with self.frame_lock:
            if self.frame is not None:
                frame_height, frame_width = self.frame.shape[:2]
    
        # Set home position (adjust as needed for your field)
        # This is typically near your own goal
        self.home_position = (frame_width // 2, int(frame_height * 0.8))
    
        # Simple retreat: move backward (negative X)
        self.get_logger().info("RETREAT: Returning to defensive position")
        self.move_robot(-1.0, 0.0, 0.0, 0.2)
    
        # Check if we've reached home position (approximate)
        if self.is_near_home_position():
            self.get_logger().info("RETREAT: Reached home position")
            self.strike_executed = False
            self.current_strategy = SoccerStrategy.DEFEND

    def execute_defend(self):
        """Stay in defensive position and track ball movement."""
        # If ball not detected, do a minimal search while staying in position
        if not self.ball_detected:
            self.get_logger().info("DEFEND: Searching for ball while defending")
            self.move_robot(0.0, 0.0, 0.5, 0.1)  # Gentle rotation
            return
        
        # Calculate direction to ball
        frame_width = 640  # Default
        with self.frame_lock:
            if self.frame is not None:
                frame_width = self.frame.shape[1]
    
        center_x = frame_width // 2
        error_x = self.ball_center[0] - center_x
    
        # Normalize error to range [-1, 1]
        max_error = center_x
        normalized_error = -float(error_x) / max_error
    
        # Just turn to face the ball, minimal movement
        self.get_logger().info(f"DEFEND: Tracking ball, error={normalized_error:.2f}")
        self.move_robot(0.0, 0.0, normalized_error * 1.5, 0.1)
    ##################################################
    # Main Soccer Loop
    ##################################################
    def play_soccer(self):
        """Main soccer playing loop."""
        self.get_logger().info("Robot soccer player initialized. Starting game loop.")
        
        while rclpy.ok() and self.running:
            try:
                # Execute the current strategy
                self.execute_strategy()
                
                # Short sleep to prevent CPU overload
                time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f"Error in play loop: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())
    
    ##################################################
    # Main Soccer Entry
    ##################################################
    def run_soccer(self):
        self.get_logger().info("⚽ ROBOT SOCCER PLAYER INITIALIZED ⚽")
        
        try:
            self.play_soccer()
        except Exception as e:
            self.get_logger().error(f"Error in soccer routine: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            self.stop_robot()

##################################################
# Shutdown Hook for Safe Stop
##################################################
def shutdown_hook(player):
    player.running = False
    if hasattr(player, 'stop_robot'):
        player.stop_robot()
        player.get_logger().info("Robot stopped safely.")
    
    # Close any open windows
    cv2.destroyAllWindows()

##################################################
# Signal Handler to catch Ctrl+C
##################################################
def signal_handler(sig, frame):
    print("\n⚠️ EMERGENCY STOP ACTIVATED ⚠️")
    print("Stopping robot and shutting down...")
    rclpy.shutdown()
    sys.exit(0)

##################################################
# Main Function with multiple exit mechanisms
##################################################
def main(args=None):
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    rclpy.init(args=args)
    player = RobotSoccerPlayer()
    atexit.register(shutdown_hook, player)
    
    # Start ROS spinning in a daemon thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(player,), daemon=True)
    spin_thread.start()
    
    
    try:
        input("⚽ PRESS ENTER TO START PLAYING SOCCER! ⚽")
    except KeyboardInterrupt:
        print("\n⚠️ STARTUP ABORTED ⚠️")
        player.running = False
        player.stop_robot()
        rclpy.shutdown()
        sys.exit(0)
    
    try:
        player.run_soccer()
    except KeyboardInterrupt:
        print("\n⚠️ GAME INTERRUPTED ⚠️")
    finally:
        player.running = False
        player.stop_robot()
        player.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("\n✅ ROBOT STOPPED - GAME OVER ✅")
        print(f"Final Score: {player.goals_scored} goals scored")

if __name__ == '__main__':
    main()
