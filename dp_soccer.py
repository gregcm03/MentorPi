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
    ATTACK_BALL = 1    # Direct approach to the ball
    DRIBBLE = 2        # Control ball while moving toward goal
    DEFEND = 3         # Block opponent from ball
    FLANK = 4          # Circle around to approach from side
    SHOOT = 5          # Fast kick toward goal

class RobotSoccerPlayer(Node):
    def __init__(self):
        super().__init__('robot_soccer_player')
        self.debug = True  # Enable debug logging
        
        ##################################################
        # Publishers
        ##################################################
        self.publisher_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        
        ##################################################
        # CV Bridge and Frame Handling
        ##################################################
        self.bridge = CvBridge()
        self.frame = None
        self.detection_frame = None  # Frame with visualization markers
        self.frame_lock = threading.Lock()
        
        ##################################################
        # Game Object Detection States
        ##################################################
        # Ball detection
        self.ball_detected = False
        self.ball_center = None
        self.ball_area = 0
        self.last_ball_detection = 0.0
        
        # Goal detection
        self.goal_detected = False
        self.goal_center = None
        self.goal_area = 0
        self.last_goal_detection = 0.0
        
        # Opponent detection
        self.opponent_detected = False
        self.opponent_center = None
        self.opponent_area = 0
        self.last_opponent_detection = 0.0
        
        ##################################################
        # Game State
        ##################################################
        self.has_ball = False  # Whether we have the ball under control
        self.ball_possession_time = 0.0
        self.current_strategy = SoccerStrategy.ATTACK_BALL
        self.last_strategy_change = 0.0
        self.strategy_min_duration = 0.5  # Min seconds to maintain a strategy
        self.goals_scored = 0
        
        # Field positions (estimated)
        self.field_position = "UNKNOWN"  # Can be DEFENSE, MIDFIELD, OFFENSE
        
        ##################################################
        # LiDAR / Obstacle Avoidance State
        ##################################################
        self.front_distance = float('inf')
        self.back_distance = float('inf')
        self.left_distance = float('inf')
        self.right_distance = float('inf')
        self.obstacle_distance_threshold = 0.3  # meters - detect obstacles
        
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
        
        # LiDAR subscription for obstacle detection
        self.create_subscription(
            LaserScan,
            '/scan_raw',
            self.lidar_callback,
            1
        )
        
        # Start display thread
        self.display_thread = threading.Thread(target=self.display_camera_feed, daemon=True)
        self.display_thread.start()
        
        # Performance tracking
        self.control_loop_times = []
        self.last_loop_time = time.time()

    ##################################################
    # Image Processing Callback
    ##################################################
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return
        
        # Create a copy for visualization
        detection_image = cv_image.copy()
        
        # Convert to HSV for better color detection
        hsv_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Process all game objects
        self.detect_ball(hsv_frame, detection_image)
        self.detect_goal(hsv_frame, detection_image)
        self.detect_opponent(hsv_frame, detection_image)
        
        # Determine if we have the ball
        self.update_ball_possession()
        
        # Add visual overlays
        self.add_game_status_overlay(detection_image)
        self.add_lidar_visualization(detection_image)
        
        with self.frame_lock:
            self.frame = cv_image
            self.detection_frame = detection_image
    
    ##################################################
    # Ball Detection (Red)
    ##################################################
    def detect_ball(self, hsv_frame, detection_image):
        current_time = time.time()
        
        # Red color often wraps around in HSV, so we need two ranges
        # Lower red range
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        
        # Upper red range
        lower_red2 = np.array([160, 100, 100])
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
        
        # Draw the ball mask in top-left corner for debugging
        small_mask = cv2.resize(ball_mask, (80, 60))
        detection_image[10:70, 10:90] = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(detection_image, "Ball", (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Reset ball detection
        self.ball_detected = False
        self.ball_center = None
        self.ball_area = 0
        
        if contours:
            # Filter by circularity and area
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 30:  # Skip tiny contours
                    continue
                    
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                    
                # Circularity = 4*pi*area/perimeter^2 (1.0 for perfect circle)
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                
                if circularity > 0.7 and area > 100:  # Must be reasonably circular
                    valid_contours.append((contour, area, circularity))
            
            # Sort by area (largest first)
            valid_contours.sort(key=lambda x: x[1], reverse=True)
            
            if valid_contours:
                # Use the largest circular contour as the ball
                largest_contour, area, circularity = valid_contours[0]
                
                # Get center using moments
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    self.ball_detected = True
                    self.ball_center = (cx, cy)
                    self.ball_area = area
                    self.last_ball_detection = current_time
                    
                    # Draw ball visualization
                    cv2.drawContours(detection_image, [largest_contour], -1, (0, 0, 255), 2)
                    cv2.circle(detection_image, (cx, cy), 5, (0, 255, 255), -1)
                    cv2.putText(detection_image, f"BALL ({area:.0f})", (cx - 40, cy - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    ##################################################
    # Goal Detection (Blue)
    ##################################################
    def detect_goal(self, hsv_frame, detection_image):
        current_time = time.time()
        
        # Blue color range
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        goal_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        
        # Noise reduction
        goal_mask = cv2.GaussianBlur(goal_mask, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        goal_mask = cv2.erode(goal_mask, kernel, iterations=1)
        goal_mask = cv2.dilate(goal_mask, kernel, iterations=2)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(goal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the goal mask in top-center for debugging
        small_mask = cv2.resize(goal_mask, (80, 60))
        detection_image[10:70, 100:180] = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(detection_image, "Goal", (100, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Reset goal detection
        self.goal_detected = False
        self.goal_center = None
        self.goal_area = 0
        
        if contours:
            # Find the largest contour (likely the goal)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 200:  # Minimum area threshold
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Goal should be wider than tall (typically)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                if aspect_ratio > 0.7:  # Could be adjusted based on your goal shape
                    self.goal_detected = True
                    self.goal_center = (x + w//2, y + h//2)
                    self.goal_area = area
                    self.last_goal_detection = current_time
                    
                    # Draw goal visualization
                    cv2.rectangle(detection_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(detection_image, "GOAL", (x + w//2 - 30, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    ##################################################
    # Opponent Detection (Orange)
    ##################################################
    def detect_opponent(self, hsv_frame, detection_image):
        current_time = time.time()
        
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
        
        # Draw the opponent mask in top-right corner for debugging
        small_mask = cv2.resize(opponent_mask, (80, 60))
        detection_image[10:70, 190:270] = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(detection_image, "Opponent", (190, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # Reset opponent detection
        self.opponent_detected = False
        self.opponent_center = None
        self.opponent_area = 0
        
        if contours:
            # Sort contours by area (largest first)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            if sorted_contours and cv2.contourArea(sorted_contours[0]) > 150:  # Minimum area threshold
                largest_contour = sorted_contours[0]
                area = cv2.contourArea(largest_contour)
                
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    self.opponent_detected = True
                    self.opponent_center = (cx, cy)
                    self.opponent_area = area
                    self.last_opponent_detection = current_time
                    
                    # Draw opponent visualization
                    cv2.drawContours(detection_image, [largest_contour], -1, (0, 165, 255), 2)
                    cv2.putText(detection_image, "OPPONENT", (cx - 50, cy - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    ##################################################
    # Ball Possession Logic
    ##################################################
    def update_ball_possession(self):
        if not self.ball_detected:
            self.has_ball = False
            self.ball_possession_time = 0.0
            return
            
        # Ball is considered "possessed" if it's large and in the bottom portion of the frame
        # This would mean the ball is very close to our robot
        frame_height = 480  # Default if no frame available
        with self.frame_lock:
            if self.frame is not None:
                frame_height = self.frame.shape[0]
        
        # Ball is possessed if it's large and in bottom third of frame
        if (self.ball_area > 2000 and 
            self.ball_center[1] > frame_height * 0.7):
            
            if not self.has_ball:
                self.has_ball = True
                self.ball_possession_time = time.time()
            # Else we already have the ball, possession time continues
        else:
            self.has_ball = False
            self.ball_possession_time = 0.0
    
    ##################################################
    # Game Status Overlay
    ##################################################
    def add_game_status_overlay(self, image):
        h, w = image.shape[:2]
        
        # Add strategy and status text
        strategy_text = f"Strategy: {self.get_strategy_name()}"
        cv2.putText(image, strategy_text, (10, h - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Ball possession status
        possession_text = "HAS BALL" if self.has_ball else "NO BALL"
        possession_color = (0, 255, 0) if self.has_ball else (0, 0, 255)
        cv2.putText(image, possession_text, (10, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, possession_color, 2)
        
        # Goals scored
        cv2.putText(image, f"GOALS: {self.goals_scored}", (w - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Line from ball to goal if both detected
        if self.ball_detected and self.goal_detected:
            cv2.line(image, self.ball_center, self.goal_center, (255, 255, 0), 2)
            
            # Calculate angle and distance
            dx = self.goal_center[0] - self.ball_center[0]
            dy = self.goal_center[1] - self.ball_center[1]
            angle = math.degrees(math.atan2(dy, dx))
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Add angle and distance text
            cv2.putText(image, f"{distance:.0f}px, {angle:.1f}°", 
                       ((self.ball_center[0] + self.goal_center[0])//2, 
                        (self.ball_center[1] + self.goal_center[1])//2 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    ##################################################
    # Strategy Name Helper
    ##################################################
    def get_strategy_name(self):
        strategy_names = {
            SoccerStrategy.ATTACK_BALL: "ATTACK BALL",
            SoccerStrategy.DRIBBLE: "DRIBBLE TO GOAL",
            SoccerStrategy.DEFEND: "DEFEND",
            SoccerStrategy.FLANK: "FLANK",
            SoccerStrategy.SHOOT: "SHOOT!"
        }
        return strategy_names.get(self.current_strategy, "UNKNOWN")
    
    ##################################################
    # LiDAR Callback and Visualization
    ##################################################
    def lidar_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        valid = ranges[np.isfinite(ranges)]
        if valid.size > 0:
            # Compute distances in front, back, left, and right sectors
            front_vals = np.concatenate((ranges[0:30], ranges[330:360]))
            self.front_distance = np.min(front_vals) if front_vals.size > 0 else float('inf')
            self.back_distance = np.min(ranges[150:210])
            self.left_distance = np.min(ranges[60:120])
            self.right_distance = np.min(ranges[240:300])
        else:
            self.front_distance = self.back_distance = self.left_distance = self.right_distance = float('inf')
    
    def add_lidar_visualization(self, image):
        try:
            # Create a small radar-like visualization in the bottom-right corner
            h, w = image.shape[:2]
            radar_size = 120
            radar_center = (w - radar_size//2 - 10, h - radar_size//2 - 10)
            
            # Draw radar background
            cv2.circle(image, radar_center, radar_size//2, (30, 30, 30), -1)
            cv2.circle(image, radar_center, radar_size//2, (50, 50, 50), 2)
            
            # Draw robot position
            cv2.circle(image, radar_center, 5, (0, 255, 255), -1)
            
            # Scale distances to fit radar visualization (max 2 meters)
            max_distance = 2.0
            scale = radar_size / 2 / max_distance
            
            # Function to draw a radar line
            def draw_radar_line(angle_deg, distance, color):
                # Skip invalid distance values
                if not math.isfinite(distance):
                    return
                    
                angle_rad = math.radians(angle_deg)
                scaled_dist = min(distance, max_distance) * scale
                
                # Calculate endpoint with bounds checking
                try:
                    end_x = int(radar_center[0] + scaled_dist * math.sin(angle_rad))
                    end_y = int(radar_center[1] - scaled_dist * math.cos(angle_rad))
                    cv2.line(image, radar_center, (end_x, end_y), color, 2)
                except (ValueError, TypeError):
                    pass
            
            # Draw radar lines for main directions
            draw_radar_line(0, self.front_distance, (0, 255, 0))    # Front
            draw_radar_line(180, self.back_distance, (0, 255, 0))   # Back
            draw_radar_line(90, self.right_distance, (0, 255, 0))   # Right
            draw_radar_line(270, self.left_distance, (0, 255, 0))   # Left
            
            # Add radar label
            cv2.putText(image, "LIDAR", (radar_center[0]-20, radar_center[1]-radar_size//2-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            # Silently handle visualization errors
            pass
    
    ##################################################
    # Display Thread: Shows camera feed continuously
    ##################################################
    def display_camera_feed(self):
        cv2.namedWindow("Robot Soccer Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Robot Soccer Vision", 800, 600)
        
        while self.running and not self.is_shutdown:
            with self.frame_lock:
                if self.detection_frame is not None:
                    cv2.imshow("Robot Soccer Vision", self.detection_frame)
            
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
    # Soccer Strategy Logic
    ##################################################
    def determine_strategy(self):
        """Determines the best strategy based on current game state."""
        current_time = time.time()
        
        # Check if we should persist with current strategy
        if current_time - self.last_strategy_change < self.strategy_min_duration:
            return self.current_strategy
            
        # If we have the ball, prioritize getting to the goal
        if self.has_ball:
            if self.goal_detected:
                # If goal is detected and we have the ball, shoot!
                new_strategy = SoccerStrategy.SHOOT
            else:
                # If we have the ball but don't see the goal, dribble and look for it
                new_strategy = SoccerStrategy.DRIBBLE
        else:
            # We don't have the ball
            if self.ball_detected:
                # Ball detected but opponent is closer to it
                if (self.opponent_detected and 
                    self.is_opponent_closer_to_ball()):
                    new_strategy = SoccerStrategy.DEFEND
                else:
                    # Ball detected and we should go for it
                    new_strategy = SoccerStrategy.ATTACK_BALL
            else:
                # No ball detected, search for it with a flanking movement
                new_strategy = SoccerStrategy.FLANK
        
        # If strategy changed, update timestamp
        if new_strategy != self.current_strategy:
            self.last_strategy_change = current_time
            self.get_logger().info(f"Strategy changed: {self.get_strategy_name()} -> {new_strategy}")
            
        return new_strategy
    
    def is_opponent_closer_to_ball(self):
        """Check if opponent is closer to the ball than we are."""
        if not (self.ball_detected and self.opponent_detected):
            return False
            
        # Calculate distances
        ball_center_y = self.ball_center[1]
        opponent_center_y = self.opponent_center[1]
        
        # In image coordinates, larger Y means closer to the camera
        # So opponent is closer if their Y is greater
        return opponent_center_y > ball_center_y
    
    ##################################################
    # Soccer Strategy Execution
    ##################################################
    def execute_strategy(self, strategy):
        """Execute the selected soccer strategy."""
        if strategy == SoccerStrategy.ATTACK_BALL:
            self.execute_attack_ball()
        elif strategy == SoccerStrategy.DRIBBLE:
            self.execute_dribble()
        elif strategy == SoccerStrategy.DEFEND:
            self.execute_defend()
        elif strategy == SoccerStrategy.FLANK:
            self.execute_flank()
        elif strategy == SoccerStrategy.SHOOT:
            self.execute_shoot()
        else:
            # Default behavior: search for ball
            self.execute_search()
    
    def execute_attack_ball(self):
        """Move directly to the ball."""
        if not self.ball_detected:
            self.execute_search()
            return
            
        # Get ball position error (how far off-center the ball is)
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
        max_speed = 2.0  # m/s
        if self.ball_area > 5000:
            speed = max_speed * 0.5  # Slow down when close
        else:
            speed = max_speed  # Full speed when far
            
        # Execute movement: forward motion with turning to center the ball
        turn_rate = normalized_error * 5.0
        self.get_logger().info(f"ATTACK_BALL: speed={speed:.2f}, turn={turn_rate:.2f}")
        self.move_robot(speed, 0.0, turn_rate, 0.05)
    
    def execute_dribble(self):
        """Control the ball while moving toward the goal."""
        # If we lost the ball, go back to attack
        if not self.has_ball:
            self.current_strategy = SoccerStrategy.ATTACK_BALL
            return
            
        # If we can see the goal, aim toward it
        if self.goal_detected:
            frame_width = 640  # Default
            with self.frame_lock:
                if self.frame is not None:
                    frame_width = self.frame.shape[1]
            
            center_x = frame_width // 2
            error_x = self.goal_center[0] - center_x
            
            # Normalize error to range [-1, 1]
            max_error = center_x
            normalized_error = -float(error_x) / max_error
            
            # Dribble toward goal with controlled speed
            speed = 1.5  # Controlled speed for dribbling
            turn_rate = normalized_error * 3.0
            
            self.get_logger().info(f"DRIBBLE: toward goal, speed={speed:.2f}, turn={turn_rate:.2f}")
            self.move_robot(speed, 0.0, turn_rate, 0.05)
        else:
            # No goal in sight, move forward while searching for goal
            # Do a gentle sweep to find the goal
            self.get_logger().info("DRIBBLE: searching for goal")
            search_direction = math.sin(time.time() * 2)  # Oscillating direction
            self.move_robot(1.0, 0.0, search_direction * 1.0, 0.05)
    
    def execute_defend(self):
        """Block opponent from reaching the ball."""
        if not (self.ball_detected and self.opponent_detected):
            # If we can't see both, go back to attack
            self.current_strategy = SoccerStrategy.ATTACK_BALL
            return
            
        # Calculate vector from opponent to ball
        dx = self.ball_center[0] - self.opponent_center[0]
        dy = self.ball_center[1] - self.opponent_center[1]
        
        # Calculate angle to intercept
        angle = math.atan2(dy, dx)
        
        # Project a position between opponent and ball
        intercept_x = self.opponent_center[0] + dx * 0.5
        intercept_y = self.opponent_center[1] + dy * 0.5
        
        # Calculate error to intercept position
        frame_width = 640  # Default
        with self.frame_lock:
            if self.frame is not None:
                frame_width = self.frame.shape[1]
        
        center_x = frame_width // 2
        error_x = intercept_x - center_x
        
        # Normalize error to range [-1, 1]
        max_error = center_x
        normalized_error = -float(error_x) / max_error
        
        # Move to intercept position
        speed = 3.0  # Fast to block
        turn_rate = normalized_error * 4.0
        
        self.get_logger().info(f"DEFEND: intercepting, speed={speed:.2f}, turn={turn_rate:.2f}")
        self.move_robot(speed, 0.0, turn_rate, 0.05)
    
    def execute_flank(self):
        """Circle around to approach the ball from a strategic angle."""
        if self.ball_detected:
            # If we see the ball, calculate the best approach angle
            
            # Check if goal is also visible
            if self.goal_detected:
                # Try to position ourselves so the ball is between us and the goal
                # First, get vector from ball to goal
                goal_to_ball_x = self.ball_center[0] - self.goal_center[0]
                goal_to_ball_y = self.ball_center[1] - self.goal_center[1]
                
                # Normalize this vector
                magnitude = math.sqrt(goal_to_ball_x**2 + goal_to_ball_y**2)
                if magnitude > 0:
                    goal_to_ball_x /= magnitude
                    goal_to_ball_y /= magnitude
                
                # Project beyond the ball to get our target position
                # (approach ball from opposite side of goal)
                target_x = self.ball_center[0] + goal_to_ball_x * 100  # Project 100px beyond ball
                target_y = self.ball_center[1] + goal_to_ball_y * 100
            else:
                # Without seeing the goal, try to get behind the ball
                # Assume bottom of frame is our robot, so approach from top
                target_x = self.ball_center[0]
                # Project a position 150px beyond the ball from our perspective
                frame_height = 480  # Default
                with self.frame_lock:
                    if self.frame is not None:
                        frame_height = self.frame.shape[1]
                
                # Target is beyond the ball from our position
                target_y = self.ball_center[1] - 150  # Go above the ball
            
            # Calculate error to target position
            frame_width = 640  # Default
            with self.frame_lock:
                if self.frame is not None:
                    frame_width = self.frame.shape[1]
            
            center_x = frame_width // 2
            error_x = target_x - center_x
            
            # Normalize error to range [-1, 1]
            max_error = center_x
            normalized_error = -float(error_x) / max_error
            
            # Execute flanking movement
            # Speed depends on how well aligned we are
            alignment = 1.0 - abs(normalized_error)  # 1.0 = perfectly aligned
            speed = 2.0 * alignment  # Slow down if not well aligned
            turn_rate = normalized_error * 3.0
            
            self.get_logger().info(f"FLANK: positioning for approach, speed={speed:.2f}, turn={turn_rate:.2f}")
            self.move_robot(speed, 0.0, turn_rate, 0.05)
        else:
            # No ball in sight, search with a wider scanning pattern
            self.execute_search()
    
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
        if abs(normalized_error) < 0.1:  # Well aligned
            # Execute powerful shot: maximum forward speed, minimal turning
            self.get_logger().info("SHOOT: GOAL!!!")
            shot_speed = 5.0  # m/s - maximum speed
            self.move_robot(shot_speed, 0.0, normalized_error * 0.5, 0.1)
            
            # Check if we might have scored
            if self.goal_area > 5000 and self.has_ball:
                # Large goal area and we have the ball = likely goal
                self.goals_scored += 1
                self.get_logger().info(f"GOAL SCORED! Total: {self.goals_scored}")
                # Small celebration spin
                self.move_robot(0.0, 0.0, 6.0, 0.3)
        else:
            # Not well aligned, adjust position
            self.get_logger().info(f"SHOOT: aligning with goal, error={normalized_error:.2f}")
            align_speed = 0.5  # Slow for precise alignment
            self.move_robot(align_speed, 0.0, normalized_error * 4.0, 0.05)
    
    def execute_search(self):
        """Search for the ball when it's not visible."""
        # Use a time-based oscillating pattern for efficient search
        t = time.time()
        
        # Decide search pattern based on when we last saw the ball
        if time.time() - self.last_ball_detection < 3.0:
            # Recently saw the ball, search nearby with gentle oscillation
            turn_rate = 1.5 * math.sin(t * 3.0)
            speed = 0.7  # Moderate forward speed while searching
            self.get_logger().info(f"SEARCH: gentle scanning, turn={turn_rate:.2f}")
            self.move_robot(speed, 0.0, turn_rate, 0.05)
        else:
            # Haven't seen ball in a while, do a more thorough search
            # Combine forward motion with oscillating rotation
            turn_rate = 3.0 * math.sin(t * 2.0)
            
            # Periodically stop to do a full spin
            if math.sin(t * 0.5) > 0.9:  # About every 6 seconds
                self.get_logger().info("SEARCH: full spin")
                self.move_robot(0.0, 0.0, 3.0, 0.3)  # Spin in place
            else:
                speed = 1.0 * (0.5 + 0.5 * math.cos(t * 0.7))  # Varying speed
                self.get_logger().info(f"SEARCH: wide scanning, speed={speed:.2f}, turn={turn_rate:.2f}")
                self.move_robot(speed, 0.0, turn_rate, 0.05)
    
    ##################################################
    # Obstacle Avoidance
    ##################################################
    def avoid_obstacles(self):
        """Check for obstacles and avoid if necessary."""
        # Check if any obstacle is too close
        obstacle_detected = (
            self.front_distance < self.obstacle_distance_threshold or
            self.left_distance < self.obstacle_distance_threshold or
            self.right_distance < self.obstacle_distance_threshold
        )
        
        if not obstacle_detected:
            return False  # No obstacle avoidance needed
            
        # Handle different obstacle scenarios
        if self.front_distance < self.obstacle_distance_threshold:
            # Front obstacle, turn away from closest side
            if self.left_distance < self.right_distance:
                # Right has more space
                self.get_logger().info("Avoiding front obstacle - turning right")
                self.move_robot(0.5, 0.0, -2.0, 0.1)
            else:
                # Left has more space
                self.get_logger().info("Avoiding front obstacle - turning left")
                self.move_robot(0.5, 0.0, 2.0, 0.1)
        elif self.left_distance < self.obstacle_distance_threshold:
            # Left obstacle, turn right
            self.get_logger().info("Avoiding left obstacle - turning right")
            self.move_robot(1.0, 0.0, -1.5, 0.1)
        elif self.right_distance < self.obstacle_distance_threshold:
            # Right obstacle, turn left
            self.get_logger().info("Avoiding right obstacle - turning left")
            self.move_robot(1.0, 0.0, 1.5, 0.1)
            
        return True  # Obstacle avoidance was performed
    
    ##################################################
    # Main Soccer Loop
    ##################################################
    def play_soccer(self):
        """Main soccer playing loop."""
        self.get_logger().info("Robot soccer player initialized.")
        
        while rclpy.ok() and self.running:
            # Track loop performance
            loop_start = time.time()
            
            # First check for obstacles - safety first
            if self.avoid_obstacles():
                # Skip strategy execution if we're avoiding obstacles
                continue
            
            # Determine best strategy for current situation
            self.current_strategy = self.determine_strategy()
            
            # Execute the selected strategy
            self.execute_strategy(self.current_strategy)
            
            # Adaptive control rate
            loop_end = time.time()
            loop_time = loop_end - loop_start
            self.control_loop_times.append(loop_time)
            
            if len(self.control_loop_times) > 100:
                avg_loop_time = sum(self.control_loop_times[-100:]) / 100
                sleep_time = max(0.001, 0.01 - avg_loop_time)
                self.control_loop_times = self.control_loop_times[-100:]
            else:
                sleep_time = 0.01
                
            time.sleep(sleep_time)
    
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
    
    print("\n" + "="*60)
    print("ROBOT SOCCER PLAYER")
    print("="*60)
    print("FEATURES:")
    print("  Red ball detection and tracking")
    print("  Blue goal detection and targeting")
    print("  Orange opponent detection and avoidance")
    print("  Multiple soccer strategies")
    print("  Obstacle avoidance")
    print("="*60)
    print(" CONTROLS:")
    print("  • Press Enter to start playing soccer")
    print("  • Press 'q' or ESC in camera window to stop")
    print("  • Press Ctrl+C in terminal for emergency stop")
    print("="*60 + "\n")
    
    try:
        input("PRESS ENTER TO START PLAYING SOCCER!")
    except KeyboardInterrupt:
        print("\nSTARTUP ABORTED ")
        player.running = False
        player.stop_robot()
        rclpy.shutdown()
        sys.exit(0)
    
    try:
        player.run_soccer()
    except KeyboardInterrupt:
        print("\nGAME INTERRUPTED")
    finally:
        player.running = False
        player.stop_robot()
        player.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("\nROBOT STOPPED - GAME OVER")
        print(f"Final Score: {player.goals_scored} goals scored")

if __name__ == '__main__':
    main()
