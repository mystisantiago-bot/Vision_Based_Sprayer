
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import cv2
import numpy as np
import time
import socket
import torch
from ultralytics import YOLO
from turbojpeg import TurboJPEG
from pathlib import Path
from typing import Literal

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.event_service_pb2 import EventServiceConfigList
from farm_ng.core.event_service_pb2 import SubscribeRequest
from farm_ng.core.events_file_reader import payload_to_protobuf
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.uri_pb2 import Uri
#from farm_ng.core.event_pb2 import Event # Added this for canbus
#from farm_ng.canbus.proto.amiga_v6_pb2 import AmigaTpdo1
#from farm_ng.canbus.canbus_pb2 import KinematicState
from turbojpeg import TurboJPEG
from collections import deque

os.environ["KIVY_NO_ARGS"] = "1"

from kivy.config import Config  # noreorder # noqa: E402

Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1280")
Config.set("graphics", "height", "800")
Config.set("graphics", "fullscreen", "false")
Config.set("input", "mouse", "mouse,disable_on_activity")
Config.set("kivy", "keyboard_mode", "systemanddock")

from kivy.app import App  # noqa: E402
from kivy.lang.builder import Builder  # noqa: E402
from kivy.graphics.texture import Texture  # noqa: E402
from kivy.clock import Clock


logger = logging.getLogger("amiga.apps.camera")


class CameraApp(App):

    STREAM_NAMES = ["rgb", "disparity", "left", "right"]

    def __init__(self, service_config: EventServiceConfig) -> None:
        super().__init__()

        self.service_config = service_config
        self.image_decoder = TurboJPEG()
        self.view_name = "rgb"
        self.async_tasks: list[asyncio.Task] = []
        self.tasks: list[asyncio.Task] = []
        self.system = None
        self.last_frame = None
        self.current_roll = 90 # Default center
        self.current_pitch = 90 # Default center
        self.last_depth_frame = None
        self.last_smoothed_roll = 0
        self.last_smoothed_pitch = 0
        self.smoothed_roll = 0
        self.smoothed_pitch = 0
        self.sock = None
        self.last_send_time = 0
        self.current_pos = "0, 0"
        self.current_psi = 0
        self.current_pwm = 0
        self.spray_queue = deque()
        self.current_spray = "OFF"
        #self.update_status_label()
        self.amiga_speed = 0.0
        self.current_distance = 0.0  # in meters, distance_m is calculated in the strip loop
        #self.service_config = service_config
        #self.canbus_client = None  # Placeholder, will be set in app_func
        self.is_sweeping = False
        self.use_geometry_pitch = True

    def build(self):
        # Defines user interface and where to find file
        return Builder.load_file("src/res/main.kv")

    def on_exit_btn(self) -> None:
        """Kills the running kivy application."""
        for task in self.tasks:
            task.cancel()
        App.get_running_app().stop()

    def on_spray_btn(self) -> None:
        """Start Spraying"""
        print("Spray started...")

    def on_toggle_spray(self, state: str) -> None:
        """ Handles the ToggleButton state change."""
        if state == 'down':
            print("Spraying System Armed")
            self.root.ids.arduino_status.text = "System: ARMED / IDLE"
            self.is_sweeping = True
        else:
             print("Spraying System Disarmed")
             self.root.ids.arduino_status.text = "System: DISARMED"
             self.is_sweeping = False
             self.current_spray = "OFF"
             # Send a "Stop" command to the Arduino immediately
             if self.sock:
                 self.send_to_arduino(self.current_roll, self.current_pitch, 0,
                                     distance=self.current_distance, pwm=0)
             self.update_status_label()

    def update_view(self, view_name: str):
        self.view_name = view_name

    def calc_pwm(self, tree_detected, strip_area, roll_angle, distance=0.0, speed=0.0):
        """Calculates the PWM duty cycle for the solenoid"""
        if not tree_detected:
            return 0  # Force solenoid OFF if no target
        # Base Pulse from Roll (e.g., 90 deg center = 50ms, 0/180 deg tilt = 20ms)
        roll_offset = abs(roll_angle - 90)
        base_pulse = 50 - (roll_offset * (30 / 90))
        # Area Multiplier (Scales spray based on target size)
        # Adjust 2000 to match your 'standard' strip area
        area_mult = max(0.5, min(1.5, strip_area / 2000))
        # Distance & Speed Multipliers
        dist_mult = max(1.0, min(3.0, distance))
        speed_mult = (speed / 0.5) if speed > 0.1 else 1.0
        # Final Calculation
        final_pwm = base_pulse * area_mult * dist_mult * speed_mult
        # Return clamped value (20-100ms for your 100ms cycle)
        return int(max(20, min(100, final_pwm)))

    def send_to_arduino(self, roll, pitch, spray, distance=0.0, pwm=0):
        try:
            message = f"{int(roll)},{int(pitch)},{int(spray)},{float(distance)},{int(pwm)}"
            self.sock.sendto(message.encode(), (self.arduino_ip, self.udp_port))
            self.current_pos = f"{int(roll)},{int(pitch)}"
            self.current_spray = "SPRAYING" if spray else "OFF"
            self.current_pwm = int(pwm)
            self.update_status_label()

        except Exception as e:
            print(f"Network Error: {e}")

    def update_status_label(self):
        def update(dt):
            if self.root:
                self.root.ids.arduino_status.text = (
                    f"Dist: {getattr(self, 'current_distance_text', 'N/A')} | "
                    f"Pos: {self.current_pos} | "
                    f"Spray: {self.current_spray} | "
                    f"PSI: {self.current_psi} | "
                    f"PWM: {self.current_pwm}"
                )
        Clock.schedule_once(update)

    def track_red_line(self, red_x, red_y, frame_width, frame_height, tree_detected,distance=1.0, angle=None, spray=1, pwm=0):
        # Map pixel to servo (With Physical Offsets)
        if not tree_detected:
            self.current_roll = 90
            self.current_pitch = 90
            self.send_to_arduino(self.current_roll, self.current_pitch, 0)
            return
        # Offsets
        Z_dynamic = max(float(distance), 1.0) # in Meters, make sure it's never zero so it doesn't throw of the calcs
        # Nozzle relative to camera: [Left, Above, Behind]
        nozzle_pos = np.array([-1.1938, 0.0254, 0.1397])
        # PITCH
        pixel_y_offset = red_y - (frame_height / 2)
        V_FOV_RAD = np.deg2rad(43) # Estimated vertical FOV, Converts 43 degrees (the camera's vertical field of view) to radians.
        # Calculate tree height relative to camera center, Calculates the vertical distance (Y) from the camera to the target in meters using trigonometry, factoring in pixel displacement and distance (Z).
        Y_cam = (pixel_y_offset / (frame_height / 2)) * np.tan(V_FOV_RAD / 2) * Z_dynamic
        # nozzle_pos[1] is height offset, Adjusts the (Y) distance to account for the nozzle being higher than the camera
        Y_to_target = Y_cam - nozzle_pos[1]
        #  Calculates the necessary pitch angle in degrees using inverse tangent (arctan) of (height/distance)
        angle_y_deg = np.degrees(np.arctan2(Y_to_target, Z_dynamic))
        print(f"distance={distance}, red_y={red_y}, angle_y={angle_y_deg:.1f}")
        # Result: If nozzle is above tree, angle is negative, servo goes below 90
        if self.use_geometry_pitch:
            self.current_pitch = max(0, min(180, int(90 + angle_y_deg))) # Calculates the final servo value. It adjusts the (90) neutral point based on the angle and clamps (limits) the value between (30 degrees and 150) to prevent damage.
        else:
            self.current_pitch = max(0, min(180, int((red_y / frame_height) * 180)))

        calculated_roll = int((red_x / frame_width) * 180) #  Calculates raw angle based on horizontal position (left=0, right=180)
        self.current_roll = max(0, min(180, calculated_roll)) # Ensures the resulting servo command is within physical limits
        # SEND TO ARDUINO
        raw_pitch = int(90 + angle_y_deg)
        print(f"raw_pitch={raw_pitch} -> clamped={self.current_pitch}")

        if angle is not None:
            print(f"raw_roll(angle)={angle} -> clamped={self.current_roll}")
        else:
            print(f"raw_roll(x)={(red_x / frame_width)*180:.1f} -> clamped={self.current_roll}")
        self.send_to_arduino(self.current_roll, self.current_pitch, spray,
        distance, pwm=pwm)

    def on_touch_move(self, touch):
        # Smoothing factor, adjust as needed
        alpha = 0.2
        # Calculate smoothed coordinates
        self.smoothed_roll = (alpha * touch.x) + ((1 - alpha) * self.last_smoothed_roll)
        self.smoothed_pitch = (alpha * touch.y) + ((1 - alpha) * self.last_smoothed_pitch)
        # Update last position for next frame
        self.last_smoothed_roll, self.last_smoothed_pitch = self.smoothed_roll, self.smoothed_pitch
        # Send smoothed_roll/pitch to servo controller
        self.send_to_arduino(self.smoothed_roll, self.smoothed_pitch, 0)

    async def spray_worker(self):
        while self.root is None or not self.root.ids:
            await asyncio.sleep(0.1)
        print("UI Ready - Sweep Worker starting...")
        while True:
            try:
                btn_down = (self.root.ids.spray_btn.state == 'down')
                if not btn_down or not self.is_sweeping:
                    self.track_red_line(320, 240, 640, 480, False)
                    await asyncio.sleep(0.1)
                    continue
                if len(self.spray_queue) > 0:
                    # Take a snapshot of CURRENT frame targets only
                    targets = sorted(list(self.spray_queue), key=lambda k: k['x'])
                    self.spray_queue.clear()
                    print(f"Sweeping {len(targets)} targets")
                    # Wait before spraying so Amiga can roll into position
                    await asyncio.sleep(1.0) # Change this to delay sparyer turning on so nozzel has time to reach target
                    for target in targets:
                        if self.root.ids.spray_btn.state != 'down' or not self.is_sweeping:
                            break
                        # Move the nozzle into position, don't spray yet
                        self.track_red_line(target['x'],
                                            target['y'],
                                            640, 480,
                                            True,
                                            distance=self.current_distance,
                                            angle=target.get('angle'),spray=0,
                                            pwm=0
                                            )
                        await asyncio.sleep(0.1) # Change this to delay nozzle moving to next dot
                        # Spray now
                        self.send_to_arduino(
                            self.current_roll,
                            self.current_pitch,
                            1,
                            distance=self.current_distance,
                            pwm=target.get('pwm', 0)
                        )
                        await asyncio.sleep(0.2)  # Spray duration
                    # Done with tree stop spraying
                    self.track_red_line(320, 240, 640, 480, False)
                    self.is_sweeping = False
                    await asyncio.sleep(0.1)
                else:
                    # Idle
                    self.track_red_line(320, 240, 640, 480, False)
                    await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Sweep Worker Error: {e}")
                await asyncio.sleep(0.1)

    async def listen_to_arduino(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
                message = data.decode().split(',')
                if len(message) >= 1:
                    self.current_psi = message[0]
                    self.update_status_label() # Call the master update
            except BlockingIOError:
                pass # No data yet
            await asyncio.sleep(0.1)

    async def app_func(self):
        try:
            self.arduino_ip = "192.168.1.212" #Starlink Wifi
            self.udp_port = 12345
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
        except Exception as e:
            logger.error(f"Failed to setup UDP: {e}")

        async def run_wrapper():
            # we don't actually need to set asyncio as the lib because it is
            # the default, but it doesn't hurt to be explicit
            await self.async_run(async_lib="asyncio")
            for task in self.tasks:
                task.cancel()
        config_list = proto_from_json_file(
            self.service_config, EventServiceConfigList())
        oak0_client: EventClient | None = None
        for config in config_list.configs:

            if config.name == "oak0":
                oak0_client = EventClient(config)

        if oak0_client is None:
            raise RuntimeError(f"No {config.name} service config provided in service_config.json")

        # stream camera frames
        self.tasks: list[asyncio.Task] = [
            asyncio.create_task(self.stream_camera(oak0_client, view_name))
            for view_name in self.STREAM_NAMES
        ]
        self.tasks.append(asyncio.create_task(self.spray_worker()))
        self.tasks.append(asyncio.create_task(self.listen_to_arduino()))
        try:
           return await asyncio.gather(run_wrapper(), *self.tasks)
        except asyncio.CancelledError:
           # This is a normal shutdown, just catch it and exit quietly
            print("App is shutting down... cleaning up tasks.")
        finally:
            # Ensure the sprayer is OFF before the script fully dies
            try:
               # Center servos, stops spraying, sets distance and speed to 0
               self.send_to_arduino(90, 90, 0, distance=0.0, pwm=0)
            except:
                pass

    async def stream_camera(self, oak_client: EventClient, view_name: Literal["rgb", "disparity", "left", "right"]
= "rgb", ) -> None:
        """Subscribes to the camera service and populates the tabbed panel with all 4 image streams."""
        while self.root is None:
            await asyncio.sleep(0.01)
        rate = oak_client.config.subscriptions[0].every_n
        async for event, payload in oak_client.subscribe(SubscribeRequest(uri=Uri(path=f"oak/0/{view_name}"), every_n=rate), decode=False):
            try:
                message = payload_to_protobuf(event, payload)
                try:
                    raw_img = self.image_decoder.decode(message.image_data)
                except AttributeError:
                    raw_img = self.image_decoder.decode(message.frame)
                img = cv2.resize(raw_img, (640, 480), interpolation=cv2.INTER_AREA)
                # Capture Disparity
                if "/disparity" in event.uri.path:
                    self.last_depth_frame = img  # Grayscale disparity map
                    # display disparity map for debugging
                    self.last_frame = img
                # RGB processing block Find trees and check their distance
                if "/rgb" in event.uri.path:
               #if view_name == self.view_name:
                    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    lower_green = np.array([35, 40, 40])
                    upper_green = np.array([90, 255, 255])
                    mask = cv2.inRange(img_hsv, lower_green, upper_green)
                    mask = cv2.GaussianBlur(mask, (11, 11), 0)
                    mask = cv2.erode(mask, None, iterations=3)
                    mask = cv2.dilate(mask, None, iterations=5)
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 0:
                        self.tree_detected = True
                        largest_contour = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(largest_contour) > 1000:
                            # Only calculate centers if a target is actually found
                            M = cv2.moments(largest_contour)
                            if M["m00"] != 0:
                                center_x = int(M["m10"] / M["m00"])
                                center_y = int(M["m01"] / M["m00"])
                                # DEPTH LOGIC HERE (Inside the detection block)
                                if self.last_depth_frame is not None:
                                    # Add safety bounds check
                                    h, w = self.last_depth_frame.shape[:2]
                                    if 2 < center_x < w-2 and 2 < center_y < h-2:
                                        patch = self.last_depth_frame[center_y-2:center_y+2, center_x-2:center_x+2]
                                        disparity_val = np.mean(patch)
                                        if disparity_val > 0:
                                            # Calculate distance
                                            baseline = 0.075
                                            focal_length = 441.25
                                            distance_m = (focal_length * baseline) / disparity_val
                                            self.current_distance = distance_m
                                            self.current_distance_text = f"{distance_m:.2f} m"
                                # Create a mask of the largest tree
                                # STRIP PROCESSING
                                tree_mask = np.zeros_like(mask)
                                cv2.drawContours(tree_mask, [largest_contour], -1, 255, -1)
                                # Divide into vertical strips (e.g., every 20 pixels)
                                strip_width = 20
                                new_targets = []
                                for x in range(0, img.shape[1], strip_width):
                                    # Get a single vertical column from the mask
                                    column = tree_mask[:, x:x+strip_width]
                                    # Find all Y-coordinates where the tree exists in this strip
                                    y_indices = np.where(column > 0)[0]
                                    if len(y_indices) > 0:
                                        top_y = np.min(y_indices)    # Highest point (lowest Y)
                                        bottom_y = np.max(y_indices) # Lowest point (highest Y)
                                        center_y = int((top_y + bottom_y) / 2)
                                        center_x = x + (strip_width // 2)
                                        # Calculate the slope of the canopy edge at this strip
                                        delta_y = bottom_y - top_y
                                        delta_x = strip_width
                                        # Calculate area of the strip
                                        strip_height = bottom_y - top_y
                                        strip_area = strip_width * strip_height
                                        # Calculate the angle of the canopy line
                                        canopy_angle_rad = np.arctan2(delta_y, delta_x)
                                        roll_servo_angle = int(90 + np.degrees(canopy_angle_rad))
                                        # Draw the visual indicators
                                        # Purple dot at the top (BGR: 128, 0, 128)
                                        cv2.circle(img, (center_x, top_y), 3, (128, 0, 128), -1)
                                        # Yellow dot at the bottom (BGR: 0, 255, 255)
                                        cv2.circle(img, (center_x, bottom_y), 3, (0, 255, 255), -1)
                                        # Red dot for the Nozzle target center
                                        cv2.circle(img, (center_x, center_y), 4, (0, 0, 255), -1)
                                        # Call PWM function
                                        pwm_value = self.calc_pwm(
                                        tree_detected=True,
                                        strip_area=strip_area,
                                        roll_angle=roll_servo_angle,
                                        distance=self.current_distance,
                                        speed=0.0)
                                        # Queue the center for the sprayer
                                        new_targets.append({
                                        'x': center_x,
                                        'y': center_y,
                                        'angle': roll_servo_angle,
                                        'area': strip_area,
                                        'pwm': pwm_value})
                                self.spray_queue.clear()
                                self.spray_queue.extend(new_targets)
                                self.last_frame = img
            except Exception as e:
                logger.exception(f"Error in processing: {e}")
                continue # Skip to next frame if this one failed
            # Convert to texture (Must be OUTSIDE the except block, but INSIDE the async for)
            texture = Texture.create(size=(img.shape[1], img.shape[0]), icolorfmt="bgr")
            texture.flip_vertical()
            texture.blit_buffer(img.tobytes(), colorfmt="bgr", bufferfmt="ubyte")

            if self.root:
                self.root.ids[view_name].texture = texture


def find_config_by_name(
    service_configs: EventServiceConfigList, name: str
) -> EventServiceConfig | None:
    """Utility function to find a service config by name.

    Args:
        service_configs: List of service configs
        name: Name of the service to find
    """
    for config in service_configs.configs:
        if config.name == name:
            return config
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="template-app")
    # Add additional command line arguments here
    parser.add_argument("--service-config", type=Path, default="service_config.json")
    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    try:
        app = CameraApp(args.service_config)
        loop.run_until_complete(app.app_func())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()