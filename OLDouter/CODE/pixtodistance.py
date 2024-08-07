import numpy as np

# Adjusted parameters
focal_length_mm = 100  # Focal length in mm
sensor_width_mm = 36  # Sensor width in mm
sensor_height_mm = 24  # Sensor height in mm
image_width_px = 4182  # Image width in pixels
image_height_px = 3264  # Image height in pixels
altitude_km = 100  # Altitude in km (further adjusted)

# Convert focal length to km
focal_length_km = focal_length_mm / 1e6

# Calculate field of view
fov_h_rad = 2 * np.arctan((sensor_width_mm / 2) / focal_length_mm)
fov_v_rad = 2 * np.arctan((sensor_height_mm / 2) / focal_length_mm)

# Ground distance covered by the entire image
ground_width_km = 2 * altitude_km * np.tan(fov_h_rad / 2)
ground_height_km = 2 * altitude_km * np.tan(fov_v_rad / 2)

# Distance per pixel
ground_distance_per_pixel_km = ground_width_km / image_width_px

# Given pixel shift
pixel_shift = 329  # Pixel shift measured

# Convert pixel distance to km
mean_distance_km = pixel_shift * ground_distance_per_pixel_km

print(f"Ground width covered by the image (km): {ground_width_km}")
print(f"Ground height covered by the image (km): {ground_height_km}")
print(f"Distance per pixel (km): {ground_distance_per_pixel_km}")
print(f"Mean change in distance (km): {mean_distance_km}")
