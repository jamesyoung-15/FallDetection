import math

def calculate_angle_with_x_axis(vector):
    x, y = vector
    angle_rad = math.atan2(y, x)
    angle_deg = math.degrees(angle_rad)
    angle_deg = int(angle_deg)
    return angle_deg

vector = (3, 4)  # Example vector with components (3, 4)
angle = calculate_angle_with_x_axis(vector)
print(angle)  # Output: 53.13010235415598 degrees