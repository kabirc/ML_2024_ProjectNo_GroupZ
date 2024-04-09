import cv2
import pandas as pd
import numpy as np

# Function to generate random angles ranging from 0 to 360 degrees
def generate_random_angles(num_angles):
    return np.random.randint(0, 361, size=num_angles)

# Function to generate random widths and heights
def generate_random_sizes(num_sizes):
    return np.random.randint(10, 101, size=(num_sizes, 2))  # Generating sizes between 10 and 100 for width and height

# Function to generate rectangle points with an arbitrary angle, width, and height
def generate_rectangle_points(width, height, angle):
    # Calculate half width and half height
    half_width = width / 2
    half_height = height / 2

    # Define the corner points of the rectangle
    points = np.array([[-half_width, -half_height],
                       [half_width, -half_height],
                       [half_width, half_height],
                       [-half_width, half_height]])

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])

    # Rotate the points
    rotated_points = np.dot(points, rotation_matrix.T)

    return rotated_points

# Number of angles to generate
num_angles = 360
num_sizes = 20

# Generate random angles
real_angles = generate_random_angles(num_angles)

# Generate random widths and heights for each angle
random_sizes = np.zeros((num_angles, num_sizes, 2))
for i, angle in enumerate(real_angles):
    random_sizes[i] = generate_random_sizes(num_sizes)

# Calculate angles reported by cv2.minAreaRect() and rectangle points
reported_angles = []
rectangle_points = []

for angle, sizes in zip(real_angles, random_sizes):
    for size in sizes:
        # Generate rectangle contour points
        points = generate_rectangle_points(size[0], size[1], angle)
        rect = cv2.minAreaRect(points.astype(np.float32))  # Generate minimum area rectangle
        reported_angle = rect[-1]  # Get the angle reported by cv2.minAreaRect()
        reported_angles.append(reported_angle)
        rectangle_points.append(points)

# Flatten the list of rectangle points
rectangle_points_flat = np.concatenate(rectangle_points)

# Create DataFrame
df2 = pd.DataFrame({
    'Actual/Real Angle': np.repeat(real_angles, num_sizes),
    'Reported Angle by cv2.minAreaRect()': reported_angles,
    'Width': random_sizes[:, :, 0].flatten(),
    'Height': random_sizes[:, :, 1].flatten(),
    'Point_A_X': rectangle_points_array[:, 0, 0],
    'Point_A_Y': rectangle_points_array[:, 0, 1],
    'Point_B_X': rectangle_points_array[:, 1, 0],
    'Point_B_Y': rectangle_points_array[:, 1, 1],
    'Point_C_X': rectangle_points_array[:, 2, 0],
    'Point_C_Y': rectangle_points_array[:, 2, 1],
    'Point_D_X': rectangle_points_array[:, 3, 0],
    'Point_D_Y': rectangle_points_array[:, 3, 1]
})

df3 = pd.DataFrame({
    'Actual/Real Angle': np.repeat(real_angles, num_sizes),
    'Reported Angle by cv2.minAreaRect()': reported_angles,
    'Width': random_sizes[:, :, 0].flatten(),
    'Height': random_sizes[:, :, 1].flatten(),
    'Point_A_X': np.zeros(len(real_angles) * num_sizes),  # Set X-coordinate of Point A to 0
    'Point_A_Y': np.zeros(len(real_angles) * num_sizes),  # Set Y-coordinate of Point A to 0
    'Point_B_X': rectangle_points_array[:, 1, 0],
    'Point_B_Y': rectangle_points_array[:, 1, 1],
    'Point_C_X': rectangle_points_array[:, 2, 0],
    'Point_C_Y': rectangle_points_array[:, 2, 1],
    'Point_D_X': rectangle_points_array[:, 3, 0],
    'Point_D_Y': rectangle_points_array[:, 3, 1]
})



# Display DataFrame
print(df3)
df3.to_excel('MLprojdata2.xlsx', index=False)
rectangle_points_array = np.array(rectangle_points)
print("Rectangle points shape:", rectangle_points_array.shape)
