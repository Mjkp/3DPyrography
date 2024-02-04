import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform
from prettytable import PrettyTable
from mpl_toolkits.mplot3d import Axes3D
import csv

# Define your specific RGB colors as rows in a 2D array
# For example, here are four colors: Red, Green, Blue, and White
rgb_colors = np.array([

#(colorfabb filament)
    [215,190,151],
    [217,191,153],
    [214,189,153],
    [213,186,145],
    [207,179,137],
    [206,177,133],
    [204,172,127],
    [196,162,117],
    [189,153,104],
    [181,142,94],
    [178,141,93],
    [172,133,83],
    [167,127,78],
    [160,120,74],
    [151,110,65],
    [140,100,57],
    [132,93,54],
    [123,86,50],
    [116,81,47],
    [109,75,43],
    [100,67,39],
    [96,65,39]
   
    ], dtype=np.uint8)
    

rgb_colors_reshaped = rgb_colors.reshape(-1, 1, 3)

# Perform the conversion
lab_colors = cv2.cvtColor(rgb_colors_reshaped, cv2.COLOR_RGB2Lab)
lab_colors = lab_colors.astype(np.float32)

# Reshape back to original shape
lab_colors = lab_colors.reshape(-1, 3)

# Convert to standard CIE Lab values
lab_colors_standard = np.zeros_like(lab_colors, dtype=np.float32)
lab_colors_standard[:, 0] = lab_colors[:, 0] / 2.55
lab_colors_standard[:, 1] = lab_colors[:, 1] - 128
lab_colors_standard[:, 2] = lab_colors[:, 2] - 128

print(lab_colors_standard)

# Extract L*, a*, and b* channels for plotting
L = lab_colors_standard[:, 0]
a = lab_colors_standard[:, 1]
b = lab_colors_standard[:, 2]

# Calculate pairwise Euclidean distances
distances = pdist(lab_colors_standard, metric='euclidean')
dist_matrix = squareform(distances)


np.savetxt("output.csv",dist_matrix,delimiter=",")

# Initialize pretty table
pt = PrettyTable()
pt.field_names = [" ",] + [f"T{i}" for i in range(len(rgb_colors))]

# Populate pretty table with upper triangular part of the matrix
for i in range(len(rgb_colors)):
    row = [f"T{i}",] + ['-' if j < i else f"{dist_matrix[i, j]:.2f}" for j in range(len(rgb_colors))]
    pt.add_row(row)

# Print the distances
print("Euclidean Distances between colors in CIE Lab space:")
print(pt)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Scatter plot
scatter = ax.scatter(a, b, L, c=rgb_colors.reshape(-1, 3) / 255.0, s=100)


ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([0, 100])

scale_x = 1
scale_y = 1
scale_z = 1

ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))

# Annotate each point with "T0", "T1", etc.
for i in range(len(rgb_colors)):
    ax.text(a[i], b[i], L[i], f"T{i}", fontsize=10)

# Set axis labels
ax.set_xlabel('a*')
ax.set_ylabel('b*')
ax.set_zlabel('L*')




plt.show()