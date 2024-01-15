# %% 
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt

# Create a grid of x, y values
x_grid, y_grid = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
z_grid = np.sin(0.2*x_grid) + np.cos(0.4*y_grid) - np.sin(0.2*y_grid)

# Create a 3D plot of energy landscape
fig = plt.figure(figsize=(10, 8), dpi=200)
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x_grid, y_grid, z_grid, cmap='Spectral')

# Adjust the viewing angle
ax.view_init(-140, 150)
plt.axis('off')
