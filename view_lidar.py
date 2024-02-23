import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import laspy

#las = laspy.read('./points_lrg.las')
las = laspy.read('./points_fray_farm.laz')

points = np.asarray([las.x,las.y,las.z]).T
print(points.shape)

point_cloud = pv.PolyData(points)
point_cloud
point_cloud.plot(eye_dome_lighting=True)

surf = point_cloud.delaunay_2d()
surf.plot(show_edges=True)

