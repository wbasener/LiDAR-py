import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
from PIL import Image


class lidar:
    def __init__(self, fname):
        self.las = laspy.read(fname)
        self.coords = np.asarray([self.las.x,self.las.y,self.las.z]).T
        self.nPts = len(self.las.x)
        self.cell_size = 0 # a value of - indicates thatno raster image has been created
        self.create_raster()

    def create_raster(self, cell_size = 2.5):
        # Inputs: 
        #   cell_size = size (in meters) of the edge of a square in the binning
        self.cell_size = cell_size
        
        # Get x, y, and z coordinates
        #coords = np.vstack((self.las.x, self.las.y, self.las.z)).T

        # Create the grid
        self.x_min = self.coords[:, 0].min()
        self.x_max = self.coords[:, 0].max()
        self.y_min = self.coords[:, 1].min()
        self.y_max = self.coords[:, 1].max()

        # Create bins for the x and y dimensions
        self.x_bins = np.arange(self.x_min, self.x_max, cell_size)
        self.y_bins = np.arange(self.y_min, self.y_max, cell_size)
        self.nRows = len(self.y_bins)
        self.nCols = len(self.x_bins)

        # Compute standard deviation for each cell
        self.grid_min, _, _, _ = binned_statistic_2d(
            x=self.coords[:, 0],
            y=self.coords[:, 1],
            values=self.coords[:, 2],
            statistic='min',
            bins=[self.x_bins, self.y_bins])
        # Compute standard deviation for each cell
        self.grid_max, _, _, _ = binned_statistic_2d(
            x=self.coords[:, 0],
            y=self.coords[:, 1],
            values=self.coords[:, 2],
            statistic='max',
            bins=[self.x_bins, self.y_bins])
        # Compute standard deviation for each cell
        self.grid_mean, _, _, _ = binned_statistic_2d(
            x=self.coords[:, 0],
            y=self.coords[:, 1],
            values=self.coords[:, 2],
            statistic='mean',
            bins=[self.x_bins, self.y_bins])
        # Compute standard deviation for each cell
        self.grid_stddev, _, _, _ = binned_statistic_2d(
            x=self.coords[:, 0],
            y=self.coords[:, 1],
            values=self.coords[:, 2],
            statistic='std',
            bins=[self.x_bins, self.y_bins])
        
        for data in [self.grid_min, self.grid_max, self.grid_mean, self.grid_stddev]:
            mask = np.isnan(data)
            data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
        
        # Create the dem
        self.create_dem()
        
        self.features = []
        self.feature_names = []
        self.features.append(self.grid_min)
        self.feature_names.append('min')
        self.features.append(self.grid_max)
        self.feature_names.append('max')
        self.features.append(self.grid_mean)
        self.feature_names.append('mean')
        self.features.append(self.grid_stddev)
        self.feature_names.append('stddev')
    
    def create_dem(self):
        pass
        
    
    def show_im(self, i=0):
        plt.imshow(self.features[i])
        plt.title(self.feature_names[i]+' LiDAR Heights in Each Cell')
        plt.show()            

    def save_tiff(self, fname):
        # Inputs: 
        #   fname = file name to save as
        
        if self.cell_size == 0:
            self.create_raster()            
        
        # Define the raster metadata
        transform = from_origin(self.x_min, self.y_max, self.cell_size, self.cell_size)
        profile = {
            'driver': 'GTiff',
            'dtype': rasterio.float32,
            'count': len(self.features),
            'width': self.nRows,
            'height': self.nCols,
            'transform': transform,
            'crs': self.las.header.parse_crs()
        }
        
        with rasterio.open(fname+'.tif', 'w', **profile) as dst:
            for i, [feature, feature_name] in enumerate(zip(self.features,self.feature_names)):
                dst.write_band(i+1, feature.astype('float32'))
                dst.set_band_description(1, feature_name)
    
    
    def save_png(self, fname):
        # Inputs: 
        #   fname = file name to save as
        
        if self.cell_size == 0:
            self.create_raster()  
                
        select_indices = [0,2,3]
        arrays = [
            (255*(self.features[i]-np.min(self.features[i]))/(np.max(self.features[i])-np.min(self.features[i]))).astype(np.uint8) 
             for i in select_indices]
        arr = np.stack(arrays, axis=2)
        im = Image.fromarray(arr)
        im.save(fname+'.png', quality=100, subsampling=0)