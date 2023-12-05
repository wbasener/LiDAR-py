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
from scipy import signal
from scipy import ndimage
from skimage import feature


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
            data = self.interpolate_nans(data)
        
        # Create the dem
        self.dem = self.create_dem()
        self.height = self.create_height()
        self.min_dem = self.create_min_dem()
        #self.edges = self.create_edges()
        
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
        self.features.append(self.dem)
        self.feature_names.append('dem')
        self.features.append(self.height)
        self.feature_names.append('height')
        self.features.append(self.min_dem)
        self.feature_names.append('min_dem')
        #self.features.append(self.edges)
        #self.feature_names.append('edges')
        
        self.nFeatures = len(self.features)
        
    def interpolate_nans(self, data):
        mask = ~np.isnan(data)
        # wile the min of the mask is False, same as while nans exists in the data
        while ~np.min(mask):
            edges = np.logical_xor(mask, ndimage.binary_dilation(mask))
            pts = np.where(edges)
            x_rng = [[np.max((x-1,0)),np.min((x+1,self.nCols))+1] for x in pts[0]]
            y_rng = [[np.max((y-1,0)),np.min((y+1,self.nRows))+1] for y in pts[1]]
            for x,y,xr,yr in zip(pts[0],pts[1],x_rng,y_rng):
                data[x,y] = np.nanmean(data[xr[0]:xr[1], yr[0]:yr[1]])
            mask = ~np.isnan(data)
        return data
        
    def create_dem(self):
        dem = ndimage.minimum_filter(self.grid_min, size=2)
        dem = ndimage.median_filter(dem, size=8)
        dem = ndimage.minimum_filter(dem, size=8)
        dem = ndimage.median_filter(dem, size=8)
        dem = ndimage.minimum_filter(dem, size=8)
        dem = ndimage.median_filter(dem, size=8)
        dem = ndimage.minimum_filter(dem, size=8)
        dem = ndimage.median_filter(dem, size=8)
        dem = ndimage.minimum_filter(dem, size=8)
        dem = ndimage.median_filter(dem, size=8)
        dem = ndimage.minimum_filter(dem, size=8)
        return dem
        
    def create_height(self):
        height = self.grid_max - self.dem
        return height             
        
    def create_min_dem(self):
        height = self.grid_min - self.dem
        return height  
    
    def create_edges(self):
        edges = feature.canny(self.min_dem, sigma=1, use_quantiles=True, low_threshold=0.901, high_threshold=0.99)
        sobel_h = ndimage.sobel(self.min_dem, 0)  # horizontal gradient
        sobel_v = ndimage.sobel(self.min_dem, 1)  # vertical gradient
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        edges = edges*1 + magnitude/np.max(magnitude)
        return edges
    
    def show_im(self, i=0):
        plt.imshow(self.features[i])
        plt.title(self.feature_names[i]+' LiDAR Heights in Each Cell')
        plt.colorbar();
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
        
        def stretch(X):
            low = np.percentile(X, 1)
            high = np.percentile(X,99.5)
            X = (X-low)/(high-low)
            X[X<0] = 0
            X[X>1] = 1
            return X
      
        if self.cell_size == 0:
            self.create_raster()  
                
        select_indices = [5,1,6]
        arrays = [
            (255*(stretch(self.features[i]))).astype(np.uint8) 
             for i in select_indices]
        arr = np.stack(arrays, axis=2)
        
        im = Image.fromarray(arr)
        im.save(fname+'.png', quality=100, subsampling=0)
        
     

    def save_tiffs(self, fname, sz=640):
        # Inputs: 
        #   fname = file name to save as
        
        if self.cell_size == 0:
            self.create_raster()     
            
        
        ncTiles = int(np.floor(self.nCols/sz))
        nrTiles = int(np.floor(self.nRows/sz))
        for r in range(nrTiles):
            for c in range(ncTiles):       
        
                # Define the raster metadata
                transform = from_origin(self.x_min, self.y_max, self.cell_size, self.cell_size)
                profile = {
                    'driver': 'GTiff',
                    'dtype': rasterio.float32,
                    'count': len(self.features),
                    'width': sz,
                    'height': sz,
                    'transform': transform,
                    'crs': self.las.header.parse_crs()
                }
                
                with rasterio.open(fname+'_'+str(r)+'_'+str(c)+'.tif', 'w', **profile) as dst:
                    for i, [feature, feature_name] in enumerate(zip(self.features,self.feature_names)):
                        chip_feature = feature[c*sz:(c+1)*sz,r*sz:(r+1)*sz]
                        dst.write_band(i+1, chip_feature.astype('float32'))
                        dst.set_band_description(1, feature_name)
    
    def save_pngs(self, fname, sz=640):      
        
        def stretch(X):
            low = np.percentile(X, 1)
            high = np.percentile(X,99.5)
            X = (X-low)/(high-low)
            X[X<0] = 0
            X[X>1] = 1
            return X
      
        if self.cell_size == 0:
            self.create_raster()  
                
        select_indices = [5,1,6]
        arrays = [
            (255*(stretch(self.features[i]))).astype(np.uint8) 
             for i in select_indices]
        arr = np.stack(arrays, axis=2)
        
        ncTiles = int(np.floor(self.nCols/sz))
        nrTiles = int(np.floor(self.nRows/sz))
        for r in range(nrTiles):
            for c in range(ncTiles):
                chip = arr[c*sz:(c+1)*sz,r*sz:(r+1)*sz,:]
                im = Image.fromarray(chip)
                im.save(fname+'_'+str(r)+'_'+str(c)+'.png', quality=100, subsampling=0)
        
        
