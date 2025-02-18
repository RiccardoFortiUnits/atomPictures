
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d, griddata


def ainy(u, v, r=1):
    # Calculate the 2D Fourier transform
    rho = np.sqrt(u**2 + v**2)
    result = 2 * np.pi * r**2 * j1(2 * np.pi * r * rho) / (2 * np.pi * r * rho +.1)
    return np.abs(result)

def CoordinateChange_extended(data_tuple, x0, y0, z0, angleXY, angleXZ, angleYZ):
    (x, y, z) = data_tuple
    x, y, z = x - x0, y - y0, z - z0
    
    # Rotation around the XY plane
    y_Prime = y * np.cos(angleXY) + x * np.sin(angleXY)
    x_temp = x * np.cos(angleXY) - y * np.sin(angleXY)
    
    # Rotation around the XZ plane
    x_Prime = x_temp * np.cos(angleXZ) - z * np.sin(angleXZ)
    z_temp = x_temp * np.sin(angleXZ) + z * np.cos(angleXZ)
    
    
    # Rotation around the YZ plane
    z_Prime = y_Prime * np.sin(angleYZ) + z_temp * np.cos(angleYZ)
    y_Prime = y_Prime * np.cos(angleYZ) - z_temp * np.sin(angleYZ)
    
    return x_Prime, y_Prime, z_Prime
def reverseCoordinateChange(data_tuple, x0, y0, z0, angleXY, angleXZ, angleYZ):
    #reverts the coordinate change done by CoordinateChange_extended
    (x, y, z) = data_tuple
    angleXY = - angleXY
    angleXZ = - angleXZ
    angleYZ = - angleYZ
 
    # Rotation around the YZ plane
    z_temp = y * np.sin(angleYZ) + z * np.cos(angleYZ)
    y_temp = y * np.cos(angleYZ) - z * np.sin(angleYZ)

    # Rotation around the XZ plane
    x_temp = x * np.cos(angleXZ) - z_temp * np.sin(angleXZ)
    z_Prime = x * np.sin(angleXZ) + z_temp * np.cos(angleXZ)

    # Rotation around the XY plane
    y_Prime = x_temp * np.sin(angleXY) + y_temp * np.cos(angleXY)
    x_Prime = x_temp * np.cos(angleXY) - y_temp * np.sin(angleXY)

    return x_Prime + x0, y_Prime + y0, z_Prime + z0

'''---------------------------------------------------------------------------------'''

class pixelGrid:
    def __init__(self, xsize,ysize,nofXpixels,nofYpixels, PSF):
        #let's always work with the center of the grid being addressed as (0,0), so that it's easier to change from grid to grid
        self.xsize = xsize
        self.ysize = ysize
        self.nofXpixels = nofXpixels
        self.nofYpixels = nofYpixels
        self.pixels = np.zeros((nofXpixels,nofYpixels))
        self.PSF = PSF

    def _normalizeCoordinate(self,x,y):
        # Normalize the coordinates to the pixel grid
        x_normalized = np.round((x / self.xsize + .5) * (self.nofXpixels - 1)).astype(int)
        y_normalized = np.round((y / self.ysize + .5) * (self.nofYpixels - 1)).astype(int)
        
        # Ensure the indices are within the valid range
        x_normalized = np.clip(x_normalized, 0, self.nofXpixels - 1)
        y_normalized = np.clip(y_normalized, 0, self.nofYpixels - 1)
        return x_normalized, y_normalized

    def get_pixel(self, x, y):
        x_normalized, y_normalized = self._normalizeCoordinate(x,y)
        return self.pixels[x_normalized, y_normalized]

    def __call__(self, x, y):
        return self.get_pixel(x, y)

    def set_pixel(self, x, y, value):
        x_normalized, y_normalized = self._normalizeCoordinate(x,y)
        
        self.pixels[x_normalized, y_normalized] = value
    def getRawPositions(self):
        #gives a list of the raw position of all the lit pixels.
        coordinates = np.argwhere(self.pixels > 0)
        counts = self.pixels[self.pixels > 0].astype(int)
        coordinates = np.repeat(coordinates, counts, axis=0)
        coordinates[:,0] = (coordinates[:,0] / (self.nofXpixels - 1) - .5) * self.xsize
        coordinates[:,1] = (coordinates[:,1] / (self.nofYpixels - 1) - .5) * self.ysize
        return coordinates
    def fillFromLens(self, rawPhotonPositions, lensRadius, transformedLensRadius):#given a photon at position (lensRadius,0), the expected pixel to be hit will be the one with coordinates (transformedLensRadius,0)
        psfPhotonPosition = self.PSF(rawPhotonPositions) * transformedLensRadius / lensRadius
        x_normalized, y_normalized = self._normalizeCoordinate(psfPhotonPosition[:,0], psfPhotonPosition[:,1])
        np.add.at(self.pixels, (x_normalized, y_normalized), 1)
    def fillFromOtherGrid(self, inputGrid : 'pixelGrid'):
        psfPositions = self.PSF(inputGrid.getRawPositions())
        x_normalized, y_normalized = self._normalizeCoordinate(psfPositions[:,0], psfPositions[:,1])
        np.add.at(self.pixels, (x_normalized, y_normalized), 1)

    @staticmethod
    def looseShotsForQuantumEfficiency(shotsCoordinates, QE : float):
        mask = np.random.rand(shotsCoordinates.shape[1]) > QE
        return shotsCoordinates[mask]


        


class Camera:
    def __init__(self, position, orientation, y_range, z_range, f_orientationIsCorrect, resolution = 100):
        self.position = np.array(position)
        self.resolution = np.array(resolution)
        self.y_range = y_range
        self.z_range = z_range
        self.orientation = orientation
        self.f_orientationIsCorrect = f_orientationIsCorrect
    
    @staticmethod
    def intersect_yz_plane(start_points, direction_vectors):
        x0, y0, z0 = start_points.T
        dx, dy, dz = direction_vectors.T
                
        # Calculate the parameter t for the intersection with the YZ plane (x = 0)        
        t = -x0 / dx #it shouldn't happen that dx==0, because we should only have the photons that are oriented towards the camera

        # Calculate the intersection points
        y = y0 + t * dy
        z = z0 + t * dz
        
        # Combine the results into an array of intersection points
        return np.vstack((y, z)).T

    @staticmethod
    def hitsLens(startPoints, directionVectors, lensPosition, lensAngle, lensRadius):
        #rotate all the points and directions to the lens's reference frame
        startPoints = np.array([CoordinateChange_extended(point, *lensPosition, *lensAngle) for point in startPoints])
        directionVectors = np.array([CoordinateChange_extended(direction, 0,0,0, *lensAngle) for direction in directionVectors])
        #let's see which photons have the correct orientation to hit the lens
        correctOriented = np.where((directionVectors[:, 0] > 0) & (startPoints[:, 0] < 0))
        #let's see which photons hit the lens
        hittingPositions = Camera.intersect_yz_plane(startPoints[correctOriented], directionVectors[correctOriented])
        actuallyHitting = np.where(np.linalg.norm(hittingPositions, axis=1) <= lensRadius)
        return hittingPositions[actuallyHitting]
    
    

    def takePicture(self, photonStartPoints, photonDirections, plot = False):
        #rotate all the points and directions to the camera's reference frame
        photonStartPoints = np.array([CoordinateChange_extended(point, *self.position, *self.orientation) for point in photonStartPoints])
        photonDirections = np.array([CoordinateChange_extended(direction, 0,0,0, *self.orientation) for direction in photonDirections])
        #let's see which photons have the correct orientation to hit the camera
        correctOriented = np.where(self.f_orientationIsCorrect(photonDirections))
        #let's see which photons hit the camera
        hittingPositions = self.intersect_yz_plane(photonStartPoints[correctOriented], photonDirections[correctOriented])
        actuallyHitting = np.where((self.y_range[0] <= hittingPositions[:, 1]) & (hittingPositions[:, 1] <= self.y_range[1]) & (self.z_range[0] <= hittingPositions[:, 2]) & (hittingPositions[:, 2] <= self.z_range[1]))

        #let's fill the pixels
        # x = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        # y = np.linspace(self.z_range[0], self.z_range[1], self.resolution)
        # X, Y = np.meshgrid(x, y)
        normalizedHits = (hittingPositions[actuallyHitting][:,1:]).T
        normalizedHits[0] = np.floor((normalizedHits[0] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * self.resolution)
        normalizedHits[1] = np.floor((normalizedHits[1] - self.z_range[0]) / (self.z_range[1] - self.z_range[0]) * self.resolution)
        normalizedHits = normalizedHits.astype(int)
        image = np.zeros((self.resolution, self.resolution))
        for hit in normalizedHits.T:
            image[hit[1]][hit[0]] += 1
        if plot:
            plt.figure(figsize=(14, 12))  # Ensure the plot is always square
            plt.imshow(image, extent=(self.y_range[0], self.y_range[1], self.z_range[0], self.z_range[1]), origin='lower', cmap='viridis', aspect='auto')
            plt.xlim(self.y_range)
            plt.ylim(self.z_range)
            plt.colorbar(label='Intensity')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('2D Function Plot')
            plt.show()
        return image

class randExtractor:
    def __init__(self, distribFun, n = 1):
        self.distribFun = distribFun
        self.n = 1
    # def distribFunFromPDF(pdf, ranges, steps):
    #     #todo not working yet

    #     # Create a meshgrid for the given ranges and steps
    #     grids = [np.linspace(r[0], r[1], 1+int(np.ceil((r[1]-r[0])/s))) for r, s in zip(ranges, steps)]
    #     meshed_grids = np.meshgrid(*grids, indexing='ij')
    #     grid_points = np.stack(meshed_grids, axis=-1)

    #     # Evaluate the PDF at each point in the grid
    #     pdf_values = pdf(*[grid_points[..., i] for i in range(len(ranges))])
    #     return RegularGridInterpolator(grids, pdf_values)

    def distribFunFromPDF_2D(pdf, ranges, steps):        
        # Create a meshgrid for the given ranges and steps
        grids = [np.linspace(r[0], r[1], 1+int(np.ceil((r[1]-r[0])/s))) for r, s in zip(ranges, steps)]
        meshed_grids = np.meshgrid(*grids, indexing='ij')
        grid_points = np.stack(meshed_grids, axis=-1)

        pdf_values = pdf(*[grid_points[..., i] for i in range(len(ranges))])

        #generic probability of finding x (independently from the value of y)
        pdf_values_x = np.sum(pdf_values, axis=1)
        cdf_values_x = np.cumsum(pdf_values_x)
        cdf_values_x /= cdf_values_x[-1]

        #probability of finding y, given a fixed value for x
        cdf_values_y = np.cumsum(pdf_values, axis = 1)
        cdf_values_y = (cdf_values_y.T / cdf_values_y[ :,-1]).T
        inverse_cdf_x_interp = interp1d(cdf_values_x, grids[0], kind='cubic', fill_value="extrapolate")
        cdf_values_y[:,0]=0#if the first value is not 0, when we then interpolate it is possible that the random number is lower than this value, and thus the interpolation will return NaN
        inverse_y_points = np.dstack((meshed_grids[0], cdf_values_y)).reshape(-1, 2)#[[grids[0][i], cdf_values_y[i][j]] for i in range(len(grids[0])) for j in range(len(cdf_values_y[i]))]
        inverse_y_values = np.reshape(meshed_grids[0].T, (-1))

        def get_x_y(rand1,rand2):
            
            x = inverse_cdf_x_interp(rand1)
            y = griddata(inverse_y_points, inverse_y_values, np.column_stack((x,rand2)), method='linear')
            return x,y
        return get_x_y
        

    def __call__(self):
        return self.distribFun(np.random.random(self.n))

# positions  = np.array([[0,0,0],[0,0,0],[0,0.5,0],   [0,-0.5,0],   [0,.2,.2]])
# directions = np.array([[1,0,0],[1,0,0],[.8,-0.6,0], [.8,-0.6,0],  [1,0,0]  ])

# q=Camera.hitsLens(positions, directions, [1,0,0],[0,0,0], 1)
# print(q)

# pg = pixelGrid(.5,.5,9,9,lambda x:x)
# pg.fillFromLens(q,1,1)
# print(pg.pixels)


def gaussian(x,y, mean=0, std_dev=1):
    r=np.sqrt(x**2+y**2)
    return (1/(std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r - mean) / std_dev)**2)+ (np.sign(x)+1)*(np.sign(y)+1)/40
f=randExtractor.distribFunFromPDF_2D(ainy, [[-2,2]]*2, [.03]*2)

# plt.plot(x,y)
       
q=f(np.random.random(10000), np.random.random(10000))
plt.scatter(q[0],q[1], alpha=.03)
plt.show()


# positions  = np.array([[0,0,0],[0,0,0],[0,0.5,0],   [0,-0.5,0],   [0,.2,.2]])
# directions = np.array([[1,0,0],[1,0,0],[.8,-0.6,0], [.8,-0.6,0],  [1,0,0]  ])

# q=Camera.hitsLens(positions, directions, [1,0,0],[0,0,0], 1)
# print(q)

# pg = pixelGrid(.5,.5,9,9,randExtractor.distribFunFromPDF_2D(ainy, [[-2,2]]*2, [.13]*2))
# pg.fillFromLens(q,1,1)
# print(pg.pixels)
