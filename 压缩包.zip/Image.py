import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Image():
    """
    Define all operations to an image.
    """
    def __init__(self):
        print('Initialized successfully!\n')

    def Getpath(self, path):
        """
        Get all .jpg files in the given path.
        Return the paths in a list.
        """
        path_list = []  
        for root, dirs, files in os.walk(path):  
            for file in files:  
                if file.lower().endswith('.jpg'):  
                    path_list.append(os.path.join(root, file))  
        return path_list

    def read(self, path, bins = 256):
        """
        Read image into a matrix.
        self.path: path of the image, public variable.
        self.bins: number of bins for histogram, 256 defaultly. Public variable.
        self.img: matrix format of the image, public variable.
        self.undo_img: get the previous image.
        """
        self.path = path
        self.bins = bins
        self.img = cv2.imread(path, 0)
        self.undo_img = self.img
        if self.img.all() != None:
            print('Read succeeded!\n')

    def Histogram(self):
        """
        Plot histogram of the image.
        self.mat_hist: matrix format of the histogram, public variable.
        """
        self.mat_hist, bins = np.histogram(self.img.ravel(), self.bins, [0,self.bins])
        hist = plt.hist(self.img.ravel(), bins = self.bins)

    def GetCDF(self):
        """
        CDF is the cumulative distribution function of an image.
        Returns an array.
        """
        area = multiply(self.img)
        self.mat_hist, bins = np.histogram(self.img.ravel(), self.bins, [0,self.bins])

        self.cdf = [0] * self.bins
        for s in range(self.bins):
            self.cdf[s] = np.sum(self.mat_hist[:s+1]) / area
        return self.cdf

    def Equalization(self):
        """
        Equalize the image and show them contrarily.
        self.res: the equalized image of matrix format. Public variable.
        self.cdf_norm: the normalized CDF. Only for plotting.
        """
        cdf = self.GetCDF()
        cdf_norm = cdf
        for i in range(self.bins):
            cdf_norm[i] = cdf[i] * max(self.mat_hist)

        self.equ = cv2.equalizeHist(self.img)
        self.res = np.hstack((self.img, self.equ))

        plt.plot(cdf_norm, color = 'b')
        plt.hist(self.res.flatten(), 256, [0, 256], color = 'r')
        plt.xlim([0,256])
        plt.legend(('CDF','Equalized histogram'), loc = 'upper left')
        plt.show()

        cv2.imshow('img',self.res)
        cv2.waitKey()
        cv2.destroyAllWindows()


    def filt(self, img, model = 'gaussian', d = 3, sigma=2, t1 = 100, t2 = 200, dx = 0, dy = 0):
        """
        Apply different filters to the given image.
        Gaussian filter defaultly.
        img: matrix format, necessary parameters.
        model: filter model, string format, 'gaussian', 'laplacian', 'canny', 'average', 'median', 'sobel', 'fourier'.
        d: an even int. Range of the filter.
        sigma: standard deviation only for Gaussian filter.
        t1, t2: thresholds for Canny filter. t1 < t2.
        dx, dy: parameters for Sobel filter. Rate of the filter.
        """
        if model == 'gaussian':
            flt = cv2.GaussianBlur(img, (d, d), sigmaX=sigma, sigmaY=sigma)
        elif model == 'laplacian':
            flt = cv2.Laplacian(img, cv2.CV_64F)
            flt = cv2.convertScaleAbs(flt)
        elif model == 'canny':
            flt = cv2.Canny(img, threshold1 = t1, threshold2 = t2)
        elif model == 'average':
            flt = cv2.blur(img, (d, d))
        elif model == 'median':
            flt = cv2.medianBlur(img, d)
        elif model == 'sobel':
            if dx * dy :
                flt = cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize=d)
            else:
                print('Error! No parameters given to dx or dy! Set dx = dy = 1 defaultly!')
                sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
                sobelx = cv2.convertScaleAbs(sobelx)

                sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
                sobely = cv2.convertScaleAbs(sobely)
                flt = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        elif model == 'fourier':
            dft = np.fft.fft2(img)
            fshift = np.fft.fftshift(dft)
            flt = np.log(np.abs(fshift))
        return flt
    
    def edgeExtract(self, img, threshold1 = 10, threshold2 = 50):
        """
        Normally works after filtering. Extract the edge from image.
        t1, t2: thresholds for Canny filter. t1 < t2.
        Smaller parameters help to extract more detailed edges. 
        """
        edges = cv2.Canny(img, threshold1, threshold2)
        return edges
    
    def show(self, img):
        "Show image in a window."
        cv2.imshow('Image(Grey-Scaled)', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def save(self, img, path):
        "Save the processed image into the given path."
        cv2.imwrite(path, img)
    
    def undo(self):
        "Undo the last operation."
        return self.undo_img

def multiply(tuple):
    """
    Get the total size of a list or a tuple.
    """
    factor = 1
    shape = tuple.shape
    for elem in shape:
        factor *= elem
    return factor


def show(img):
    "Show the image in a window."
    cv2.imshow('Image(Processed)', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def imshow(img):
    """
    Show the image in the interactive plot.
    """
    plt.imshow(img)

def Getpath(path):
    """
    Get all .jpg files in the given path.
    Return the paths in a list.
    """
    path_list = []  
    for root, dirs, files in os.walk(path):  
        for file in files:  
            if file.lower().endswith('.jpg'):  
                path_list.append(os.path.join(root, file))  
    return path_list

"""
极致的去噪 + 极小的Canny参数 = 最好的边缘
"""
