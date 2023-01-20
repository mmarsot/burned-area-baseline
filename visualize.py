import numpy as np
import rasterio
import matplotlib.pyplot as plt

def normalize(array):
    array_min ,array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)
    
def brighten(band):
    alpha=2
    beta=0
    return np.clip(alpha*band+beta, 0,255)

def visualize(img):
    blue = img[...,1]
    red = img[...,3]
    green = img[...,2]
    rgb_bands = [blue, green,red]

    rgb_normalized = [normalize(band) for band in rgb_bands]
    rgb_bright = [brighten(band) for band in rgb_normalized]
    
    nrg = np.dstack(rgb_bright)
   
    return nrg