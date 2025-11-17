import warnings

def suppress_rasterio_warnings():
    warnings.filterwarnings("ignore", message="Dataset has no geotransform", category=UserWarning)

