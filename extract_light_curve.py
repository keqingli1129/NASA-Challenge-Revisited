import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def extract_light_curve(fits_file_path):
    """
    Extracts time, flux, and flux error from a Kepler/TESS FITS file.
    This is a simplified example. Real data requires more complex processing.
    """
    with fits.open(fits_file_path) as hdul:
        # The light curve data is usually in the first FITS extension (HDUL[1])
        data = hdul[1].data
        time = data['TIME']
        raw_flux = data['PDCSAP_FLUX']  # Pre-search Data Conditioning SAP flux
        flux_err = data['PDCSAP_FLUX_ERR']
        
        # Remove NaNs and outliers (simple version)
        finite_mask = np.isfinite(time) & np.isfinite(raw_flux)
        time = time[finite_mask]
        raw_flux = raw_flux[finite_mask]
        flux_err = flux_err[finite_mask]
        
        # Normalize the flux
        normalized_flux = raw_flux / np.nanmedian(raw_flux)
        
    return time, normalized_flux, flux_err

def main(fitsfile):
    time, flux, flux_err = extract_light_curve(fitsfile)
    plt.figure(figsize=(10, 4))
    plt.plot(time, flux, 'k-', linewidth=1)
    plt.xlabel('Time (days)')
    plt.ylabel('Normalized Flux')
    plt.title('Raw Light Curve')
    plt.show()

if __name__ == "__main__":
    main('koi_data\K00113.01\kplr002306756-2009131105131_llc.fits')