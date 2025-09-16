import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk
from astropy.io import fits
import os
from PIL import Image
import io

# 1. Define the Dataset Class
class LightCurveDataset(Dataset):
    def __init__(self, fits_files, labels, transform=None, img_size=(768, 512)):
        """
        Args:
            fits_files (list): List of paths to FITS files
            labels (list): Corresponding labels (1 for exoplanet, 0 for none)
            transform (callable, optional): Optional transform to be applied on image
            img_size (tuple): Size of the output image
        """
        self.fits_files = fits_files
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
        
    def __len__(self):
        return len(self.fits_files)
    
    def __getitem__(self, idx):
        fits_path = self.fits_files[idx]
        label = self.labels[idx]
        
        try:
            # # Load and process light curve
            # with fits.open(fits_path) as hdul:
            #     # Simple extraction - in practice, use lightkurve for robust processing
            #     data = hdul[1].data
            #     time = data['TIME']
            #     flux = data['PDCSAP_FLUX']
                
            #     # Remove NaNs
            #     mask = np.isfinite(time) & np.isfinite(flux)
            #     time = time[mask]
            #     flux = flux[mask]
                
            #     # Normalize
            #     flux = flux / np.nanmedian(flux)
                
            # # Create folded light curve image
            # image = self.create_folded_image(time, flux)
            image = self.create_raw_fft_images(fits_path)
            if self.transform:
                image = self.transform(image)
                
            return fits_path, image, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error processing {fits_path}: {e}")
            # Return a dummy image and label
            dummy_image = torch.zeros(3, self.img_size[0], self.img_size[1])
            return dummy_image, torch.tensor(0.0, dtype=torch.float32)
    
    def create_folded_image(self, time, flux):
        """Create a folded light curve image using BLS periodogram"""
        try:
            # Find period using BLS
            model = BoxLeastSquares(time, flux)
            periodogram = model.autopower(0.1)  # Assume 0.1-day duration
            best_period = periodogram.period[np.argmax(periodogram.power)]
            
            # Fold the light curve
            folded_time = (time % best_period) / best_period
            
            # Create figure in memory
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 4))
            
            # Plot periodogram
            ax1.plot(periodogram.period, periodogram.power, 'b-', linewidth=1)
            ax1.axvline(best_period, color='r', linestyle='--', alpha=0.7)
            ax1.set_title(f'Period: {best_period:.2f} days')
            ax1.set_xlabel('Period (days)')
            ax1.set_ylabel('Power')
            
            # Plot folded light curve
            ax2.plot(folded_time, flux, 'k.', markersize=2, alpha=0.7)
            ax2.set_xlabel('Phase')
            ax2.set_ylabel('Normalized Flux')
            
            plt.tight_layout()
            
            # Save figure to memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=50, bbox_inches='tight')
            buf.seek(0)
            
            # Convert to PIL Image and resize
            image = Image.open(buf).convert('RGB')
            image = image.resize(self.img_size)
            plt.close(fig)
            
            return image
            
        except Exception as e:
            print(f"Error creating image: {e}")
            # Return blank image if processing fails
            return Image.new('RGB', self.img_size, color='white')
    def create_raw_fft_images(self, fits_path):
        try:
            obj = lk.read(fits_path)
            # If it's a TargetPixelFile, convert to lightcurve
            if hasattr(obj, "to_lightcurve"):
                lc = obj.to_lightcurve()
            else:
                lc = obj  # Assume it's already a LightCurve
            # plt.figure(figsize=(10, 4))
            # num_points = len(lc.time)
            # print(f"Number of data points in light curve: {num_points}")    
            # Prepare FFT of flux data
            flux = lc.flux.value
            flux = flux[~np.isnan(flux)]  # Remove NaNs
            flux_norm = (flux - np.mean(flux)) / np.std(flux)

            fft_vals = np.fft.fft(flux_norm)
            fft_freq = np.fft.fftfreq(len(flux_norm), d=np.median(np.diff(lc.time.value)))

            pos_mask = fft_freq > 0
            freqs = fft_freq[pos_mask]
            amplitudes = np.abs(fft_vals[pos_mask])

            # Plot side by side
            fig, axs = plt.subplots(1, 2, figsize=(14, 8))

            # Light curve plot
            lc.plot(ax=axs[0])
            axs[0].set_title(f"Light Curve: {fits_path}")

            # FFT plot
            axs[1].plot(freqs, amplitudes)
            axs[1].set_xlabel('Frequency')
            axs[1].set_ylabel('Amplitude')
            axs[1].set_title('FFT Spectrum')

            # Remove axes and whitespace
            # for ax in axs:
            #     ax.set_frame_on(False)
            # plt.axis('off')
            plt.tight_layout(pad=0)
            # plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            # plt.show()
            # Save figure to memory buffer as HD image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0)  # Use dpi=300 for HD
            buf.seek(0)
            
            # Convert to PIL Image and resize
            image = Image.open(buf).convert('RGB')
            image = image.resize(self.img_size)
            plt.close(fig)
            
            return image
        except Exception as e:
            print(f"Could not plot {fits_path}: {e}")