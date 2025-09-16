import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

# Search for the light curve using just the TIC number
search_result = lk.search_lightcurve("50365310", author="SPOC")

print(f"Found {len(search_result)} observations")

if len(search_result) > 0:
    # Display what we found
    print(search_result)
    
    # Download the first light curve in the search results
    lc = search_result[0].download()
    
    print(f"Successfully downloaded light curve!")
    print(f"Mission: {lc.mission}")
    print(f"Sector: {lc.sector}")
    print(f"Target: {lc.targetid}")
    print(f"Time range: {lc.time[0]:.2f} to {lc.time[-1]:.2f} BTJD")
    print(f"Number of points: {len(lc)}")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'TESS Light Curve for TIC 50365310 (Sector {lc.sector})', fontsize=16)
    
    # Plot 1: Raw light curve
    axes[0, 0].scatter(lc.time.value, lc.flux.value, s=1, alpha=0.7, c='blue')
    axes[0, 0].set_xlabel('Time (BTJD)')
    axes[0, 0].set_ylabel('Raw Flux')
    axes[0, 0].set_title('Raw Light Curve')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Normalized light curve
    normalized_flux = lc.flux / np.median(lc.flux)
    axes[0, 1].scatter(lc.time.value, normalized_flux, s=1, alpha=0.7, c='green')
    axes[0, 1].set_xlabel('Time (BTJD)')
    axes[0, 1].set_ylabel('Normalized Flux')
    axes[0, 1].set_title('Normalized Light Curve')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Remove trends and flatten
    try:
        flat_lc = lc.flatten(window_length=101)
        axes[1, 0].scatter(flat_lc.time.value, flat_lc.flux, s=1, alpha=0.7, c='red')
        axes[1, 0].set_xlabel('Time (BTJD)')
        axes[1, 0].set_ylabel('Detrended Flux')
        axes[1, 0].set_title('Detrended Light Curve')
        axes[1, 0].grid(True, alpha=0.3)
    except Exception as e:
        axes[1, 0].text(0.5, 0.5, f'Flattening failed: {e}', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Flattening Failed')
    
    # Plot 4: Fold the light curve if we know the period (for TOI-1000.01)
    # You can get period from your CSV file or ExoFOP
    known_period = 4.98  # Example period for TOI-1000.01 in days - GET THIS FROM YOUR DATA!
    known_t0 = 1350.0    # Example transit time - GET THIS FROM YOUR DATA!
    
    if known_period and known_t0:
        try:
            folded_lc = flat_lc.fold(period=known_period, epoch_time=known_t0)
            axes[1, 1].scatter(folded_lc.phase, folded_lc.flux, s=2, alpha=0.7, c='purple')
            axes[1, 1].set_xlabel('Phase')
            axes[1, 1].set_ylabel('Detrended Flux')
            axes[1, 1].set_title(f'Folded Light Curve (Period: {known_period} days)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add vertical lines at expected transit phases
            transit_phases = [-0.25, -0.1, 0, 0.1, 0.25]
            for phase in transit_phases:
                axes[1, 1].axvline(phase, color='gray', linestyle='--', alpha=0.5)
                
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, f'Folding failed: {e}', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Folding Failed')
    else:
        axes[1, 1].text(0.5, 0.5, 'No period information available\nfor folding', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Folding Requires Period')
    
    plt.tight_layout()
    plt.show()
    
    # Also show the simple lightkurve plot
    print("\nSimple lightkurve plot:")
    lc.plot()
    plt.show()
    
else:
    print("No light curves found. Trying alternative search strategies...")
    
    # Try alternative search strategies
    strategies = [
        {"target": "50365310", "author": "QLP"},
        {"target": "TIC 50365310", "author": "SPOC"},
        {"target": "TIC 50365310", "author": "QLP"},
        {"target": "TOI-1000", "author": "SPOC"},
        {"target": "TOI-1000.01", "author": "SPOC"},
    ]
    
    for strategy in strategies:
        try:
            alt_search = lk.search_lightcurve(strategy["target"], author=strategy["author"])
            if len(alt_search) > 0:
                print(f"Found {len(alt_search)} observations with {strategy}")
                break
        except:
            continue