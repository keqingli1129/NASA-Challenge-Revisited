from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.table import Table
import pandas as pd
from collections import Counter
import numpy as np
import re
import csv
import os
import shutil
import matplotlib.pyplot as plt
from io import StringIO
from scipy.signal import savgol_filter
import lightkurve as lk
import json

# Function to convert string to integer for sorting
def convert_to_int(x):
    try:
        return (0, int(x))  # Tuple with 0 for numbers
    except ValueError:
        return (1, x)       # Tuple with 1 for strings
    
def mapping_hipparcos_catalog_nasaconfirmed():
    # Load Hipparcos catalog data (not used directly here but can be)
    hip_data = Table.read('hipparcos_catalog.fits')

    # Load exoplanet host data with coordinates and solution type
    exo_hosts = Table.read('exo_hosts.csv', format='csv', comment='#')

    # Extract hip_name and soltype columns
    hip_names = exo_hosts['hip_name']
    soltypes = exo_hosts['soltype']

    # Create mask for non-null, non-empty hip_name
    mask = (exo_hosts['hip_name'] != '') & (~exo_hosts['hip_name'].mask.astype(bool))
    valid_hip_hosts = exo_hosts[mask]

    # Dictionary to hold HIP IDs and corresponding solution types (may have multiple entries)
    hip_soltype_map = {}

    # Regex pattern to extract numeric part after 'HIP' prefix
    pattern = re.compile(r'HIP\s*(\d+)')

    for row in valid_hip_hosts:
        name = row['hip_name']
        soltype = row['soltype']
        if name and not isinstance(name, np.ma.core.MaskedConstant):
            match = pattern.search(name)
            if match:
                clean_name = match.group(1)  # Numeric part as string
                if clean_name in hip_soltype_map:
                    hip_soltype_map[clean_name].add(soltype)
                else:
                    hip_soltype_map[clean_name] = {soltype}

    # Sort the HIP IDs
    sorted_hips = sorted(hip_soltype_map.keys(), key=convert_to_int)

    # Print results
    print(f"Total number of distinct HIP IDs: {len(sorted_hips)}\n")
    print("Sorted HIP IDs (without 'HIP' prefix) and their solution types:")
    for hip_id in sorted_hips:
        soltype_list = ', '.join(sorted(hip_soltype_map[hip_id]))
        print(f"{hip_id}: {soltype_list}")

    # Save to CSV
    with open('cps.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['HIP_ID', 'Solution_Types'])
        for hip_id in sorted_hips:
            soltype_list = ', '.join(sorted(hip_soltype_map[hip_id]))
            writer.writerow([hip_id, 'PC'])

    print(f"\nData saved to {'cps.csv'}")

# Example call:
# mapping_hipparcos_catalog_nasaconfirmed()

# Example usage:
# combined_catalog = combine_toi_koi('TOIs.csv', 'KOIs.csv', 'combined_catalog.csv')
def combine_toi_koi(toi_csv, koi_csv, output_csv, match_radius_arcsec=2.0):
    # Load TOI and KOI catalogs
    toi = pd.read_csv(toi_csv, comment='#')
    koi = pd.read_csv(koi_csv, comment='#')
    # # RA is in hours:minutes:seconds, so specify unit='hourangle'
    ra_angles = Angle(toi['RA'], unit='hourangle')
    # Dec is in degrees:arcmin:arcsec, so specify unit='deg'
    dec_angles = Angle(toi['Dec'], unit='deg')
    # Create SkyCoord object
    toi_coords = SkyCoord(ra=ra_angles, dec=dec_angles, frame='icrs')
    # # RA is in hours:minutes:seconds, so specify unit='hourangle'
    ra_angles = Angle(koi['ra'], unit='hourangle')
    # Dec is in degrees:arcmin:arcsec, so specify unit='deg'
    dec_angles = Angle(koi['dec'], unit='deg')
    # Create SkyCoord object
    koi_coords = SkyCoord(ra=ra_angles, dec=dec_angles, frame='icrs')
    
    # Match KOI coords to TOI coords
    idx, d2d, _ = koi_coords.match_to_catalog_sky(toi_coords)
    
    # Identify KOI entries that are NOT duplicates (distance > match_radius)
    unique_mask = d2d > match_radius_arcsec * u.arcsec
    
    # KOI entries unique to KOI catalog (not matched in TOI)
    koi_unique = koi[unique_mask]
    
    # Combine TOI catalog plus unique KOIs
    combined = pd.concat([toi, koi_unique], ignore_index=True)
    
    # Write to output CSV
    combined.to_csv(output_csv, index=False)
    print(f"Combined catalog saved to {output_csv}")
    
    return combined

def map_combined_hipparcos(match_radius_arcsec = 5):
    # Load catalogs
    combined = Table.read('combined_catalog.csv', format='csv', comment='#', encoding='utf-8')
    hip = Table.read('hipparcos_catalog.fits')
    print(hip.colnames)
    # # Clean combined catalog data
    # mask = ~(combined['RA'].mask | combined['Dec'].mask)
    # combined_clean = combined[mask]
    
    # # Clean Hipparcos data - check for null/invalid values instead of masks
    # hip_mask = (hip['RAhms'] != '') & (hip['DEdms'] != '')
    # hip_clean = hip[hip_mask]
    
    # Combined catalog coords - RA and Dec in degrees as floats
    combined_coords = SkyCoord(ra=combined['RA']*u.degree, dec=combined['Dec']*u.degree, frame='icrs')

    # Hipparcos coords - parse sexagesimal strings for RA and Dec
    hip_coords = SkyCoord(ra=hip['RAhms'], dec=hip['DEdms'], unit=(u.hourangle, u.deg), frame='icrs')
    
    # hip_coords_epoch = hip_coords.apply_space_motion(new_obstime='J2015.5')
    # Cross-match catalogs
    idx, d2d, _ = hip_coords.match_to_catalog_sky(combined_coords)
    
    # Filter matches
    match_mask = d2d < match_radius_arcsec * u.arcsec
    
    # Get matched entries
    matched_hip = hip[match_mask]
    matched_combined = combined[idx[match_mask]]
    print(f"Found {len(matched_hip)} matches within {match_radius_arcsec} arcseconds")
    print("\nMatched Hipparcos stars:")
    # Prepare a list of selected fields for output
    output_rows = []

    # for hip_star, comb_star in zip(matched_hip, matched_combined):
    #     output_rows.append({
    #         'HIP': hip_star['HIP'],
    #         'RAhms': hip_star['RAhms'],
    #         'DEdms': hip_star['DEdms'],
    #         'TFOPWG Disposition': comb_star['TFOPWG Disposition']
    #     })
    for hip_star, comb_star in zip(matched_hip, matched_combined):
        output_rows.append({
            'HIP': hip_star['HIP'],
            'TFOPWG Disposition': comb_star['TFOPWG Disposition']
        })

    # Convert to pandas DataFrame
    output_df = pd.DataFrame(output_rows)

    # Save to CSV
    output_df.to_csv('fps.csv', index=False)
     # Output matched Hipparcos entries to a CSV file
    # selected = matched_hip[['HIP', 'RAhms', 'DEdms', 'TFOPWG Disposition']]
    # matched_hip.write('matched_hipparcos.csv', format='csv', overwrite=True)
    print("Matched Hipparcos entries saved to matched_hipparcos.csv")
    return matched_hip

def generate_uid_filenames_with_pandas(filepath):
    """
    Reads a comma-separated file with a header using pandas,
    uses only the first field (ID), and generates filenames like UID_0004024_PLC_001.tbl.
    Then searches ../NASA data and its subfolders for each file, and moves found files to ./data.
    """
    df = pd.read_csv(filepath)
    id_col = df.columns[0]  # Get the name of the first column
    filenames = [f"UID_{str(id).zfill(7)}_PLC_001.tbl" for id in df[id_col] if str(id).isdigit()]

    # Ensure ./data directory exists
    os.makedirs('./data', exist_ok=True)

    # Search and move files
    nasa_data_root = os.path.abspath(os.path.join('..', 'NASA data'))
    for fname in filenames:
        found = False
        for root, dirs, files in os.walk(nasa_data_root):
            if fname in files:
                src = os.path.join(root, fname)
                dst = os.path.join('./data', fname)
                shutil.move(src, dst)
                print(f"Moved: {src} -> {dst}")
                found = True
                break
        if not found:
            print(f"File not found: {fname}")

    return filenames

# Example usage:
# filenames = generate_uid_filenames('hip_ids.txt')
# for fname in filenames:
#     print(fname)
def generate_uid_filenames(filepath):
    """
    Reads a comma-separated file with a header using pandas,
    uses only the first field (ID), and generates filenames like UID_0004024_PLC_001.tbl.
    """
    df = pd.read_csv(filepath)
    id_col = df.columns[0]  # Get the name of the first column
    filenames = [f"UID_{str(id).zfill(7)}_PLC_001.tbl" for id in df[id_col] if str(id).isdigit()]
    return filenames

# Example usage:
# filenames = generate_uid_filenames_with_pandas('hip_ids.txt')
# for fname in filenames:
#     print(fname)

def combine_csv_files(file1, file2, output_file):
    """
    Combine two CPS CSV files, including only distinct HIP_IDs.
    Assumes first column is HIP_ID (may have trailing letters, which are removed).
    Writes combined distinct HIP_IDs and Solution_Types to output_file.
    """
    def clean_id(hip_id):
        # Remove trailing letters, keep only numeric part
        return ''.join(char for char in hip_id if char.isdigit())

    ids = set()
    rows = []

    # Helper to process a file
    def process_file(filename):
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                hip_id = row[0].strip()
                solution_type = row[1].strip() if len(row) > 1 else ''
                clean_hip = clean_id(hip_id)
                if clean_hip and clean_hip not in ids:
                    ids.add(clean_hip)
                    rows.append([clean_hip, solution_type])

    process_file(file1)
    process_file(file2)

    # Write output
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['HIP_ID', 'Solution_Types'])
        writer.writerows(rows)

# Example usage:
# combine_cps_files('cps.csv', 'cps2.csv', 'combined_cps.csv')
def plot_tbl_mag_vs_bjd(tbl_filepath):
    """
    Reads a .tbl file using Astropy Table, converts to pandas DataFrame,
    and plots Magnitude vs. BJD.
    """
    # Read the file, skipping header lines that start with backslashes or '#' and empty lines
    with open(tbl_filepath, 'r') as f:
        lines = f.readlines()

    # Find the line index where actual data starts (the line starting with the first BJD number)
    data_start_idx = 0
    for i, line in enumerate(lines):
        line_strip = line.strip()
        # BJD lines start with a number like 244...
        if line_strip and (line_strip[0].isdigit() or line_strip[0] == '.'):
            data_start_idx = i
            break

    # Extract data lines only
    data_lines = lines[data_start_idx:]
    from io import StringIO
    data_str = ''.join(data_lines)

    # Read data into dataframe
    df = pd.read_csv(StringIO(data_str), delim_whitespace=True, header=None,
                    names=['BJD', 'Magnitude', 'Magnitude_Uncertainty', 'Data_Quality_Flag', 'Accepted'])

    # print(df.head())  # should show numeric data only now

    # Filter out non-accepted data points (Accepted column == 0)
    df = df[df['Accepted'] == 1]
    # Normalize Magnitude before plotting
    mag_norm = (df['Magnitude'] - df['Magnitude'].mean()) / df['Magnitude'].std()
    # Apply Savitzky-Golay filter to estimate trend
    trend = savgol_filter(mag_norm, window_length=51, polyorder=3)

    # Detrended magnitude (original minus trend)
    detrended_mag = mag_norm - trend
    # Plot light curve
    # plt.scatter(df['BJD'], mag_norm, s=10, alpha=0.7)
    # plt.gca().invert_yaxis()  # Magnitude scale is inverse (brighter=lower)
    # plt.xlabel('BJD (days)')
    # plt.ylabel('Magnitude (Hp)')
    # plt.title('Hipparcos Light Curve - UID_0000065_PLC_001')
    # plt.show()
    # Plot original and detrended light curves
    plt.figure(figsize=(10, 6))

    plt.subplot(2,1,1)
    plt.scatter(df['BJD'], mag_norm, s=10, alpha=0.7, label='Original')
    plt.plot(df['BJD'], trend, color='red', label='Trend (S-G filter)')
    plt.gca().invert_yaxis()
    plt.title('Original Hipparcos Light Curve with Trend')
    plt.xlabel('Julian Date')
    plt.ylabel('Magnitude')
    plt.legend()

    plt.subplot(2,1,2)
    plt.scatter(df['BJD'], detrended_mag, s=10, alpha=0.7, color='green')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Detrended Light Curve (Residuals)')
    plt.xlabel('Julian Date')
    plt.ylabel('Magnitude (detrended)')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()
# Example usage:
# plot_tbl_mag_vs_bjd('UID_0004024_PLC_001.tbl')
def generate_label_dict_from_toi_csv(filepath):
    """
    Reads final.csv, uses the first field (HIP_ID) to generate filenames,
    and the second field (Solution_Types) to create a dictionary:
    - key: filename like UID_0004024_PLC_001.tbl
    - value: 1 if Solution_Types is 'PC' or 'CP', else 0
    """
    df = pd.read_csv(filepath)
    id_col = df.columns[0]
    sol_col = df.columns[21]
    uid_dict = {}
    for idx, row in df.iterrows():
        id = f"{str(row[id_col])}"
        value = 1 if str(row[sol_col]).strip() in ['PC', 'CP'] else 0
        # print(f"id: {id}, value: {row[sol_col]}")
        uid_dict[id] = value
    return uid_dict
def generate_label_dict_from_koi_csv(filepath):
    """
    Reads final.csv, uses the first field (HIP_ID) to generate filenames,
    and the second field (Solution_Types) to create a dictionary:
    - key: filename like UID_0004024_PLC_001.tbl
    - value: 1 if Solution_Types is 'PC' or 'CP', else 0
    """
    df = pd.read_csv(filepath, comment='#')
    kepler_name = df.columns[3]
    kepoi_name = df.columns[2]
    sol_col = df.columns[4]
    uid_dict = {}
    for idx, row in df.iterrows():
        # Method 1: Using conditional expression (one-liner)
        id = f"{row['kepler_name']}".replace(" ", "_") if 'kepler_name' in row and not pd.isna(row['kepler_name']) else f"{row['kepoi_name']}".replace(" ", "_")
        value = 1 if str(row[sol_col]).strip() in ['CONFIRMED'] else 0
        # print(f"id: {id}, value: {row[sol_col]}")
        uid_dict[id] = value
    return uid_dict
# Example usage:
# uid_dict = generate_uid_dict_from_final_csv('final.csv')
# print(uid_dict)
def copy_fits_files_from_toi_test_to_data(toi_test_data_folder, toi_data_folder):
    """
    Recursively searches for folders named as IDs under toi_test_data_folder,
    finds all .fits files inside each ID folder and its subfolders,
    and copies them to toi_data_folder/id/
    """
    for item in os.listdir(toi_test_data_folder):
        id_path = os.path.join(toi_test_data_folder, item)
        if os.path.isdir(id_path) and item.isdigit():
            dest_dir = os.path.join(toi_data_folder, item)
            os.makedirs(dest_dir, exist_ok=True)
            # Recursively walk through all subfolders
            for root, _, files in os.walk(id_path):
                for fname in files:
                    if fname.lower().endswith('.fits'):
                        src_file = os.path.join(root, fname)
                        dest_file = os.path.join(dest_dir, fname)
                        shutil.copy2(src_file, dest_file)
                        print(f"Copied {src_file} to {dest_file}")

# Example usage:
# copy_fits_files_from_test_to_data('toi_test_data', 'toi_data')
def copy_fits_files_from_koi_test_to_data(koi_test_data_folder, koi_data_folder):
    """
    Recursively searches for folders named like KOI-K00113.01 or KOI-Kepler-1_b under koi_test_data_folder,
    finds all .fits files inside each folder and its subfolders,
    and copies them to koi_data/K00113.01 or koi_data/Kepler-1_b/
    """
    for item in os.listdir(koi_test_data_folder):
        src_id_path = os.path.join(koi_test_data_folder, item)
        if os.path.isdir(src_id_path) and (item.startswith("KOI-K")):
            # Remove "KOI-" prefix for destination folder name
            dest_id = item.replace("KOI-", "")
            dest_dir = os.path.join(koi_data_folder, dest_id)
            os.makedirs(dest_dir, exist_ok=True)
            # Recursively walk through all subfolders
            for root, _, files in os.walk(src_id_path):
                for fname in files:
                    if fname.lower().endswith('.fits'):
                        src_file = os.path.join(root, fname)
                        dest_file = os.path.join(dest_dir, fname)
                        shutil.copy2(src_file, dest_file)
                        print(f"Copied {src_file} to {dest_file}")

# Example usage:
# copy_fits_files_from_koi_test_to_data('koi_test_data', 'koi_data')
def plot_first_id_fits_in_toi_data(toi_data_folder):
    """
    Counts ID folders under toi_data_folder, and only plots .fits files from the first ID folder found.
    """
    id_folders = [item for item in os.listdir(toi_data_folder)
                  if os.path.isdir(os.path.join(toi_data_folder, item))]
    print(f"Found {len(id_folders)} ID folders in {toi_data_folder}.")
    if not id_folders:
        print("No ID folders found.")
        return

    # Only plot for the first ID folder
    id_folder = id_folders[0]
    id_folder = 'Kepler-106_b'
    id_path = os.path.join(toi_data_folder, id_folder)
    print(f"Plotting .fits files in: {id_folder}")
    for fname in os.listdir(id_path):
        if fname.lower().endswith('.fits'):
            fits_path = os.path.join(id_path, fname)
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
                fig, axs = plt.subplots(1, 2, figsize=(14, 4))

                # Light curve plot
                lc.plot(ax=axs[0])
                axs[0].set_title(f"Light Curve: {fname} ({id_folder})")

                # FFT plot
                axs[1].plot(freqs, amplitudes)
                axs[1].set_xlabel('Frequency')
                axs[1].set_ylabel('Amplitude')
                axs[1].set_title('FFT Spectrum')

                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Could not plot {fits_path}: {e}")

# Example usage:
# plot_first_id_fits_in_toi_data('toi_data')
def load_json_to_dict(filepath):
    """
    Loads a JSON file and returns its contents as a Python dictionary.
    
    Args:
        filepath (str): Path to the JSON file.
        
    Returns:
        dict: Dictionary containing the JSON data.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Example usage:
# my_dict = load_json_to_dict('merged_label_dict.json')
def main():
    # mapping_hipparcos_catalog_nasaconfirmed()
    # combined_catalog = combine_toi_koi('TOIs.csv', 'KOIs.csv', 'combined_catalog.csv')
    # map_combined_hipparcos()
    # fnames = generate_uid_filenames_with_pandas('final.csv')
    # print(fnames)
    # map_combined_hipparcos()
    # combine_csv_files('cps.csv', 'fps.csv', 'final.csv')
    # plot_tbl_mag_vs_bjd('./data/UID_0001419_PLC_001.tbl')
    label_dict_koi = generate_label_dict_from_koi_csv('KOIs.csv')
    # print(label_dict_koi)
    label_dict_toi = generate_label_dict_from_toi_csv('TOIs.csv')
    # print(label_dict_toi)
     # Merge the two dictionaries, preferring TOI values if keys overlap
    merged_label_dict = {**label_dict_koi, **label_dict_toi}
    print("Merged label dict:")
    # print(merged_label_dict)
    # Output merged_label_dict to a JSON file
    with open('merged_label_dict.json', 'w') as f:
        json.dump(merged_label_dict, f, indent=2)
    print("Merged label dict saved to merged_label_dict.json")
    # plot_tbl_mag_vs_bjd('./data/UID_0001931_PLC_001.tbl')
    # uid_dict = generate_uid_dict_from_final_csv('final.csv')
    # print(uid_dict)
    # copy_fits_files_from_toi_test_to_data('toi_test_data', 'toi_data')
    copy_fits_files_from_koi_test_to_data('koi_full_data', 'koi_data')
    # plot_first_id_fits_in_toi_data('koi_data')
    # my_dict = load_json_to_dict('merged_label_dict.json')
    # print('done')
if __name__ == "__main__":
    main()