import pandas as pd
import lightkurve as lk
import os
from time import sleep
import logging

# Set up logging to track progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_koi_lightcurves_complete(csv_file_path, download_dir="./koi_complete_data", 
                                    author="Kepler", cadence="long", mission="Kepler", 
                                    delay=1.5, max_targets=None):
    """
    Download ALL available light curve data for targets in a KOI CSV file from Kepler/K2.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the KOI.csv file
    download_dir : str
        Directory where light curves will be downloaded (default: "./koi_complete_data")
    author : str
        Pipeline author ("Kepler" for original mission, "K2" for K2 mission)
    cadence : str
        Cadence type ("long" or "short") - "long" is recommended for transit search
    mission : str
        Mission name ("Kepler" or "K2")
    delay : float
        Delay between downloads in seconds (recommended: 1.5+ to avoid server overload)
    max_targets : int
        If set, only process the first N targets (useful for testing)
    
    Returns:
    --------
    dict: A detailed summary of downloads
    """
    
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Read the CSV file
    try:
        koi_df = pd.read_csv(csv_file_path, comment='#')
        logger.info(f"Successfully read KOI.csv with {len(koi_df)} entries")
        
        # If max_targets is specified, limit the dataframe
        if max_targets and max_targets < len(koi_df):
            koi_df = koi_df.head(max_targets)
            logger.info(f"Limiting to first {max_targets} targets for testing")
            
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return {"error": f"CSV read failed: {e}"}
    
    # Initialize results tracking
    results = {
        "successful": [],
        "failed": [],
        "skipped": [],
        "total_files_downloaded": 0,
        "total_data_volume_mb": 0
    }
    
    # Iterate through each KOI entry
    for index, row in koi_df.iterrows():
        # Construct KOI identifier with multiple fallback options
        koi_id = None

        # Try different possible column names for KOI identifier
        if 'kepler_name' in row and not pd.isna(row['kepler_name']):
            koi_id = f"KOI-{row['kepler_name']}"
        elif 'kepoi_name' in row and not pd.isna(row['kepoi_name']):
            koi_id = f"KOI-{row['kepoi_name']}"
        elif 'koi_name' in row and not pd.isna(row['koi_name']):
            koi_id = f"KOI-{row['koi_name']}"
        elif 'KOI' in row and not pd.isna(row['KOI']):
            koi_id = f"KOI-{row['KOI']}"
        else:
            # If no KOI identifier found, we'll need to handle this case
            logger.warning(f"No KOI identifier found for row {index}")
            # You might want to skip this target or use another identifier
            continue
        kic_id = f"KIC {row['kepid']}" if 'kepid' in row else None
        
        # Fallback: if standard column names aren't found, try common alternatives
        if not kic_id:
            for col in ['kic_id', 'KIC_ID', 'kicid', 'kepler_id']:
                if col in row:
                    kic_id = f"KIC {row[col]}"
                    break
        
        # If still no KIC ID, use KOI ID as last resort
        if not kic_id:
            kic_id = koi_id
            logger.warning(f"No KIC ID found for {koi_id}, using KOI ID for search")
        
        logger.info(f"Processing {koi_id} ({index + 1}/{len(koi_df)}) - KIC: {kic_id}")
        
        # Create a subdirectory for this target
        target_dir = os.path.join(download_dir, koi_id.replace(' ', '_'))
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            # Search for ALL available light curves - no quarter filtering!
            search_result = lk.search_lightcurve(
                target=kic_id, 
                author=author, 
                cadence=cadence,
                mission=mission
            )
            
            if len(search_result) == 0:
                logger.warning(f"No data found for {koi_id} (KIC: {kic_id})")
                results["failed"].append((koi_id, "No data found"))
                continue
            
            logger.info(f"Found {len(search_result)} observations for {koi_id}")
            
            # Count existing files before download
            existing_files = [f for f in os.listdir(target_dir) if f.endswith('.fits')]
            
            if len(existing_files) >= len(search_result):
                logger.info(f"Skipping {koi_id} - already has {len(existing_files)} files")
                results["skipped"].append((koi_id, len(existing_files)))
                continue
            
            # Download ALL light curves for this target
            logger.info(f"Downloading {len(search_result)} files for {koi_id}...")
            light_curve_collection = search_result.download_all(download_dir=target_dir)
            
            # Count downloaded files and calculate size
            downloaded_files = [f for f in os.listdir(target_dir) if f.endswith('.fits')]
            new_files = len(downloaded_files) - len(existing_files)
            
            # Calculate approximate data size (typical Kepler FITS: 5-15 MB each)
            data_size_mb = new_files * 10  # Rough estimate
            
            logger.info(f"Successfully downloaded {new_files} new files for {koi_id} (~{data_size_mb} MB)")
            results["successful"].append((koi_id, new_files, data_size_mb))
            results["total_files_downloaded"] += new_files
            results["total_data_volume_mb"] += data_size_mb
            
        except Exception as e:
            error_msg = f"Failed to download {koi_id}: {str(e)}"
            logger.error(error_msg)
            results["failed"].append((koi_id, str(e)))
        
        # Add delay to be polite to the servers
        sleep(delay)
    
    # Print comprehensive summary
    logger.info("\n" + "="*60)
    logger.info("KOI DOWNLOAD SUMMARY:")
    logger.info(f"Targets processed: {len(koi_df)}")
    logger.info(f"Successfully downloaded: {len(results['successful'])} targets")
    logger.info(f"Total files downloaded: {results['total_files_downloaded']}")
    logger.info(f"Total data volume: ~{results['total_data_volume_mb']} MB")
    logger.info(f"Failed: {len(results['failed'])} targets")
    logger.info(f"Skipped (already complete): {len(results['skipped'])} targets")
    logger.info("="*60)
    
    # Print list of failed targets for debugging
    if results['failed']:
        logger.info("Failed targets:")
        for target, error in results['failed']:
            logger.info(f"  - {target}: {error}")
    
    return results

# Example usage
if __name__ == "__main__":
    
    print("KOI Light Curve Download Script")
    print("=" * 40)
    
    # TEST FIRST with just 2-3 targets!
    test_results = download_koi_lightcurves_complete(
        csv_file_path="KOIs.csv",
        download_dir="./koi_test_data",
        author="Kepler",
        cadence="long",
        mission="Kepler",
        delay=2.0,  # Longer delay for safety
        max_targets=100  # ONLY download first 3 targets for testing
    )
    
    print("\nTest completed. Check the log above.")
    print("If successful, run without 'max_targets' parameter for full download.")
    
    # UNCOMMENT FOR FULL DOWNLOAD (after testing):
    # full_results = download_koi_lightcurves_complete(
    #     csv_file_path="KOI.csv",
    #     download_dir="./koi_full_data",
    #     author="Kepler",
    #     cadence="long",
    #     mission="Kepler",
    #     delay=1.5
    # )