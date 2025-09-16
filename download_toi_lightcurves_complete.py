import pandas as pd
import lightkurve as lk
import os
from time import sleep
import logging

# Set up logging to track progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_toi_lightcurves_complete(csv_file_path, download_dir="./toi_complete_data", 
                                     author="SPOC", cadence="long", 
                                     delay=1.5, max_targets=None):
    """
    Download ALL available light curve data for targets in a TOI CSV file from TESS.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the TOI.csv file
    download_dir : str
        Directory where light curves will be downloaded (default: "./toi_complete_data")
    author : str
        Pipeline author ("SPOC" for primary data, "QLP" for alternative pipeline)
    cadence : str
        Cadence type ("long" or "short") - "long" is recommended for transit search
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
        toi_df = pd.read_csv(csv_file_path)
        logger.info(f"Successfully read TOI.csv with {len(toi_df)} entries")
        
        # If max_targets is specified, limit the dataframe
        if max_targets and max_targets < len(toi_df):
            toi_df = toi_df.head(max_targets)
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
    
    # Iterate through each TOI entry
    for index, row in toi_df.iterrows():
        # Construct identifiers - ADJUST THESE BASED ON YOUR CSV COLUMN NAMES!
        toi_id = f"{row['TIC ID']}" if 'TIC ID' in row else f"TOI-{row['toi_num']}"
        tic_id = f"TIC {row['TIC ID']}" if 'TIC ID' in row else None
        
        # Fallback: if standard column names aren't found, try common alternatives
        if not tic_id:
            for col in ['tic_id', 'TIC_ID', 'ticid', 'tid', 'TIC']:
                if col in row:
                    tic_id = f"TIC {row[col]}"
                    break
        
        # If still no TIC ID, use TOI ID as last resort
        if not tic_id:
            tic_id = toi_id
            logger.warning(f"No TIC ID found for {toi_id}, using TOI ID for search")
        
        logger.info(f"Processing {toi_id} ({index + 1}/{len(toi_df)}) - TIC: {tic_id}")
        
        # Create a subdirectory for this target
        target_dir = os.path.join(download_dir, toi_id.replace(' ', '_'))
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            # Search for ALL available light curves - no sector filtering!
            search_result = lk.search_lightcurve(
                target=tic_id, 
                author=author, 
                # cadence=cadence
                # Note: No 'mission' parameter needed for TESS
            )
            
            if len(search_result) == 0:
                logger.warning(f"No data found for {toi_id} (TIC: {tic_id})")
                results["failed"].append((toi_id, "No data found"))
                continue
            
            logger.info(f"Found {len(search_result)} sectors for {toi_id}")
            
            # Count existing files before download
            existing_files = [f for f in os.listdir(target_dir) if f.endswith('.fits')]
            
            if len(existing_files) >= len(search_result):
                logger.info(f"Skipping {toi_id} - already has {len(existing_files)} files")
                results["skipped"].append((toi_id, len(existing_files)))
                continue
            
            # Download ALL light curves for this target
            logger.info(f"Downloading {len(search_result)} sectors for {toi_id}...")
            light_curve_collection = search_result.download_all(download_dir=target_dir)
            
            # Count downloaded files and calculate size
            downloaded_files = [f for f in os.listdir(target_dir) if f.endswith('.fits')]
            new_files = len(downloaded_files) - len(existing_files)
            
            # Calculate approximate data size (typical TESS FITS: 3-8 MB each)
            data_size_mb = new_files * 5  # Rough estimate
            
            logger.info(f"Successfully downloaded {new_files} new files for {toi_id} (~{data_size_mb} MB)")
            results["successful"].append((toi_id, new_files, data_size_mb))
            results["total_files_downloaded"] += new_files
            results["total_data_volume_mb"] += data_size_mb
            
        except Exception as e:
            error_msg = f"Failed to download {toi_id}: {str(e)}"
            logger.error(error_msg)
            results["failed"].append((toi_id, str(e)))
        
        # Add delay to be polite to the servers
        sleep(delay)
    
    # Print comprehensive summary
    logger.info("\n" + "="*60)
    logger.info("TOI DOWNLOAD SUMMARY:")
    logger.info(f"Targets processed: {len(toi_df)}")
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

def check_toi_csv_columns(csv_file_path):
    """
    Helper function to check the column names in your TOI.csv file
    """
    try:
        df = pd.read_csv(csv_file_path)
        print("Columns found in your TOI.csv:")
        print("=" * 30)
        for col in df.columns:
            print(f"  - '{col}'")
        print("\nSample data:")
        print("=" * 30)
        print(df.head(3))
        return df.columns.tolist()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

# Example usage
if __name__ == "__main__":
    
    print("TESS TOI Light Curve Download Script")
    print("=" * 45)
    
    # First, check your CSV column names
    print("Checking TOI.csv structure...")
    columns = check_toi_csv_columns("TOIs.csv")
    
    if columns:
        print(f"\nCommon TOI column patterns found: {[col for col in columns if 'TOI' in col or 'TIC' in col]}")
    
    print("\n" + "="*45)
    print("RECOMMENDATION: Run test first with 2-3 targets!")
    print("="*45)
    
    # TEST FIRST with just 2-3 targets!
    test_results = download_toi_lightcurves_complete(
        csv_file_path="TOIs.csv",
        download_dir="./toi_test_data",
        author="SPOC",  # Primary TESS pipeline
        cadence="long",
        delay=2.0,  # Longer delay for safety
        max_targets=100  # ONLY download first 3 targets for testing
    )
    
    print("\nTest completed. Check the log above.")
    print("If successful, run without 'max_targets' parameter for full download.")
    
    # UNCOMMENT FOR FULL DOWNLOAD (after testing):
    # full_results = download_toi_lightcurves_complete(
    #     csv_file_path="TOI.csv",
    #     download_dir="./toi_full_data",
    #     author="SPOC",
    #     cadence="long",
    #     delay=1.5
    # )