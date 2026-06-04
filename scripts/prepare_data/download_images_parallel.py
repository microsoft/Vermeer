import os
import requests
import gzip
import shutil
import pandas as pd
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Lock
from functools import partial
import argparse

def fix_image_url(url):
    """
    Fix malformed image URLs by removing crop identifiers (cr[hex]) from filenames.
    
    Example:
        Input: https://images.proteinatlas.org/4109/1843_B2_17_cr5af971a263864_red.tif.gz
        Output: https://images.proteinatlas.org/4109/1843_B2_17_red.tif.gz
    """
    if not isinstance(url, str) or not url:
        return url
    
    # Pattern to match: [image_id]_cr[hexadecimal]_[color].tif.gz
    # We want to remove the _cr[hexadecimal] part
    pattern = r'(_cr[a-f0-9]+)(_(?:red|green|blue|yellow)\.tif\.gz)'
    
    # Replace with just the color part (removing the crop identifier)
    fixed_url = re.sub(pattern, r'\2', url)
    
    # Also fix URLs missing underscore before color (e.g., ...cr5af971a263864red.tif.gz)
    pattern2 = r'(_cr[a-f0-9]+)((?:red|green|blue|yellow)\.tif\.gz)'
    fixed_url = re.sub(pattern2, r'_\2', fixed_url)
    
    return fixed_url


def download_single_image(url, output_tiff, output_gz):
    """
    Download and extract a single image.
    Returns: (success, message, output_path)
    """
    try:
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(output_gz), exist_ok=True)
        
        # Download the file
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Save the compressed file temporarily
        with open(output_gz, 'wb') as f:
            f.write(response.content)
        
        # Extract the file
        with gzip.open(output_gz, 'rb') as f_in:
            with open(output_tiff, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Delete the compressed file
        os.remove(output_gz)
        
        return True, f"Successfully downloaded: {os.path.basename(output_tiff)}", output_tiff
    
    except Exception as e:
        return False, f"Error downloading {url}: {str(e)}", None


def process_row(row, base_dir):
    """
    Process a single row from the dataframe - download all channels for one FOV.
    Returns: list of (success, message) tuples
    """
    uniprot = row['uniprot']
    antibody = row['antibody']
    cell_line = row['cell_line']
    fov = f"fov_{row['fov']}"
    
    results = []
    
    # Create directory structure
    image_dir = os.path.join(base_dir, uniprot, antibody, cell_line, fov)
    os.makedirs(image_dir, exist_ok=True)
    
    # Process each channel
    channels = {
        'red': row['red_imageUrl'],
        'green': row['green_imageUrl'],
        'blue': row['blue_imageUrl'],
        'yellow': row['yellow_imageUrl']
    }
    
    for channel, url in channels.items():
        if not isinstance(url, str) or not url:
            # Mark as failure and log the missing URL
            error_msg = f"Missing URL: {uniprot}/{antibody}/{cell_line}/{fov}/{channel}"
            results.append((False, error_msg, None))
            continue
        
        # Fix malformed URLs
        url = fix_image_url(url)
        
        # Extract filename from URL
        filename = os.path.basename(url)
        output_tiff = os.path.join(image_dir, filename.replace('.gz', ''))
        output_gz = os.path.join(image_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(output_tiff):
            results.append((True, f"Already exists: {os.path.basename(output_tiff)}", output_tiff))
            continue
        
        # Download the image
        success, message, path = download_single_image(url, output_tiff, output_gz)
        results.append((success, message, path))
    
    return results


def download_and_extract_images_parallel(df, base_dir="/n/netscratch/chenf2011_lab/sandeep", num_workers=None):  # base_dir: change to your path
    """
    Download and extract TIFF images from all color channels in parallel.
    Organizes files in structure: uniprot/antibody/cell_line/fov/images
    
    Args:
        df: DataFrame with uniprot, antibody, cell_line, fov, and channel URL columns
        base_dir: Base directory where to save the images
        num_workers: Number of parallel workers (default: cpu_count())
    """
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"Using {num_workers} parallel workers")
    
    # Convert dataframe rows to list of dictionaries for parallel processing
    rows = [row for _, row in df.iterrows()]
    
    # Create partial function with base_dir
    process_func = partial(process_row, base_dir=base_dir)
    
    # Process in parallel with progress bar
    total_rows = len(rows)
    successful_downloads = 0
    failed_downloads = 0
    already_exists = 0
    missing_urls = 0
    
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance with progress tracking
        results_iter = pool.imap_unordered(process_func, rows, chunksize=10)
        
        # Wrap with tqdm for progress bar
        with tqdm(total=total_rows, desc="Processing images") as pbar:
            for row_results in results_iter:
                # row_results is a list of (success, message, path) tuples for each channel
                for success, message, path in row_results:
                    if success:
                        if "Already exists" in message:
                            already_exists += 1
                        else:
                            successful_downloads += 1
                    else:
                        if "Missing URL" in message:
                            missing_urls += 1
                            tqdm.write(f"WARNING: {message}")
                        else:
                            failed_downloads += 1
                            tqdm.write(f"ERROR: {message}")
                
                pbar.update(1)
    
    print(f"\nDownload Summary:")
    print(f"  Successful downloads: {successful_downloads}")
    print(f"  Already exist: {already_exists}")
    print(f"  Missing URLs in CSV: {missing_urls}")
    print(f"  Download failures: {failed_downloads}")
    print(f"  Total images processed: {total_rows * 4}")
    print(f"  Total FOVs processed: {total_rows}")


def main():
    parser = argparse.ArgumentParser(description='Download HPA images in parallel')
    parser.add_argument('--csv', type=str, default='hpa_cell-line-download.csv',
                        help='Path to CSV file with download links')
    parser.add_argument('--output-dir', type=str, default='/n/netscratch/chenf2011_lab/sandeep',  # change to your path
                        help='Base directory for downloaded images')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count())')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of rows to process (for testing)')
    
    args = parser.parse_args()
    
    # Load the dataframe with channel URLs
    print(f"Loading CSV from {args.csv}...")
    hpa_links_df = pd.read_csv(args.csv, sep=",")
    
    if args.limit:
        print(f"Limiting to first {args.limit} rows for testing")
        hpa_links_df = hpa_links_df.head(args.limit)
    
    print(f"Found {len(hpa_links_df)} FOVs to download ({len(hpa_links_df) * 4} total images)")
    
    # Download and extract images
    print("Starting parallel downloads...")
    download_and_extract_images_parallel(hpa_links_df, args.output_dir, args.workers)
    print("Download process completed!")


if __name__ == "__main__":
    main()

