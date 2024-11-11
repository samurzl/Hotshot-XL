import requests
from bs4 import BeautifulSoup
import os
import time
import subprocess
import argparse
import re
import logging
import shutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import signal
import sys
import concurrent.futures
from threading import Lock, Event
import uuid
from tqdm import tqdm
import cloudscraper
from urllib.parse import urljoin
import json
import math

# Initialize global variables and locks
video_counter = 1
sequence_counter = 0
sequence_lock = Lock()
limit_reached = Event()
session = None  # Will be initialized later
args = None     # Command-line arguments
output_dir = None  # Dataset output directory
blacklist_tags = set()
always_tags = set()

def setup_logging():
    """Configure logging for the script with separate handlers for console and file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the root logger level to INFO

    # Formatter for both handlers
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

    # File handler - logs all messages
    file_handler = logging.FileHandler("scraper.log", mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler - logs only ERROR and CRITICAL messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Booru-style Website Video Scraper with Multiple Sequence Extraction")

    # Existing arguments
    parser.add_argument(
        '--base-url',
        type=str,
        required=True,
        help='Base URL of the booru website, e.g., https://yourbooru.com/'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='dataset',
        help='Directory to save the dataset'
    )
    parser.add_argument(
        '--num-sequences',
        type=int,
        default=None,
        help='Total number of sequences to extract across all videos'
    )
    parser.add_argument(
        '--max-sequences-per-video',
        type=int,
        default=1,
        help='Maximum number of sequences to extract per video'
    )
    parser.add_argument(
        '--frames-per-sequence',
        type=int,
        default=10,
        help='Number of frames per sequence'
    )
    parser.add_argument(
        '--extraction-framerate',
        type=int,
        default=1,
        help='Interval between frames in a sequence (frames per second)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Maximum number of concurrent workers'
    )
    parser.add_argument(
        '--max-tags',
        type=int,
        default=None,
        help='Maximum number of tags to include in prompt.txt'
    )
    parser.add_argument(
        '--always-tags',
        type=str,
        default=None,
        help='Comma-separated list of tags to always include in prompt.txt if present'
    )
    parser.add_argument(
        '--blacklist-tags',
        type=str,
        default=None,
        help='Comma-separated list of tags to exclude from prompt.txt'
    )

    # New arguments for resolution and aspect ratio
    parser.add_argument(
        '--resolution',
        type=int,
        required=True,
        help='Desired resolution for the total pixel count (e.g., 1024 for approximately 1024x1024 pixels)'
    )
    parser.add_argument(
        '--aspect-ratio',
        type=float,
        required=True,
        help='Desired aspect ratio for the extracted frames (e.g., 1.75 for width/height ratio)'
    )

    # Login credentials if authentication is needed
    parser.add_argument(
        '--username',
        type=str,
        default=None,
        help='Username for login (if required)'
    )
    parser.add_argument(
        '--password',
        type=str,
        default=None,
        help='Password for login (if required)'
    )
    return parser.parse_args()

def setup_requests_session():
    """Set up a requests session with retries and appropriate headers using cloudscraper."""
    global session
    session = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'mobile': False
        }
    )
    retries = Retry(
        total=5,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Define headers to mimic a real browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/113.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': args.base_url
    })

def signal_handler(sig, frame):
    """Handle termination signals gracefully."""
    logging.critical('Termination signal received. Shutting down...')
    sys.exit(1)

def login():
    """
    Log in to the website if authentication is required.
    """
    if args.username and args.password:
        login_url = urljoin(args.base_url, "index.php?page=account&s=login")
        payload = {
            'username': args.username,
            'password': args.password,
            'commit': 'Log in'
        }
        try:
            response = session.post(login_url, data=payload, timeout=10)
            response.raise_for_status()
            if 'logout' not in response.text.lower():
                logging.critical("Login failed. Check your credentials.")
                sys.exit(1)
            logging.info("Logged in successfully.")
        except requests.RequestException as e:
            logging.critical(f"Failed to log in: {e}")
            sys.exit(1)
    else:
        logging.info("No login credentials provided. Proceeding without login.")

def initialize_counters():
    """
    Initialize video_counter based on existing sequences in the output directory.
    This ensures that new sequences are appended without duplicating existing ones.
    """
    global video_counter
    existing_vids = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and re.match(r'^vid_\d{6}$', d)
    ]
    if existing_vids:
        existing_vid_nums = [int(d.split('_')[1]) for d in existing_vids]
        video_counter = max(existing_vid_nums) + 1
        logging.info(f"Detected existing dataset with {len(existing_vids)} sequences. Starting at vid_{video_counter:06d}.")
    else:
        video_counter = 1
        logging.info("No existing dataset found. Starting with vid_000001.")

def get_video_page_urls():
    """
    Iterate through list pages to collect all video page URLs.

    Returns:
        List of video page URLs.
    """
    video_page_urls = []
    pid = 0
    while not limit_reached.is_set():
        list_page_url = urljoin(args.base_url, f"index.php?page=post&s=list&tags=sort%3Ascore+animated+&pid={pid}")
        try:
            response = session.get(list_page_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.critical(f"Failed to retrieve list page {list_page_url}: {e}")
            break

        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=re.compile(r'index\.php\?page=post&s=view&id=\d+'))
        page_urls = set()
        for link in links:
            href = link.get('href')
            if href and href not in page_urls:
                full_url = urljoin(args.base_url, href)
                video_page_urls.append(full_url)
                page_urls.add(href)

        logging.info(f"Found {len(page_urls)} video links on page PID={pid}.")

        if len(page_urls) == 0:
            logging.info("No more video links found. Ending pagination.")
            break

        pid += 10  # Assuming pid increments by 10 per page
        time.sleep(2)  # 2-second delay between requests

        if args.num_sequences and sequence_counter >= args.num_sequences:
            logging.info(f"Reached the total sequence limit of {args.num_sequences}. Stopping pagination.")
            break

    logging.info(f"Total video pages found: {len(video_page_urls)}")
    return video_page_urls

def parse_video_page(video_page_url):
    """
    Parse a single video page to extract video URL and tags.

    Args:
        video_page_url (str): URL of the video page.

    Returns:
        Tuple (video_url, tags) or (None, None) if failed.
    """
    if limit_reached.is_set():
        return None, None

    try:
        response = session.get(video_page_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Failed to retrieve video page {video_page_url}: {e}")
        return None, None

    soup = BeautifulSoup(response.text, 'html.parser')

    video_tag = soup.find('video')
    if not video_tag:
        logging.warning(f"No <video> tag found in {video_page_url}.")
        return None, None

    source_tag = video_tag.find('source', src=True)
    if not source_tag:
        logging.warning(f"No <source> tag with src found in {video_page_url}.")
        return None, None

    video_url = source_tag['src']
    if not video_url.startswith('http'):
        video_url = urljoin(args.base_url, video_url.lstrip('/'))

    tags = []
    meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
    if meta_keywords and meta_keywords.get('content'):
        tags = [tag.strip() for tag in meta_keywords['content'].split(',')]
    else:
        tag_sidebar = soup.find('ul', id='tag-sidebar')
        if tag_sidebar:
            tag_links = tag_sidebar.find_all('a', href=re.compile(r'index\.php\?page=post&s=list&tags='))
            tags = [tag.get_text(strip=True) for tag in tag_links]
    
    if not tags:
        logging.warning(f"No tags found for video page {video_page_url}.")

    return video_url, tags

def download_video(video_url, video_path):
    """
    Download the video from the given URL to the specified path.

    Args:
        video_url (str): URL of the video to download.
        video_path (str): Local file path to save the video.

    Returns:
        Boolean indicating success or failure.
    """
    if limit_reached.is_set():
        return False

    try:
        with session.get(video_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_length = int(r.headers.get('content-length', 0))
            with open(video_path, 'wb') as f, tqdm(
                desc=os.path.basename(video_path),
                total=total_length,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        return True
    except requests.RequestException as e:
        logging.error(f"Failed to download video {video_url}: {e}")
        return False

def get_video_info(video_path):
    """
    Retrieve video duration, frame rate, and total frames using ffprobe.

    Args:
        video_path (str): Path to the video file.

    Returns:
        Tuple (duration_in_seconds, frame_rate, total_frames) or (None, None, None) if failed.
    """
    try:
        # Get duration and frame rate
        cmd_info = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=duration,r_frame_rate,nb_frames',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd_info, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        info = json.loads(result.stdout)

        if 'streams' not in info or len(info['streams']) == 0:
            logging.error(f"No video streams found in {video_path}.")
            return None, None, None

        stream = info['streams'][0]
        duration = float(stream.get('duration', 0))
        r_frame_rate = stream.get('r_frame_rate', '0/0')
        nb_frames = stream.get('nb_frames', None)

        # Calculate frame rate
        if '/' in r_frame_rate:
            num, denom = map(int, r_frame_rate.split('/'))
            frame_rate = num / denom if denom != 0 else 0
        else:
            frame_rate = float(r_frame_rate) if r_frame_rate else 0

        # Calculate total frames
        if nb_frames is not None and nb_frames.isdigit():
            total_frames = int(nb_frames)
        else:
            total_frames = int(duration * frame_rate) if frame_rate > 0 else 0

        return duration, frame_rate, total_frames
    except subprocess.TimeoutExpired:
        logging.error(f"ffprobe timed out while processing {video_path}.")
        return None, None, None
    except Exception as e:
        logging.error(f"Failed to retrieve video info for {video_path}: {e}")
        return None, None, None

def extract_sequences(video_path, sequences_dir, max_sequences_per_video, frames_per_sequence, extraction_framerate, resolution, aspect_ratio):
    """
    Extract multiple sequences of frames from the video using FFmpeg.

    Each sequence is saved as a separate vid_XXXXXX directory within sequences_dir.

    Args:
        video_path (str): Path to the video file.
        sequences_dir (str): Directory to save the extracted sequences.
        max_sequences_per_video (int): Maximum number of sequences to extract from this video.
        frames_per_sequence (int): Number of frames per sequence.
        extraction_framerate (int): Interval between frames in a sequence (frames per second).
        resolution (int): Desired resolution for the total pixel count (e.g., 1024).
        aspect_ratio (float): Desired aspect ratio for the extracted frames (e.g., 1.75).

    Returns:
        List of sequence directories extracted or empty list if failed.
    """
    try:
        duration, frame_rate, total_frames = get_video_info(video_path)
        if duration is None or frame_rate is None or total_frames is None:
            logging.error(f"Could not retrieve video info for {video_path}.")
            return []

        if frame_rate <= 0 or total_frames <= 0:
            logging.error(f"Invalid frame rate or total frames for {video_path}.")
            return []

        # Calculate desired width and height based on resolution and aspect ratio
        # Total pixels = resolution^2
        # width = aspect_ratio * height
        # width * height = resolution^2 => height = sqrt(resolution^2 / aspect_ratio)
        height = int(round(math.sqrt(resolution**2 / aspect_ratio)))
        width = int(round(aspect_ratio * height))

        logging.info(f"Calculated frame size: {width}x{height} pixels based on resolution={resolution} and aspect_ratio={aspect_ratio}.")

        # Calculate frame intervals for sequences
        interval_between_sequences = total_frames / (max_sequences_per_video + 1)
        sequence_start_frames = [int(interval_between_sequences * (i + 1)) for i in range(max_sequences_per_video)]

        extracted_sequences = []

        for start_frame in sequence_start_frames:
            # Calculate the frame numbers to extract
            frame_numbers = [start_frame + extraction_framerate * j for j in range(frames_per_sequence)]
            # Ensure frame numbers do not exceed total_frames
            frame_numbers = [fn for fn in frame_numbers if fn < total_frames]

            if not frame_numbers:
                logging.warning(f"No valid frames to extract for sequence starting at frame {start_frame} in {video_path}.")
                continue

            # Convert frame numbers to timestamps
            frame_timestamps = [fn / frame_rate for fn in frame_numbers]

            # Acquire a unique video number for each sequence
            with sequence_lock:
                global video_counter, sequence_counter
                if args.num_sequences and sequence_counter >= args.num_sequences:
                    limit_reached.set()
                    return extracted_sequences  # Reached total sequences limit
                vid_num = video_counter
                video_counter += 1
                sequence_counter += 1

            # Create sequence directory
            seq_dir = os.path.join(sequences_dir, f'vid_{vid_num:06d}')
            os.makedirs(seq_dir, exist_ok=True)

            for i, ts in enumerate(frame_timestamps):
                output_frame = os.path.join(seq_dir, f'frame_{i:05d}.png')
                # FFmpeg command with scaling to desired width and height
                cmd = [
                    'ffmpeg',
                    '-ss', str(ts),
                    '-i', video_path,
                    '-vframes', '1',
                    '-vf', f"scale={width}:{height}",
                    output_frame,
                    '-y'  # Overwrite without asking
                ]
                try:
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)
                except subprocess.TimeoutExpired:
                    logging.error(f"FFmpeg timed out while extracting frame at {ts} from {video_path}.")
                    shutil.rmtree(seq_dir)  # Remove incomplete sequence directory
                    with sequence_lock:
                        sequence_counter -= 1
                    break  # Stop extracting this sequence
                except subprocess.CalledProcessError as e:
                    stderr_output = e.stderr.decode().strip()
                    logging.error(f"FFmpeg failed to extract frame at {ts} from {video_path}: {stderr_output}")
                    shutil.rmtree(seq_dir)  # Remove incomplete sequence directory
                    with sequence_lock:
                        sequence_counter -= 1
                    break  # Stop extracting this sequence

            if os.path.exists(seq_dir):
                logging.info(f"Extracted sequence vid_{vid_num:06d} with frames from {video_path}.")
                extracted_sequences.append(seq_dir)

            # Check if total sequences limit is reached
            with sequence_lock:
                if args.num_sequences and sequence_counter >= args.num_sequences:
                    limit_reached.set()
                    break

        return extracted_sequences
    except subprocess.TimeoutExpired:
        logging.error(f"FFmpeg timed out while processing {video_path}.")
        return []
    except Exception as e:
        logging.error(f"Failed to extract sequences from {video_path}: {e}")
        return []

def save_prompt(tags, prompt_path):
    """
    Save the tags to prompt.txt.

    Args:
        tags (list): List of tags.
        prompt_path (str): Path to save prompt.txt.
    """
    try:
        # Exclude blacklisted tags
        filtered_tags = [tag for tag in tags if tag not in blacklist_tags]

        # Select top N tags if max_tags is set
        if args.max_tags:
            selected_tags = filtered_tags[:args.max_tags]
        else:
            selected_tags = filtered_tags.copy()
        
        # Add always-tags if set and present in original tags
        if always_tags:
            for tag in always_tags:
                if tag in tags and tag not in selected_tags:
                    selected_tags.append(tag)
        
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(', '.join(selected_tags))
    except Exception as e:
        logging.error(f"Failed to write prompt.txt at {prompt_path}: {e}")

def process_video(video_page_url):
    """
    Process a single video: parse, download, extract multiple sequences of frames, and save.

    Args:
        video_page_url (str): URL of the video page.
    """
    if limit_reached.is_set():
        return

    video_url, tags = parse_video_page(video_page_url)
    if not video_url:
        logging.warning(f"Skipping video page {video_page_url} due to missing video URL.")
        return

    # Create a temporary directory for downloading the video
    temp_dir = os.path.join(output_dir, f'temp_{uuid.uuid4().hex}')
    os.makedirs(temp_dir, exist_ok=True)

    # Define video file path
    video_filename = os.path.basename(video_url.split('?')[0])  # Remove query params
    video_path = os.path.join(temp_dir, video_filename)

    # Download video
    success = download_video(video_url, video_path)
    if not success:
        logging.error(f"Failed to download video from {video_url}.")
        shutil.rmtree(temp_dir)  # Clean up incomplete directory
        return

    # Extract sequences
    extracted_sequences = extract_sequences(
        video_path,
        output_dir,
        args.max_sequences_per_video,
        args.frames_per_sequence,
        args.extraction_framerate,
        args.resolution,
        args.aspect_ratio
    )

    if not extracted_sequences:
        logging.error(f"Failed to extract any sequences from {video_path}.")
        shutil.rmtree(temp_dir)  # Clean up incomplete directory
        return

    # Save prompt.txt for each sequence
    for seq_dir in extracted_sequences:
        prompt_path = os.path.join(seq_dir, 'prompt.txt')
        save_prompt(tags, prompt_path)

    # Remove the temporary video directory to save space
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        logging.error(f"Failed to remove temporary directory {temp_dir}: {e}")

    logging.info(f"Processed video and extracted {len(extracted_sequences)} sequences from {video_page_url}")

def main():
    global args, output_dir, blacklist_tags, always_tags
    setup_logging()
    args = parse_arguments()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Parse blacklist-tags and always-tags into sets
    if args.blacklist_tags:
        blacklist_tags = set(tag.strip() for tag in args.blacklist_tags.split(',') if tag.strip())
        logging.info(f"Blacklist tags: {', '.join(blacklist_tags)}")
    else:
        blacklist_tags = set()
    
    if args.always_tags:
        always_tags = set(tag.strip() for tag in args.always_tags.split(',') if tag.strip())
        logging.info(f"Always-include tags: {', '.join(always_tags)}")
    else:
        always_tags = set()

    setup_requests_session()

    # Handle termination signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Log in if credentials are provided
    login()

    # Initialize video_counter based on existing dataset
    initialize_counters()

    # If a total number of sequences is set, initialize sequence_counter
    if args.num_sequences:
        logging.info(f"Total sequences to extract: {args.num_sequences}")
    else:
        logging.info("No total sequence limit set. Extracting sequences from all available videos.")

    # Use ThreadPoolExecutor for concurrent processing
    logging.info("Starting to collect and process video pages...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        pid = 0
        while not limit_reached.is_set():
            list_page_url = urljoin(args.base_url, f"index.php?page=post&s=list&tags=sort%3Ascore+animated+&pid={pid}")
            try:
                response = session.get(list_page_url, timeout=10)
                response.raise_for_status()
            except requests.RequestException as e:
                logging.critical(f"Failed to retrieve list page {list_page_url}: {e}")
                break

            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=re.compile(r'index\.php\?page=post&s=view&id=\d+'))
            page_urls = set()
            for link in links:
                href = link.get('href')
                if href and href not in page_urls:
                    full_url = urljoin(args.base_url, href)
                    page_urls.add(href)
                    futures.append(executor.submit(process_video, full_url))

            logging.info(f"Found {len(page_urls)} video links on page PID={pid}.")

            if len(page_urls) == 0:
                logging.info("No more video links found. Ending pagination.")
                break

            pid += 42  # Assuming pid increments by 10 per page
            time.sleep(1)  # 2-second delay between requests

            if args.num_sequences and sequence_counter >= args.num_sequences:
                logging.info(f"Reached the total sequence limit of {args.num_sequences}. Stopping pagination.")
                break

        # Use tqdm to show progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Videos"):
            try:
                future.result()
                # After processing each video, check if the total sequence limit is reached
                if args.num_sequences and sequence_counter >= args.num_sequences:
                    logging.info(f"Reached the total sequence limit of {args.num_sequences}. No more sequences will be extracted.")
                    limit_reached.set()
                    break  # Exit the loop to allow executor to shutdown gracefully
            except Exception as e:
                logging.critical(f"Exception occurred while processing a video: {e}")
                # Depending on the severity, you might choose to exit
                # sys.exit(1)

    logging.info("All videos have been processed.")

if __name__ == '__main__':
    main()