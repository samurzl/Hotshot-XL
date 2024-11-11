import requests
from bs4 import BeautifulSoup
import os
import time
import subprocess
import argparse
import re
import logging
import shutil
import math
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import signal
import sys

def setup_logging(log_file='scraper.log', verbose=False):
    """
    Configure logging to write warnings and errors to a log file, overwriting it on each run.
    If verbose is True, also output INFO level logs to the console.
    """
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # Overwrite the log file at each run
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    if verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def setup_session(retries=5, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)):
    """
    Set up a requests Session with retry strategy using the updated 'allowed_methods'.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # Updated parameter to replace deprecated 'method_whitelist'
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def scrape_videos(num_sequences, start_id, output_dir, target_resolution, aspect_ratio,
                 frames_per_sequence=16, max_sequences_per_video=None, sampling_framerate=None,
                 min_resolution_ratio=0.5):
    downloaded_sequences = 0
    current_id = start_id
    global_sequence_num = 1  # To ensure unique sequence folder names across all videos

    # Determine the number of digits needed for leading zeros
    # Assuming a maximum of 999999 sequences; adjust as needed
    num_digits = 6

    while downloaded_sequences < num_sequences:
        url = f"https://sakugabooru.com/post/show/{current_id}"
        try:
            response = session.get(url)
            if response.status_code != 200:
                logging.warning(f"Post ID {current_id} not found. Skipping.")
                current_id += 1
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the video URL
            video_url = find_video_url(soup)
            if video_url:
                logging.info(f"Found video URL: {video_url}")

                # Download the video
                video_filename = get_filename_from_url(video_url)
                video_path = os.path.join(output_dir, video_filename)
                if not os.path.exists(video_path):
                    download_video(session, video_url, video_path)
                else:
                    logging.warning(f"Video already downloaded at {video_path}")

                # Get video resolution
                video_width, video_height = get_video_resolution(video_path)
                if video_width is None or video_height is None:
                    logging.warning(f"Could not determine resolution for Video ID {current_id}. Skipping.")
                    os.remove(video_path)
                    current_id += 1
                    continue

                # Calculate target dimensions
                target_width = target_resolution * math.sqrt(aspect_ratio)
                target_height = target_resolution / math.sqrt(aspect_ratio)

                # Calculate minimum acceptable dimensions
                min_width = target_width * min_resolution_ratio
                min_height = target_height * min_resolution_ratio

                # Round dimensions to nearest integer
                target_width = int(round(target_width))
                target_height = int(round(target_height))
                min_width = int(round(min_width))
                min_height = int(round(min_height))

                logging.info(f"Video ID {current_id} resolution: {video_width}x{video_height}")
                logging.info(f"Minimum required resolution: {min_width}x{min_height}")

                # Check if video meets minimum resolution
                if video_width < min_width or video_height < min_height:
                    logging.warning(f"Video ID {current_id} resolution {video_width}x{video_height} "
                                    f"is below the minimum required {min_width}x{min_height}. Skipping.")
                    os.remove(video_path)
                    current_id += 1
                    continue

                # Check frame count before processing
                frame_count = get_frame_count(video_path)
                if frame_count < frames_per_sequence:
                    logging.warning(f"Video ID {current_id} has only {frame_count} frames. Skipping.")
                    os.remove(video_path)
                    current_id += 1
                    continue

                # Process the video (extract frames)
                frames_dir = process_video(video_path, target_resolution, aspect_ratio, sampling_framerate)

                # Chop frames into sequences
                sequences_created = chop_into_sequences(
                    frames_dir,
                    output_dir,
                    frames_per_sequence,
                    global_sequence_num,
                    num_digits,
                    num_sequences=num_sequences,
                    downloaded_sequences=downloaded_sequences,
                    max_sequences_per_video=max_sequences_per_video
                )
                global_sequence_num += sequences_created
                downloaded_sequences += sequences_created

                # Delete the frames directory after chopping
                shutil.rmtree(frames_dir)

                # Delete the original video
                try:
                    os.remove(video_path)
                    logging.info(f"Deleted original video at {video_path}")
                except Exception as e:
                    logging.error(f"Failed to delete video {video_path}: {e}")

                # Extract and save tags
                tags = extract_tags(soup)
                # Save tags to each sequence
                for seq_num in range(sequences_created):
                    sequence_number = global_sequence_num - sequences_created + seq_num
                    seq_dir = os.path.join(output_dir, f"vid_{str(sequence_number).zfill(num_digits)}")
                    prompt_file = os.path.join(seq_dir, "prompt.txt")  # Changed from tags.txt to prompt.txt
                    try:
                        with open(prompt_file, 'w') as f:
                            f.write(' '.join(tags))
                    except Exception as e:
                        logging.error(f"Failed to write prompt for {seq_dir}: {e}")

                print(f"Progress: {downloaded_sequences}/{num_sequences} sequences extracted.", end='\r')

            else:
                logging.warning(f"No video found at post ID {current_id}. Skipping.")

        except Exception as e:
            logging.error(f"Error processing post ID {current_id}: {e}")

        current_id += 1
        time.sleep(1)  # Be polite and avoid overloading the server

    print(f"\nScraping completed.")

def find_video_url(soup):
    """
    Extract the video URL from the BeautifulSoup object.
    """
    # Try to find the high-resolution link
    highres_link = soup.find('a', {'id': 'highres'})
    if highres_link and 'href' in highres_link.attrs:
        return highres_link['href']
    else:
        # Alternative method: find the file_url in the script tag
        scripts = soup.find_all('script', {'type': 'text/javascript'})
        for script in scripts:
            if 'Post.register_resp' in script.text:
                try:
                    match = re.search(r'Post\.register_resp\((.*?)\);', script.text, re.DOTALL)
                    if match:
                        data = match.group(1)
                        # Replace single quotes with double quotes to make it valid JSON
                        data = data.replace("'", '"')
                        # Unescape forward slashes
                        data = data.replace('\\/', '/')
                        import json
                        json_data = json.loads(data)
                        file_url = json_data['posts'][0]['file_url']
                        return file_url
                except Exception as e:
                    logging.error(f"Error parsing script for video URL: {e}")
    return None

def get_filename_from_url(url):
    """
    Extract the filename from the URL.
    """
    return url.split('/')[-1].split('?')[0]

def download_video(session, video_url, save_path):
    """
    Download the video using the provided session.
    """
    headers = {
        'Referer': 'https://sakugabooru.com/',
        'User-Agent': 'SakugabooruVideoScraper/1.0 (your_email@example.com)'  # Replace with your contact info
    }
    try:
        with session.get(video_url, stream=True, headers=headers) as response:
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logging.info(f"Downloaded video to {save_path}")
    except Exception as e:
        logging.error(f"Failed to download video from {video_url}: {e}")
        raise

def get_frame_count(video_path):
    """
    Use ffprobe to get the number of frames in the video.
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-count_frames',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames',
        '-of', 'default=nokey=1:noprint_wrappers=1',
        video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
        frame_count = int(result.stdout.strip())
        logging.info(f"Video {video_path} has {frame_count} frames.")
        return frame_count
    except Exception as e:
        logging.error(f"Error getting frame count for {video_path}: {e}")
        return 0

def get_video_resolution(video_path):
    """
    Use ffprobe to get the video's width and height.
    Returns (width, height) or (None, None) if unable to determine.
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0',
        video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
        width, height = map(int, result.stdout.strip().split('x'))
        return width, height
    except Exception as e:
        logging.error(f"Error getting resolution for {video_path}: {e}")
        return None, None

def process_video(video_path, target_resolution, aspect_ratio, sampling_framerate):
    """
    Extract frames from the video using ffmpeg.
    Returns the directory containing the extracted frames.
    """
    # Create a temporary frames directory
    frames_dir = os.path.join(os.path.dirname(video_path), "frames_temp")
    os.makedirs(frames_dir, exist_ok=True)

    # Calculate width and height based on target_resolution and aspect_ratio
    R = target_resolution
    A = aspect_ratio

    # Calculate width and height to maintain total pixels and aspect ratio
    width = R * math.sqrt(A)
    height = R / math.sqrt(A)

    # Round to the nearest even integer
    width = int(round(width))
    height = int(round(height))

    # Ensure width and height are even
    if width % 2 != 0:
        width += 1
    if height % 2 != 0:
        height += 1

    # Log the calculated dimensions
    logging.info(f"Calculated dimensions: width={width}, height={height} for resolution={R} and aspect_ratio={A}")

    # Define the aspect ratio handling filter
    aspect_ratio_filter = f"setsar=1, scale={width}:{height}:force_original_aspect_ratio=decrease, pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"

    # Define the sampling filter for framerate
    if sampling_framerate:
        framerate_filter = f"fps={sampling_framerate}"
        vf_filters = f"{framerate_filter}, {aspect_ratio_filter}"
    else:
        vf_filters = aspect_ratio_filter

    # Define the ffmpeg command
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', vf_filters,
        os.path.join(frames_dir, 'frame_%05d.png')
    ]

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logging.info(f"Extracted frames to {frames_dir}")
        return frames_dir
    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing video {video_path}: {e}")
        return frames_dir

def extract_tags(soup):
    """
    Extract tags from the BeautifulSoup object.
    """
    tag_list = []
    # First, try to find tags in the textarea (provided in your code snippet)
    textarea = soup.find('textarea', {'id': 'post_tags'})
    if textarea:
        tags_text = textarea.get_text(strip=True)
        tag_list = tags_text.split()
        return tag_list

    # Fallback to parsing the tag sidebar
    tag_sidebar = soup.find('ul', {'id': 'tag-sidebar'})
    if tag_sidebar:
        tags = tag_sidebar.find_all('li')
        for tag in tags:
            tag_link = tag.find('a', href=True, text=True)
            if tag_link:
                tag_name = tag_link.text.strip()
                tag_list.append(tag_name)
    return tag_list

def chop_into_sequences(frames_dir, output_dir, frames_per_sequence, start_sequence_num, num_digits,
                        num_sequences, downloaded_sequences, max_sequences_per_video=None):
    """
    Chop extracted frames into sequences and save them as individual training examples.
    Returns the number of sequences created.
    """
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    total_frames = len(frame_files)
    total_possible_sequences = total_frames // frames_per_sequence

    sequences_created = 0
    sequences_from_current_video = 0

    if max_sequences_per_video and max_sequences_per_video < total_possible_sequences:
        # Calculate the interval to spread sequences evenly
        interval = total_possible_sequences / max_sequences_per_video
        sequence_indices = [int(i * interval) for i in range(max_sequences_per_video)]
    else:
        # Take all possible sequences
        sequence_indices = list(range(total_possible_sequences))

    for idx in sequence_indices:
        if sequences_created >= (num_sequences - downloaded_sequences):
            break  # Stop if we've reached the desired number of sequences

        if max_sequences_per_video is not None and sequences_from_current_video >= max_sequences_per_video:
            break  # Stop if we've reached the max sequences per video

        # Calculate the starting frame index
        start_frame = idx * frames_per_sequence
        sequence_frames = frame_files[start_frame:start_frame + frames_per_sequence]
        if len(sequence_frames) < frames_per_sequence:
            continue  # Skip incomplete sequences

        sequence_dir = os.path.join(output_dir, f"vid_{str(start_sequence_num + sequences_created).zfill(num_digits)}")
        os.makedirs(sequence_dir, exist_ok=True)

        for frame_idx, frame in enumerate(sequence_frames):
            src = os.path.join(frames_dir, frame)
            dst = os.path.join(sequence_dir, f"frame_{frame_idx:05d}.png")  # Numbering starts from 0
            try:
                shutil.copy(src, dst)
            except Exception as e:
                logging.error(f"Failed to copy {src} to {dst}: {e}")

        sequences_created += 1
        sequences_from_current_video += 1

    logging.info(f"Chopped frames into {sequences_created} sequences.")
    return sequences_created

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Sakugabooru Video Scraper')

    parser.add_argument('--num_sequences', type=int, default=1,
                        help='Total number of sequences to extract')
    parser.add_argument('--start_id', type=int, default=1,
                        help='Starting post ID')
    parser.add_argument('--output_dir', type=str, default='../datasets/testv1',
                        help='Output directory')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Target resolution width equivalent to 512x512 pixels (e.g., 512)')
    parser.add_argument('--aspect_ratio', type=float, default=1.0,
                        help='Aspect ratio of the saved frames (width / height, e.g., 1.75)')
    parser.add_argument('--frames_per_sequence', type=int, default=16,
                        help='Number of frames per sequence')
    parser.add_argument('--max_sequences_per_video', type=int, default=None,
                        help='Maximum number of sequences to extract from each video')
    parser.add_argument('--sampling_framerate', type=int, default=None,
                        help='Framerate for frame extraction (e.g., 30)')
    parser.add_argument('--min_resolution_ratio', type=float, default=0.5,
                        help='Minimum resolution ratio relative to output resolution (e.g., 0.5)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging to console')

    args = parser.parse_args()
    return args

def check_dependencies():
    """
    Check if ffmpeg and ffprobe are installed.
    """
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(['ffprobe', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        logging.error("ffmpeg and ffprobe must be installed and accessible in the system's PATH.")
        print("Error: ffmpeg and ffprobe must be installed and accessible in the system's PATH.")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("ffmpeg and ffprobe must be installed and accessible in the system's PATH.")
        print("Error: ffmpeg and ffprobe must be installed and accessible in the system's PATH.")
        sys.exit(1)

def signal_handler(sig, frame):
    print('\nInterrupted! Cleaning up...')
    sys.exit(0)

if __name__ == "__main__":
    # Register the signal handler for graceful interrupt handling
    signal.signal(signal.SIGINT, signal_handler)

    # Set up logging to a file, overwriting previous logs
    # Verbose mode can be enabled via command-line argument
    # We'll set verbose=False here and adjust after parsing arguments
    setup_logging(verbose=False)

    # Set up a requests session with retries
    session = setup_session()

    # Parse command-line arguments
    args = parse_arguments()

    # Reconfigure logging based on verbose flag
    # Remove all handlers and set up again
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    setup_logging(log_file='scraper.log', verbose=args.verbose)

    # Check dependencies
    check_dependencies()

    # Validate aspect_ratio
    if args.aspect_ratio <= 0:
        logging.error("Aspect ratio must be a positive number.")
        print("Error: Aspect ratio must be a positive number.")
        sys.exit(1)

    # Validate resolution
    if args.resolution <= 0:
        logging.error("Resolution must be a positive integer.")
        print("Error: Resolution must be a positive integer.")
        sys.exit(1)

    # Validate min_resolution_ratio
    if args.min_resolution_ratio <= 0:
        logging.error("Minimum resolution ratio must be a positive number.")
        print("Error: Minimum resolution ratio must be a positive number.")
        sys.exit(1)
    elif args.min_resolution_ratio > 1.0:
        logging.warning("Minimum resolution ratio greater than 1.0 might not make sense.")

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Start scraping
    scrape_videos(
        num_sequences=args.num_sequences,
        start_id=args.start_id,
        output_dir=args.output_dir,
        target_resolution=args.resolution,
        aspect_ratio=args.aspect_ratio,
        frames_per_sequence=args.frames_per_sequence,
        max_sequences_per_video=args.max_sequences_per_video,
        sampling_framerate=args.sampling_framerate,
        min_resolution_ratio=args.min_resolution_ratio
    )
