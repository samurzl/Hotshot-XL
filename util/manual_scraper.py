import argparse
import os
import re
import sys
import logging
import json
import math
import subprocess
import cloudscraper
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import uuid
import shutil

def setup_logging():
    """Configure logging for the script with separate handlers for console and file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the root logger level to INFO

    # Formatter for both handlers
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

    # File handler - logs all messages
    file_handler = logging.FileHandler("script.log", mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler - logs INFO and higher messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Download a video and extract sequences at specified timestamps.")
    parser.add_argument('--video-url', type=str, required=True, help='URL of the video to download.')
    parser.add_argument('--timestamps', type=str, help='Comma-separated list of timestamps (e.g., "0m0s0f,1m20s5f")')
    parser.add_argument('--timestamps-file', type=str, help='File containing list of timestamps, one per line.')
    parser.add_argument('--output-dir', type=str, default='dataset', help='Directory to save the extracted sequences.')
    parser.add_argument('--min-resolution', type=int, help='Minimum total number of pixels (e.g., 640*480).')
    parser.add_argument('--resolution', type=int, required=True, help='Total number of pixels for output frames (e.g., 1024^2 for 1024x1024).')
    parser.add_argument('--aspect-ratio', type=float, required=False, help='Aspect ratio (width divided by height). If omitted, uses the original aspect ratio of the video.')
    parser.add_argument('--frame-rate', type=float, required=True, help='Frame rate at which to extract frames.')
    parser.add_argument('--frames-per-sequence', type=int, required=True, help='Number of frames per sequence.')
    parser.add_argument('--username', type=str, default=None, help='Username for login (if required).')
    parser.add_argument('--password', type=str, default=None, help='Password for login (if required).')
    return parser.parse_args()

def calculate_dimensions(total_pixels, aspect_ratio):
    """
    Calculate width and height from total pixels and aspect ratio.
    """
    height = int(round(math.sqrt(total_pixels / aspect_ratio)))
    width = int(round(aspect_ratio * height))
    return width, height

def setup_session(args):
    """
    Set up a cloudscraper session with appropriate headers.
    """
    session = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'mobile': False
        }
    )
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/113.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': args.video_url
    })
    return session

def login(session, base_url, username, password):
    """
    Log in to the website if authentication is required.
    """
    if username and password:
        login_url = urljoin(base_url, "index.php?page=account&s=login")
        payload = {
            'username': username,
            'password': password,
            'commit': 'Log in'
        }
        try:
            response = session.post(login_url, data=payload, timeout=10)
            response.raise_for_status()
            if 'logout' not in response.text.lower():
                logging.critical("Login failed. Check your credentials.")
                sys.exit(1)
            logging.info("Logged in successfully.")
        except Exception as e:
            logging.critical(f"Failed to log in: {e}")
            sys.exit(1)
    else:
        logging.info("No login credentials provided. Proceeding without login.")

def get_video_url_from_page(session, page_url):
    """
    Parse the video page to extract the direct video URL.
    """
    try:
        response = session.get(page_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to retrieve video page {page_url}: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    video_tag = soup.find('video')
    if not video_tag:
        logging.warning(f"No <video> tag found in {page_url}.")
        return None

    source_tag = video_tag.find('source', src=True)
    if not source_tag:
        logging.warning(f"No <source> tag with src found in {page_url}.")
        return None

    video_url = source_tag['src']
    if not video_url.startswith('http'):
        video_url = urljoin(page_url, video_url.lstrip('/'))

    return video_url

def download_video(session, video_url, video_path):
    """
    Download the video from the given URL to the specified path.

    Args:
        session: The cloudscraper session to use for downloading.
        video_url (str): URL of the video to download.
        video_path (str): Local file path to save the video.

    Returns:
        Boolean indicating success or failure.
    """
    try:
        with session.get(video_url, stream=True, timeout=30) as r:
            r.raise_for_status()

            # Check if the response is a video
            content_type = r.headers.get('Content-Type', '')
            if not content_type.startswith('video/'):
                logging.error(f"Unexpected Content-Type: {content_type}. Expected a video.")
                return False

            with open(video_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logging.info(f"Downloaded video to {video_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to download video {video_url}: {e}")
        return False

def get_video_info(video_path):
    """
    Retrieve video duration, frame rate, width, and height using ffprobe.

    Args:
        video_path (str): Path to the video file.

    Returns:
        Tuple (frame_rate, duration_in_seconds, width, height) or (None, None, None, None) if failed.
    """
    try:
        # Get duration, frame rate, width, and height
        cmd_info = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=duration,r_frame_rate,width,height',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd_info, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        info = json.loads(result.stdout)

        if 'streams' not in info or len(info['streams']) == 0:
            logging.error(f"No video streams found in {video_path}.")
            return None, None, None, None

        stream = info['streams'][0]
        duration = float(stream.get('duration', 0))
        r_frame_rate = stream.get('r_frame_rate', '0/0')
        width = int(stream.get('width', 0))
        height = int(stream.get('height', 0))

        # Calculate frame rate
        if '/' in r_frame_rate:
            num, denom = map(int, r_frame_rate.split('/'))
            frame_rate = num / denom if denom != 0 else 0
        else:
            frame_rate = float(r_frame_rate) if r_frame_rate else 0

        return frame_rate, duration, width, height
    except subprocess.TimeoutExpired:
        logging.error(f"ffprobe timed out while processing {video_path}.")
        return None, None, None, None
    except Exception as e:
        logging.error(f"Failed to retrieve video info for {video_path}: {e}")
        return None, None, None, None

def parse_timestamp(timestamp_str, frame_rate):
    """
    Parse timestamp string in the format XmYsZf and convert to seconds.

    Args:
        timestamp_str (str): Timestamp string (e.g., "2m15s10f").
        frame_rate (float): Frame rate of the video.

    Returns:
        Float: timestamp in seconds, or None if parsing failed.
    """
    pattern = r'^(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+)s)?(?:(?P<frames>\d+)f)?$'
    match = re.match(pattern, timestamp_str)
    if not match:
        logging.warning(f"Invalid timestamp format: {timestamp_str}")
        return None

    minutes = match.group('minutes')
    seconds = match.group('seconds')
    frames = match.group('frames')

    total_seconds = 0.0
    if minutes:
        total_seconds += int(minutes) * 60
    if seconds:
        total_seconds += int(seconds)
    if frames:
        if frame_rate > 0:
            total_seconds += int(frames) / frame_rate
        else:
            logging.warning(f"Frame rate is zero, cannot calculate frame time for {timestamp_str}")
            return None

    return total_seconds

def extract_frame(video_path, timestamp_in_seconds, output_frame_path, width=None, height=None):
    """
    Extract a frame from the video at the specified timestamp.

    Args:
        video_path (str): Path to the video file.
        timestamp_in_seconds (float): Timestamp in seconds.
        output_frame_path (str): Path to save the extracted frame.
        width (int): Width of the output frame.
        height (int): Height of the output frame.
    """
    try:
        cmd = [
            'ffmpeg',
            '-ss', f'{timestamp_in_seconds:.6f}',
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '2'  # High quality
        ]
        if width and height:
            cmd.extend(['-vf', f'scale={width}:{height}'])
        cmd.extend([
            output_frame_path,
            '-y'  # Overwrite without asking
        ])
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logging.info(f"Extracted frame at {timestamp_in_seconds}s to {output_frame_path}")
    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.strip()
        logging.error(f"FFmpeg failed to extract frame at {timestamp_in_seconds}s from {video_path}: {stderr_output}")
    except Exception as e:
        logging.error(f"Failed to extract frame at {timestamp_in_seconds}s: {e}")

def extract_sequence(video_path, start_time, sequence_dir, n_frames, extraction_fps, video_duration, width=None, height=None):
    """
    Extract a sequence of frames from the video starting at start_time.

    Args:
        video_path (str): Path to the video file.
        start_time (float): Start time in seconds.
        sequence_dir (str): Directory to save the frames.
        n_frames (int): Number of frames to extract.
        extraction_fps (float): Frame rate at which to extract frames.
        video_duration (float): Duration of the video in seconds.
        width (int): Width to scale the frames to.
        height (int): Height to scale the frames to.
    """
    os.makedirs(sequence_dir, exist_ok=True)

    # Create empty prompt.txt file
    prompt_path = os.path.join(sequence_dir, 'prompt.txt')
    open(prompt_path, 'w').close()

    frame_interval = 1.0 / extraction_fps
    extracted_frames = []
    for i in range(n_frames):
        timestamp = start_time + i * frame_interval
        if timestamp > video_duration:
            logging.warning(f"Timestamp {timestamp}s exceeds video duration {video_duration}s. Stopping sequence extraction.")
            break
        output_frame_path = os.path.join(sequence_dir, f'frame_{i+1:05d}.png')
        extract_frame(video_path, timestamp, output_frame_path, width, height)
        extracted_frames.append(output_frame_path)

    # Create low-resolution GIF
    try:
        if extracted_frames:
            gif_path = os.path.join(sequence_dir, 'preview.gif')
            # Set GIF width to 256 pixels, adjust height to maintain aspect ratio
            gif_width = 256
            gif_height = int(height * (gif_width / width)) if width != 0 else 256
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-framerate', str(extraction_fps),
                '-i', os.path.join(sequence_dir, 'frame_%05d.png'),
                '-vf', f'scale={gif_width}:{gif_height}',
                '-loop', '0',  # Loop indefinitely
                gif_path
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            logging.info(f"Created low-resolution GIF at {gif_path}")
        else:
            logging.warning("No frames extracted; skipping GIF creation.")
    except Exception as e:
        logging.error(f"Failed to create GIF: {e}")

def main():
    setup_logging()
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    session = setup_session(args)

    # If credentials are provided, log in
    if args.username and args.password:
        base_url = '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(args.video_url))
        login(session, base_url, args.username, args.password)

    # Check if the video URL is a direct link or a page URL
    video_url = args.video_url
    if not video_url.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        logging.info("Provided URL does not seem to be a direct video file. Attempting to extract video URL from the page.")
        video_url = get_video_url_from_page(session, video_url)
        if not video_url:
            logging.error("Failed to extract video URL from the page.")
            sys.exit(1)

    # Save video to a temporary directory
    temp_dir = os.path.join(args.output_dir, f'temp_{uuid.uuid4().hex}')
    os.makedirs(temp_dir, exist_ok=True)
    video_filename = os.path.basename(video_url.split('?')[0])
    video_path = os.path.join(temp_dir, video_filename)

    success = download_video(session, video_url, video_path)
    if not success:
        logging.error(f"Failed to download video from {video_url}.")
        shutil.rmtree(temp_dir)  # Clean up incomplete directory
        sys.exit(1)

    # Get video info
    frame_rate, duration, video_width, video_height = get_video_info(video_path)
    if frame_rate is None or duration is None or video_width is None or video_height is None:
        logging.error("Failed to get video info.")
        shutil.rmtree(temp_dir)
        sys.exit(1)

    # Calculate aspect ratio
    if args.aspect_ratio:
        aspect_ratio = args.aspect_ratio
        logging.info(f"Using specified aspect ratio: {aspect_ratio}")
    else:
        if video_height != 0:
            aspect_ratio = video_width / video_height
            logging.info(f"No aspect ratio specified. Using video's original aspect ratio: {aspect_ratio}")
        else:
            aspect_ratio = 1.0
            logging.warning("Video height is zero. Defaulting aspect ratio to 1.0.")

    # Calculate output width and height based on total pixels and aspect ratio
    output_width, output_height = calculate_dimensions(args.resolution, aspect_ratio)
    logging.info(f"Output frames will be scaled to {output_width}x{output_height} pixels.")

    # Check if video meets minimum resolution
    if args.min_resolution:
        total_pixels = video_width * video_height
        if total_pixels < args.min_resolution:
            logging.error(f"Video resolution {video_width}x{video_height} ({total_pixels} pixels) is below minimum {args.min_resolution} pixels. Skipping extraction.")
            shutil.rmtree(temp_dir)
            sys.exit(1)

    # Get the list of timestamps
    timestamps = []

    if args.timestamps:
        # Split the timestamps string by commas
        timestamps.extend([ts.strip() for ts in args.timestamps.split(',') if ts.strip()])

    if args.timestamps_file:
        try:
            with open(args.timestamps_file, 'r') as f:
                file_timestamps = [line.strip() for line in f if line.strip()]
                timestamps.extend(file_timestamps)
        except Exception as e:
            logging.error(f"Failed to read timestamps from file {args.timestamps_file}: {e}")
            shutil.rmtree(temp_dir)
            sys.exit(1)

    if not timestamps:
        logging.error("No valid timestamps provided.")
        shutil.rmtree(temp_dir)
        sys.exit(1)

    # Parse timestamps
    timestamps_in_seconds = []
    for ts_str in timestamps:
        ts_in_seconds = parse_timestamp(ts_str, frame_rate)
        if ts_in_seconds is not None:
            timestamps_in_seconds.append((ts_str, ts_in_seconds))
        else:
            logging.warning(f"Invalid timestamp format: {ts_str}")

    if not timestamps_in_seconds:
        logging.error("No valid timestamps after parsing.")
        shutil.rmtree(temp_dir)
        sys.exit(1)

    # Initialize vid_counter based on existing sequences
    existing_sequences = [d for d in os.listdir(args.output_dir) if re.match(r'^vid_\d{4}$', d)]
    if existing_sequences:
        existing_vid_numbers = [int(re.search(r'\d{4}', d).group()) for d in existing_sequences]
        vid_counter = max(existing_vid_numbers) + 1
        logging.info(f"Found existing sequences. Starting vid_counter at {vid_counter}.")
    else:
        vid_counter = 1
        logging.info("No existing sequences found. Starting vid_counter at 1.")

    # Extract sequences
    for ts_str, ts_in_seconds in timestamps_in_seconds:
        # Create sequence directory
        seq_dir = os.path.join(args.output_dir, f'vid_{vid_counter:04d}')
        extract_sequence(
            video_path,
            ts_in_seconds,
            seq_dir,
            args.frames_per_sequence,
            args.frame_rate,
            duration,
            output_width,
            output_height
        )
        vid_counter += 1

    # Remove the temporary video directory to save space
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        logging.error(f"Failed to remove temporary directory {temp_dir}: {e}")

    logging.info("All sequences have been extracted.")

if __name__ == '__main__':
    main()
