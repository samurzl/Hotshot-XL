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

def setup_logging(log_file='scraper.log'):
    """
    Configure logging to write warnings and errors to a log file, overwriting it on each run.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # Overwrite the log file at each run
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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

def scrape_videos(num_sequences, start_id, output_dir, target_resolution, frames_per_sequence=16):
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

                # Check frame count before processing
                frame_count = get_frame_count(video_path)
                if frame_count < frames_per_sequence:
                    logging.warning(f"Video ID {current_id} has only {frame_count} frames. Skipping.")
                    os.remove(video_path)
                    current_id += 1
                    continue

                # Process the video (extract frames)
                frames_dir = process_video(video_path, target_resolution)

                # Chop frames into sequences
                sequences_created = chop_into_sequences(
                    frames_dir,
                    output_dir,
                    frames_per_sequence,
                    global_sequence_num,
                    num_digits,
                    num_sequences=num_sequences,
                    downloaded_sequences=downloaded_sequences
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

                    # **Removed the captions.txt creation**
                    # If you ever need to add other files or perform additional actions, you can do so here.

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

def process_video(video_path, target_resolution):
    """
    Extract frames from the video using ffmpeg.
    Returns the directory containing the extracted frames.
    """
    # Create a temporary frames directory
    frames_dir = os.path.join(os.path.dirname(video_path), "frames_temp")
    os.makedirs(frames_dir, exist_ok=True)

    # Define the ffmpeg command (stretch video to fill target resolution)
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'scale={target_resolution}',
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

def chop_into_sequences(frames_dir, output_dir, frames_per_sequence, start_sequence_num, num_digits, num_sequences, downloaded_sequences):
    """
    Chop extracted frames into 16-frame sequences and save them as individual training examples.
    Returns the number of sequences created.
    """
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    total_frames = len(frame_files)
    sequences_created = 0

    for i in range(0, total_frames, frames_per_sequence):
        if sequences_created >= (num_sequences - downloaded_sequences):
            break  # Stop if we've reached the desired number of sequences

        sequence_frames = frame_files[i:i + frames_per_sequence]
        if len(sequence_frames) < frames_per_sequence:
            break  # Skip incomplete sequences

        sequence_dir = os.path.join(output_dir, f"vid_{str(start_sequence_num + sequences_created).zfill(num_digits)}")
        os.makedirs(sequence_dir, exist_ok=True)

        for idx, frame in enumerate(sequence_frames):
            src = os.path.join(frames_dir, frame)
            dst = os.path.join(sequence_dir, f"frame_{idx:05d}.png")  # Numbering starts from 0
            try:
                shutil.copy(src, dst)
            except Exception as e:
                logging.error(f"Failed to copy {src} to {dst}: {e}")

        sequences_created += 1

    logging.info(f"Chopped frames into {sequences_created} sequences.")
    return sequences_created

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Sakugabooru Video Scraper')
    parser.add_argument('--num_videos', type=int, default=1, help='Number of sequences to extract')
    parser.add_argument('--start_id', type=int, default=1, help='Starting post ID')
    parser.add_argument('--output_dir', type=str, default='../datasets/testv1', help='Output directory')
    parser.add_argument('--resolution', type=str, default='512:288', help='Target resolution (e.g., 1280:720)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Set up logging to a file, overwriting previous logs
    setup_logging()

    # Set up a requests session with retries
    session = setup_session()

    # Parse command-line arguments
    args = parse_arguments()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Start scraping
    scrape_videos(
        num_sequences=args.num_videos,
        start_id=args.start_id,
        output_dir=args.output_dir,
        target_resolution=args.resolution,
        frames_per_sequence=16
    )
