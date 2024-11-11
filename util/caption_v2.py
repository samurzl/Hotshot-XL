import os
import re
import argparse
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import json
import shutil
import logging
import tensorflow as tf
import deepdanbooru as dd
from typing import Union, List, Any, Iterable, Tuple

# Setup logging
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

# Function to load frames from a video folder
def load_frames(vid_folder, frame_step=1):
    """
    Load and sample frames from a video folder.

    Args:
        vid_folder (str): Path to the video folder containing frame images.
        frame_step (int): Step size for frame sampling.

    Returns:
        List[PIL.Image.Image] or None: List of loaded frames or None if no valid frames are found.
    """
    # Get list of frame images
    frame_files = [f for f in os.listdir(vid_folder) if re.match(r'frame_\d+\.(jpg|jpeg|png)', f, re.IGNORECASE)]
    if not frame_files:
        logging.warning(f"No frames found in {vid_folder}")
        return None
    # Sort frames numerically
    frame_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    # Sample frames
    frame_files = frame_files[::frame_step]
    # Load images
    frames = []
    for f in frame_files:
        try:
            img = Image.open(os.path.join(vid_folder, f)).convert('RGB')
            frames.append(img)
        except Exception as e:
            logging.error(f"Error loading image {f} in {vid_folder}: {e}")
            continue
    if not frames:
        logging.warning(f"No valid frames to process in {vid_folder}")
        return None
    return frames

# Function to evaluate a single image using DeepDanbooru
def evaluate_image(
    image_input: Union[str, bytes], model: Any, tags: List[str], threshold: float
) -> Iterable[Tuple[str, float]]:
    """
    Evaluate an image using DeepDanbooru to generate tags.

    Args:
        image_input (Union[str, bytes]): Path to the image or image bytes.
        model (Any): Loaded DeepDanbooru model.
        tags (List[str]): List of possible tags.
        threshold (float): Confidence threshold for tag inclusion.

    Yields:
        Tuple[str, float]: Tuple of tag and its confidence score.
    """
    width = model.input_shape[2]
    height = model.input_shape[1]

    image = dd.data.load_image_for_evaluate(image_input, width=width, height=height)

    image_shape = image.shape
    image = image.reshape((1, image_shape[0], image_shape[1], image_shape[2]))
    y = model.predict(image)[0]

    result_dict = {}

    for i, tag in enumerate(tags):
        result_dict[tag] = y[i]

    for tag in tags:
        if result_dict[tag] >= threshold:
            yield tag, result_dict[tag]

# Function to generate DeepDanbooru tags
def generate_deepdanbooru_tags(image_path, model, tags, threshold=0.5):
    """
    Generate tags using DeepDanbooru.

    Args:
        image_path (str): Path to the image.
        model (Any): Loaded DeepDanbooru model.
        tags (List[str]): List of possible tags.
        threshold (float): Confidence threshold for tag inclusion.

    Returns:
        str: Comma-separated string of tags.
    """
    try:
        tag_scores = evaluate_image(image_path, model, tags, threshold)
        tags_above_threshold = [tag for tag, score in tag_scores]
        tags_str = ', '.join(tags_above_threshold)
        return tags_str
    except Exception as e:
        logging.error(f"Error generating DeepDanbooru tags for {image_path}: {e}")
        return ""

# Function to extract existing tags from prompt.txt
def extract_existing_tags(prompt_path):
    """
    Extract existing tags from prompt.txt.

    Args:
        prompt_path (str): Path to prompt.txt.

    Returns:
        List[str]: List of existing tags.
    """
    if not os.path.isfile(prompt_path):
        return []
    with open(prompt_path, 'r') as f:
        content = f.read().strip()
    if not content:
        return []
    # Assume tags are before the first comma
    if ',' in content:
        tags_part = content.split(',', 1)[0]
    else:
        # If no comma, assume all content is tags
        tags_part = content
    # Split by comma or space
    tags = re.split(r',\s*|\s+', tags_part)
    # Clean tags
    tags = [tag.strip() for tag in tags if tag.strip()]
    return tags

# Function to process videos in a batch
def process_videos_in_batch(batch_vid_folders, qwen_model, qwen_processor, device, frame_step=1,
                            deepdanbooru_model=None, deepdanbooru_tags=None, threshold=0.5):
    """
    Process a batch of videos: generate tags with DeepDanbooru, generate captions with Qwen2, and write captions to prompt.txt.

    Args:
        batch_vid_folders (List[str]): List of video folder paths.
        qwen_model (Qwen2VLForConditionalGeneration): Loaded Qwen2 model.
        qwen_processor (AutoProcessor): Loaded Qwen2 processor.
        device (str): Device to run the models on ('cuda' or 'cpu').
        frame_step (int): Step size for frame sampling.
        deepdanbooru_model (Any): Loaded DeepDanbooru model.
        deepdanbooru_tags (List[str]): List of DeepDanbooru tags.
        threshold (float): Confidence threshold for tag inclusion.
    """
    batch_frames = []
    batch_texts = []
    vid_folder_names = []
    deepdanbooru_tags_list = []

    for vid_folder in batch_vid_folders:
        frames = load_frames(vid_folder, frame_step)
        if frames is None:
            continue
        # Select a key frame for DeepDanbooru (e.g., first frame)
        key_frame = frames[0]
        key_frame_path = os.path.join(vid_folder, 'key_frame.png')
        try:
            key_frame.save(key_frame_path)
        except Exception as e:
            logging.error(f"Error saving key frame for {vid_folder}: {e}")
            continue
        # Generate DeepDanbooru tags
        tags = generate_deepdanbooru_tags(key_frame_path, deepdanbooru_model, deepdanbooru_tags, threshold)
        deepdanbooru_tags_list.append(tags)
        # Remove the temporary key frame image
        try:
            os.remove(key_frame_path)
        except Exception as e:
            logging.warning(f"Could not remove temporary key frame {key_frame_path}: {e}")

        # Extract existing tags from prompt.txt
        prompt_path = os.path.join(vid_folder, 'prompt.txt')
        existing_tags = extract_existing_tags(prompt_path)
        # Combine existing tags with new tags
        if tags:
            new_tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
            combined_tags_set = set(existing_tags + new_tags)
        else:
            combined_tags_set = set(existing_tags)
        combined_tags = ', '.join(sorted(combined_tags_set)) if combined_tags_set else ''

        # Prepare messages for Qwen2, including combined tags
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": f"Tags: {combined_tags}"},
                    {"type": "text", "text": """Describe the video.
The description should only cover what happens, not what is seen."""},
                ],
            }
        ]
        try:
            text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            logging.error(f"Error preparing Qwen2 input for {vid_folder}: {e}")
            continue
        batch_frames.append(frames)
        batch_texts.append(text)
        vid_folder_names.append(vid_folder)

    if not batch_frames:
        logging.warning("No valid videos to process in this batch.")
        return

    # Prepare inputs for Qwen2
    try:
        inputs = qwen_processor(
            text=batch_texts,
            images=None,
            videos=batch_frames,
            padding=True,
            return_tensors="pt",
        ).to(device)
    except Exception as e:
        logging.error(f"Error preparing Qwen2 inputs: {e}")
        return

    # Inference with Qwen2
    try:
        with torch.no_grad():
            generated_ids = qwen_model.generate(**inputs, max_new_tokens=256)
    except Exception as e:
        logging.error(f"Error during Qwen2 inference: {e}")
        return

    # Process outputs and write to prompt.txt
    for idx, vid_folder in enumerate(vid_folder_names):
        try:
            in_ids = inputs.input_ids[idx]
            out_ids = generated_ids[idx]
            generated_ids_trimmed = out_ids[len(in_ids):]
            output_text_qwen2 = qwen_processor.decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        except Exception as e:
            logging.error(f"Error decoding Qwen2 output for {vid_folder}: {e}")
            output_text_qwen2 = ""

        # Get DeepDanbooru tags
        deepdanbooru_tags = deepdanbooru_tags_list[idx]

        # Print DeepDanbooru tags to console
        print(f"Tags for '{os.path.basename(vid_folder)}': {deepdanbooru_tags}")

        # Combine tags and captions, danbooru tags first
        if deepdanbooru_tags and combined_tags:
            # To avoid duplicating the original tags which are already in combined_tags
            # Assuming combined_tags includes existing_tags and new danbooru tags
            # So, to maintain the desired order, extract existing_tags
            existing_tags = extract_existing_tags(os.path.join(vid_folder, 'prompt.txt'))
            # Remove any overlapping tags between danbooru_tags and existing_tags
            # to prevent duplication
            existing_tags_set = set(existing_tags)
            danbooru_tags_set = set(deepdanbooru_tags.split(', '))
            unique_danbooru_tags = danbooru_tags_set - existing_tags_set
            # Combine unique danbooru tags and existing tags
            final_tags_set = unique_danbooru_tags.union(existing_tags_set)
            final_tags = ', '.join(sorted(final_tags_set)) if final_tags_set else ''

            combined_caption = f"{final_tags}, {output_text_qwen2}"
        elif deepdanbooru_tags:
            combined_caption = f"{deepdanbooru_tags}, {output_text_qwen2}"
        else:
            combined_caption = output_text_qwen2  # If DeepDanbooru failed, use only Qwen2 caption

        # Write output to prompt.txt
        prompt_path = os.path.join(vid_folder, 'prompt.txt')
        try:
            with open(prompt_path, 'w') as f:
                f.write(combined_caption)
            logging.info(f"Saved combined caption to {prompt_path}")
        except Exception as e:
            logging.error(f"Error writing to {prompt_path}: {e}")

        # Print the natural language caption to console
        print(f"Caption for '{os.path.basename(vid_folder)}': {output_text_qwen2}")
        print("--- End of Captions ---\n")
        print(f"Processed {vid_folder}\n")

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process videos and caption them using Qwen2 and DeepDanbooru.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset folder.')
    parser.add_argument('--num_videos', type=int, default=None, help='Number of videos to process.')
    parser.add_argument('--start_index', type=int, default=0, help='Index of the video to start processing from.')
    parser.add_argument('--start_video', type=str, default=None, help='Name of the video folder to start processing from.')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of videos to process in a batch.')
    parser.add_argument('--frame_step', type=int, default=1, help='Step size for frame sampling.')
    parser.add_argument('--deepdanbooru_model_path', type=str, required=True, help='Path to DeepDanbooru model file (e.g., model.h5).')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging to console.')
    args = parser.parse_args()
    
    dataset_folder = args.dataset
    num_videos = args.num_videos
    start_index = args.start_index
    start_video = args.start_video
    batch_size = args.batch_size
    frame_step = args.frame_step
    deepdanbooru_model_path = args.deepdanbooru_model_path
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Verify DeepDanbooru model path
    if not os.path.isfile(deepdanbooru_model_path):
        logging.error(f"DeepDanbooru model file not found at {deepdanbooru_model_path}")
        print(f"DeepDanbooru model file not found at {deepdanbooru_model_path}")
        return
    
    # Load the Qwen2 model and processor with optimizations
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16,  # Use FP16 for faster inference
            attn_implementation="flash_attention_2",  # Use optimized attention
            device_map="auto",
        ).to(device)
        qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    except Exception as e:
        logging.error(f"Error loading Qwen2 model and processor: {e}")
        print(f"Error loading Qwen2 model and processor: {e}")
        return
    
    # Load DeepDanbooru model and tags
    try:
        if args.verbose:
            logging.info(f"Loading DeepDanbooru model from {deepdanbooru_model_path} ...")
            print(f"Loading DeepDanbooru model from {deepdanbooru_model_path} ...")
        # Enable TensorFlow GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logging.error(f"Error setting TensorFlow GPU memory growth: {e}")
        deepdanbooru_model = tf.keras.models.load_model(deepdanbooru_model_path, compile=False)
        tags_path = deepdanbooru_model_path.replace('.h5', '.tags')  # Adjust if different
        if not os.path.isfile(tags_path):
            logging.error(f"Tags file not found at {tags_path}")
            print(f"Tags file not found at {tags_path}")
            return
        deepdanbooru_tags = dd.data.load_tags(tags_path)
        if args.verbose:
            logging.info(f"Loaded tags from {tags_path}")
            print(f"Loaded tags from {tags_path}")
    except Exception as e:
        logging.error(f"Error loading DeepDanbooru model or tags: {e}")
        print(f"Error loading DeepDanbooru model or tags: {e}")
        return
    
    # Get list of video folders
    try:
        vid_folders = [
            os.path.join(dataset_folder, d) for d in os.listdir(dataset_folder)
            if os.path.isdir(os.path.join(dataset_folder, d))
        ]
        vid_folders.sort()
    except Exception as e:
        logging.error(f"Error listing video folders in {dataset_folder}: {e}")
        print(f"Error listing video folders in {dataset_folder}: {e}")
        return
    
    # Determine starting index based on start_video if provided
    if start_video is not None:
        try:
            start_index = next(i for i, v in enumerate(vid_folders) if os.path.basename(v) == start_video)
        except StopIteration:
            logging.error(f"Start video folder '{start_video}' not found in dataset.")
            print(f"Start video folder '{start_video}' not found in dataset.")
            return
    
    # Adjust the list of video folders to process
    vid_folders = vid_folders[start_index:]
    if num_videos is not None:
        vid_folders = vid_folders[:num_videos]
    
    # Process videos in batches
    for i in range(0, len(vid_folders), batch_size):
        batch_vid_folders = vid_folders[i:i+batch_size]
        process_videos_in_batch(
            batch_vid_folders, 
            qwen_model, 
            qwen_processor, 
            device, 
            frame_step,
            deepdanbooru_model=deepdanbooru_model,
            deepdanbooru_tags=deepdanbooru_tags,
            threshold=0.5
        )

# Entry point
if __name__ == '__main__':
    main()