import os
import re
import argparse
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_frames(vid_folder, frame_step=1):
    # Get list of frame images
    frame_files = [f for f in os.listdir(vid_folder) if re.match(r'frame_\d+\.(jpg|jpeg|png)', f, re.IGNORECASE)]
    if not frame_files:
        print(f"No frames found in {vid_folder}")
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
            print(f"Error loading image {f} in {vid_folder}: {e}")
            continue
    if not frames:
        print(f"No valid frames to process in {vid_folder}")
        return None
    return frames

def process_videos_in_batch(batch_vid_folders, model, processor, device, frame_step=1):
    batch_frames = []
    batch_texts = []
    vid_folder_names = []
    for vid_folder in batch_vid_folders:
        frames = load_frames(vid_folder, frame_step)
        if frames is None:
            continue
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": """Describe the video.
                    The description should cover objects, entities, actions, movements, background, context, emotion/tone, transitions, camera movements and angles, and the speed and order of actions."""},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch_frames.append(frames)
        batch_texts.append(text)
        vid_folder_names.append(vid_folder)
    if not batch_frames:
        return
    # Prepare inputs
    inputs = processor(
        text=batch_texts,
        images=None,
        videos=batch_frames,
        padding=True,
        return_tensors="pt",
    ).to(device)
    # Inference
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    # Process outputs
    for idx, vid_folder in enumerate(vid_folder_names):
        in_ids = inputs.input_ids[idx]
        out_ids = generated_ids[idx]
        generated_ids_trimmed = out_ids[len(in_ids):]
        output_text = processor.decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # Write output to prompt.txt
        prompt_path = os.path.join(vid_folder, 'prompt.txt')
        with open(prompt_path, 'w') as f:
            f.write(output_text)
        # Print the output to console in a nice format
        print(f"\n--- Generated Caption for '{os.path.basename(vid_folder)}' ---")
        print(output_text)
        print("--- End of Caption ---\n")
        print(f"Processed {vid_folder}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process videos and caption them.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset folder.')
    parser.add_argument('--num_videos', type=int, default=None, help='Number of videos to process.')
    parser.add_argument('--start_index', type=int, default=0, help='Index of the video to start processing from.')
    parser.add_argument('--start_video', type=str, default=None, help='Name of the video folder to start processing from.')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of videos to process in a batch.')
    parser.add_argument('--frame_step', type=int, default=1, help='Step size for frame sampling.')
    args = parser.parse_args()
    dataset_folder = args.dataset
    num_videos = args.num_videos
    start_index = args.start_index
    start_video = args.start_video
    batch_size = args.batch_size
    frame_step = args.frame_step
    # Load the model and processor with optimizations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,  # Use FP16 for faster inference
        attn_implementation="flash_attention_2",  # Use optimized attention
        device_map="auto",
    ).to(device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    # Get list of video folders
    vid_folders = [
        os.path.join(dataset_folder, d) for d in os.listdir(dataset_folder)
        if os.path.isdir(os.path.join(dataset_folder, d))
    ]
    vid_folders.sort()
    # Determine starting index based on start_video if provided
    if start_video is not None:
        try:
            start_index = next(i for i, v in enumerate(vid_folders) if os.path.basename(v) == start_video)
        except StopIteration:
            print(f"Start video folder '{start_video}' not found in dataset.")
            return
    # Adjust the list of video folders to process
    vid_folders = vid_folders[start_index:]
    if num_videos is not None:
        vid_folders = vid_folders[:num_videos]
    # Process videos in batches
    for i in range(0, len(vid_folders), batch_size):
        batch_vid_folders = vid_folders[i:i+batch_size]
        process_videos_in_batch(batch_vid_folders, model, processor, device, frame_step)

if __name__ == '__main__':
    main()
