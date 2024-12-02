import argparse
import yaml
import torch
import os
import cv2
import pickle
import numpy as np
from PIL import Image
from torchvision.io import read_video

from roadvision3d.src.engine.model_builder import build_model
from roadvision3d.src.engine.model_saver import load_checkpoint
from roadvision3d.tools.inference import inference
from roadvision3d.src.datasets.object_3d import Calibration, get_affine_transform
from roadvision3d.visualization import Visualizer


def video_to_frames_opencv(video_path, device='cuda:0'):
    """
    Reads video frames using OpenCV and returns them as a list of PyTorch tensors.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.tensor(frame_rgb, device=device)
        frames.append(frame_tensor)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames.")

    cap.release()
    print(f"Total frames processed: {frame_count}")
    return torch.stack(frames), fps

def video_to_frames_pytorch(video_path, device='cuda:0'):
    """
    Extracts frames from a video using TorchVision, returns them as a tensor on the specified device,
    and retrieves the video's frames per second (FPS).
    """
    # Load video using TorchVision
    print('reading video...')
    video_frames, _, info = read_video(video_path, pts_unit='sec')
    print('video readed')
    
    # Move frames to specified device
    video_frames = video_frames.to(device)
    
    # Extract FPS from the info dictionary
    fps = info['video_fps']
    
    print(f"Loaded video with {video_frames.shape[0]} frames at {fps} FPS.")
    
    return video_frames, fps

def process_frame(frame, cfg):
    raw_img = Image.fromarray(frame)
    img_size = np.array(raw_img.size)

    img_resolution = np.array(cfg['dataset']['resolution'])
    # statistics
    mean = np.array(cfg['dataset']['mean'], dtype=np.float32)
    std  = np.array(cfg['dataset']['std'], dtype=np.float32)

    center = np.array(img_size) / 2
    crop_size = img_size

    trans, trans_inv = get_affine_transform(center, crop_size, 0, img_resolution, inv=1)

    img_rescaled = raw_img.transform(tuple(img_resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)

    # image encoding
    img = np.array(img_rescaled).astype(np.float32) / 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # C * H * W

    img_tensor = torch.from_numpy(img)

    # Ensure the tensor is of the correct dtype (float32) and device (CPU or CUDA)
    img_tensor = img_tensor.float().unsqueeze(0)  # Ensures tensor is in float32, if not already

    coord_range = torch.from_numpy(np.array([center - crop_size / 2, center + crop_size / 2]).astype(np.float32)).float()

    return img_tensor, coord_range

def inference_video(video, calib, model, cfg, device='cuda:0'):
    """
    Perform inference on a video using the specified model and calibration data.
    """
    # Initialize list to store predictions
    predictions = []
    # Move model to specified device
    model = model.to(device)
    # Set model to evaluation mode
    model.eval()
    # Iterate over video frames
    for i in range(video.shape[0]):
        # Extract frame and move it to specified device
        frame = video[i]
        # Convert frame to NumPy
        frame_numpy = frame.cpu().numpy()
        img_tensor, coord = process_frame(frame_numpy, cfg)
        # Perform inference on the frame
        results = inference(img_tensor, calib, model, cfg, device, coord)
        # Append prediction to list
        predictions.append(results)
        print(f"Processed frame {i + 1}/{video.shape[0]}.")
    return predictions

def create_results_frames(frames, results, calib):
    visualizer = Visualizer(calib=calib, pitch=0.22)
    results_2d_video = []
    results_3d_video = []

    for frame, result in zip(frames, results):
        frame_numpy = frame.cpu().numpy()
        pil_img = Image.fromarray(frame_numpy)
        image_with_2d = visualizer.draw_2d_bboxes(pil_img, result, color='red', width=3, display=False)
        image_with_3d = visualizer.draw_3d_bboxes(pil_img, result, color='blue', color_front='green', width=4, display=False)
        results_2d_video.append(image_with_2d)
        results_3d_video.append(image_with_3d)

    return results_2d_video, results_3d_video

def frames_to_video_mp4(pil_images, output_path="output_video.mp4", fps=30):
    """
    Saves an array of PIL images directly to an MP4 video file without using temporary image files.
    """
    if len(pil_images) == 0:
        raise ValueError("The input list of images is empty.")

    # Get frame dimensions from the first image
    width, height = pil_images[0].size

    # Define the codec and create VideoWriter object for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each PIL image to the video
    for img in pil_images:
        # Convert PIL image to NumPy array
        frame_np = np.array(img)

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        out.write(frame_bgr)

    # Release the VideoWriter object
    out.release()
    print(f"Video saved to {output_path}")

def cargar_pkl(ruta):
    with open(ruta, 'rb') as archivo:
        return pickle.load(archivo)

def main():
    parser = argparse.ArgumentParser(description='Perform inference on a video using a trained model.')

    parser.add_argument('--config_file', type=str, default=None, help='Path to a configuration file (YAML or JSON) containing all parameters.')

    parser.add_argument('--video_path', type=str, help='Path to the input video file.')
    parser.add_argument('--cfg_path', type=str, help='Path to the configuration YAML file.')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the model checkpoint file.')
    parser.add_argument('--calib_path', type=str, help='Path to the calibration file (pickle format).')
    parser.add_argument('--device', type=str, help='Device to run the inference on (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--output_dir', type=str, help='Directory to save the output videos.')
    parser.add_argument('--fps', type=int, help='Frames per second for the output video.')

    args = parser.parse_args()

    # Load configuration from file if provided
    config_data = {}
    if args.config_file:
        if args.config_file.endswith('.yaml') or args.config_file.endswith('.yml'):
            with open(args.config_file, 'r') as f:
                config_data = yaml.load(f, Loader=yaml.Loader)
        elif args.config_file.endswith('.json'):
            import json
            with open(args.config_file, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError("Config file must be a .yaml, .yml, or .json file.")

    # Merge config data and command-line arguments
    # Command-line arguments override config file parameters
    arg_dict = vars(args)
    for key in ['video_path', 'cfg_path', 'checkpoint_path', 'calib_path', 'device', 'output_dir', 'fps']:
        if arg_dict.get(key) is None and key in config_data:
            arg_dict[key] = config_data[key]

    # Now check required arguments
    required_args = ['video_path', 'cfg_path', 'checkpoint_path', 'calib_path']
    for req_arg in required_args:
        if arg_dict.get(req_arg) is None:
            parser.error(f"Argument '{req_arg}' is required but not provided.")

    # Assign values back to args
    args.video_path = arg_dict['video_path']
    args.cfg_path = arg_dict['cfg_path']
    args.checkpoint_path = arg_dict['checkpoint_path']
    args.calib_path = arg_dict['calib_path']
    args.device = arg_dict.get('device', 'cuda:0')
    args.output_dir = arg_dict.get('output_dir', './output_videos')
    # args.fps = arg_dict.get('fps', 30)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model configuration
    cfg = yaml.load(open(args.cfg_path, 'r'), Loader=yaml.Loader)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Build the model
    model = build_model(cfg)

    # Load checkpoint
    load_checkpoint(model=model,
                    optimizer=None,
                    filename=args.checkpoint_path,
                    map_location=device)

    # Load calibration data
    calib_file = cargar_pkl(args.calib_path)
    calib_matrix = calib_file['calib_1']['camera_matrix']
    calib = Calibration.from_intrinsic_matrix(calib_matrix)

    # Read video frames
    # frames, fps = video_to_frames_pytorch(args.video_path, device=device)
    frames, fps = video_to_frames_opencv(args.video_path, device=device)

    # Perform inference
    results_video = inference_video(frames, calib, model, cfg, device=device)

    # Create result frames
    result_frames_2d, result_frames_3d = create_results_frames(frames, results_video, calib)

    # Save output videos
    output_video_path_2d = os.path.join(args.output_dir, 'video_2d.mp4')
    output_video_path_3d = os.path.join(args.output_dir, 'video_3d.mp4')

    frames_to_video_mp4(result_frames_2d, output_path=output_video_path_2d, fps=fps)
    frames_to_video_mp4(result_frames_3d, output_path=output_video_path_3d, fps=fps)

if __name__ == '__main__':
    main()
