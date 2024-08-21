import os
import subprocess


def process_video(input_path, output_path, resolution, framerate):
    """Convert video to specified resolution, framerate, and format, applying HDR to SDR tonemapping."""
    command = [
        "ffmpeg",
        "-i", input_path,
        "-vf", (
            f"scale={resolution},"
            "zscale=transfer=linear,"
            "tonemap=hable:peak=8,"
            "zscale=transfer=bt709,"
            "format=yuv420p,"
            "colorspace=all=bt709"
        ),
        "-r", str(framerate),
        "-an",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        output_path,
    ]
    subprocess.run(command, stderr=subprocess.DEVNULL, check=True)


def create_datasets(original_dataset_path, new_dataset_720p60, new_dataset_1080p30):
    """Create two new datasets with 720p60 and 1080p30 versions of videos."""
    for class_name in os.listdir(original_dataset_path):
        if (class_name == '.DS_Store'):
            continue
        class_dir = os.path.join(original_dataset_path, class_name)
        if os.path.isdir(class_dir):
            videos = [f for f in os.listdir(class_dir) if f.endswith(".MOV")]
            total = len(videos)
            for idx, video_name in enumerate(sorted(videos), 1):
                video_path = os.path.join(class_dir, video_name)
                base_name = f"{class_name}_{idx:03d}.mp4"

                print(
                    f"Processing {
                        video_name} ({idx}/{total}) in class '{class_name}' - {total - idx} videos left"
                )

                # 720p60 dataset
                output_dir_720p60 = os.path.join(
                    new_dataset_720p60, class_name)
                os.makedirs(output_dir_720p60, exist_ok=True)
                output_path_720p60 = os.path.join(output_dir_720p60, base_name)
                process_video(video_path, output_path_720p60, "720x1280", 60)

                # 1080p30 dataset
                output_dir_1080p30 = os.path.join(
                    new_dataset_1080p30, class_name)
                os.makedirs(output_dir_1080p30, exist_ok=True)
                output_path_1080p30 = os.path.join(
                    output_dir_1080p30, base_name)
                process_video(video_path, output_path_1080p30, "1080x1920", 30)


# Paths to the datasets
original_dataset_path = "/Users/blurridge/Desktop/FSL"
new_dataset_720p60 = "/Users/blurridge/Desktop/FSL_720_SDR"
new_dataset_1080p30 = "/Users/blurridge/Desktop/FSL_1080_SDR"

# Create the new datasets
create_datasets(original_dataset_path, new_dataset_720p60, new_dataset_1080p30)
