import os
import random
import shutil
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm  # To display progress
import concurrent.futures


def encode_labels(labels):
    unique_labels = sorted(set(labels))
    label_dict = {label: idx for idx, label in enumerate(unique_labels)}
    return label_dict


def apply_augmentation(video_path, save_path, frame_size=None):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_width = original_width
    frame_height = original_height
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

    flip = random.random() > 0.5
    angle = random.randint(-15, 15)
    while angle == 0:
        angle = random.randint(-15, 15)

    max_translate = min(frame_width, frame_height) * 0.1
    tx = random.randint(-int(max_translate), int(max_translate))
    ty = random.randint(-int(max_translate), int(max_translate))
    while tx == 0 and ty == 0:
        tx = random.randint(-int(max_translate), int(max_translate))
        ty = random.randint(-int(max_translate), int(max_translate))

    M_rotate = cv2.getRotationMatrix2D((frame_width / 2, frame_height / 2), angle, 1)
    T_translate = np.float32([[1, 0, tx], [0, 1, ty]])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if flip:
            frame = cv2.flip(frame, 1)

        frame = cv2.warpAffine(frame, M_rotate, (frame_width, frame_height))
        frame = cv2.warpAffine(frame, T_translate, (frame_width, frame_height))

        frame = cv2.resize(frame, (frame_width, frame_height))

        out.write(frame)

    cap.release()
    out.release()


def process_video(video_info):
    (class_name, src_1080, src_720, dst_1080, dst_720, augment_per_video, remainder) = (
        video_info
    )
    data_1080 = []
    data_720 = []

    # Ensure the destination directories exist
    os.makedirs(os.path.dirname(dst_1080), exist_ok=True)
    os.makedirs(os.path.dirname(dst_720), exist_ok=True)

    shutil.copy(src_1080, dst_1080)
    shutil.copy(src_720, dst_720)
    data_1080.append([dst_1080, class_name, "train"])
    data_720.append([dst_720, class_name, "train"])

    for aug_idx in range(augment_per_video + (1 if remainder else 0)):
        augmented_dst_1080 = dst_1080.replace(".mp4", f"_aug{aug_idx}.mp4")
        augmented_dst_720 = dst_720.replace(".mp4", f"_aug{aug_idx}.mp4")
        apply_augmentation(src_1080, augmented_dst_1080)
        apply_augmentation(src_720, augmented_dst_720, frame_size=(1280, 720))
        data_1080.append([augmented_dst_1080, class_name, "train"])
        data_720.append([augmented_dst_720, class_name, "train"])

    return data_1080, data_720


def split_dataset(
    root_dir_1080,
    root_dir_720,
    train_dir_1080,
    val_dir_1080,
    test_dir_1080,
    train_csv_1080,
    val_csv_1080,
    test_csv_1080,
    train_dir_720,
    val_dir_720,
    test_dir_720,
    train_csv_720,
    val_csv_720,
    test_csv_720,
    target_videos_per_class=1000,
    test_split=0.2,
    val_split=0.2,
    random_seed=42,
    apply_aug=True,
):
    data_1080 = []
    data_720 = []
    all_labels = []

    random.seed(random_seed)

    for class_name in os.listdir(root_dir_1080):
        if class_name in ["train", "test", "val"]:
            continue
        class_path_1080 = os.path.join(root_dir_1080, class_name)
        class_path_720 = os.path.join(root_dir_720, class_name)
        if not os.path.isdir(class_path_1080) or not os.path.isdir(class_path_720):
            continue

        videos = [
            f
            for f in os.listdir(class_path_1080)
            if os.path.isfile(os.path.join(class_path_1080, f))
            and not f.startswith("._")
        ]
        random.shuffle(videos)

        all_labels.append(class_name)

        total_videos = len(videos)
        if total_videos == 0:
            continue

        test_size = int(total_videos * test_split)
        val_size = int((total_videos - test_size) * val_split)

        test_videos = videos[:test_size]
        val_videos = videos[test_size : test_size + val_size]
        train_videos = videos[test_size + val_size :]

        current_train_count = len(train_videos)
        augmentations_needed = max(0, target_videos_per_class - current_train_count)
        augment_per_video = augmentations_needed // current_train_count
        remainder = augmentations_needed % current_train_count

        video_info_list = [
            (
                class_name,
                os.path.join(class_path_1080, video),
                os.path.join(class_path_720, video),
                os.path.join(train_dir_1080, class_name, video),
                os.path.join(train_dir_720, class_name, video),
                augment_per_video,
                idx < remainder,
            )
            for idx, video in enumerate(train_videos)
        ]

        print(
            f"\nProcessing class '{class_name}' ({current_train_count}/{target_videos_per_class})"
        )
        with concurrent.futures.ProcessPoolExecutor() as executor:
            with tqdm(
                total=len(video_info_list), desc=f"Augmenting videos for {class_name}"
            ) as pbar:
                futures = [
                    executor.submit(process_video, video_info)
                    for video_info in video_info_list
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    augmented_data_1080, augmented_data_720 = result
                    data_1080.extend(augmented_data_1080)
                    data_720.extend(augmented_data_720)
                    pbar.update(1)  # Manually update the progress bar

        for video in val_videos:
            src_1080 = os.path.join(class_path_1080, video)
            src_720 = os.path.join(class_path_720, video)
            dst_1080 = os.path.join(val_dir_1080, class_name, video)
            dst_720 = os.path.join(val_dir_720, class_name, video)
            os.makedirs(os.path.join(val_dir_1080, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir_720, class_name), exist_ok=True)

            shutil.copy(src_1080, dst_1080)
            shutil.copy(src_720, dst_720)
            data_1080.append([dst_1080, class_name, "val"])
            data_720.append([dst_720, class_name, "val"])

        for video in test_videos:
            src_1080 = os.path.join(class_path_1080, video)
            src_720 = os.path.join(class_path_720, video)
            dst_1080 = os.path.join(test_dir_1080, class_name, video)
            dst_720 = os.path.join(test_dir_720, class_name, video)
            os.makedirs(os.path.join(test_dir_1080, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir_720, class_name), exist_ok=True)
            shutil.copy(src_1080, dst_1080)
            shutil.copy(src_720, dst_720)
            data_1080.append([dst_1080, class_name, "test"])
            data_720.append([dst_720, class_name, "test"])

    label_dict = encode_labels(all_labels)

    for i in range(len(data_1080)):
        data_1080[i][1] = label_dict[data_1080[i][1]]
    for i in range(len(data_720)):
        data_720[i][1] = label_dict[data_720[i][1]]

    df_1080 = pd.DataFrame(data_1080, columns=["path", "label", "split"])
    df_720 = pd.DataFrame(data_720, columns=["path", "label", "split"])

    df_1080[df_1080["split"] == "train"][["path", "label"]].to_csv(
        train_csv_1080, index=False
    )
    df_1080[df_1080["split"] == "val"][["path", "label"]].to_csv(
        val_csv_1080, index=False
    )
    df_1080[df_1080["split"] == "test"][["path", "label"]].to_csv(
        test_csv_1080, index=False
    )

    df_720[df_720["split"] == "train"][["path", "label"]].to_csv(
        train_csv_720, index=False
    )
    df_720[df_720["split"] == "val"][["path", "label"]].to_csv(val_csv_720, index=False)
    df_720[df_720["split"] == "test"][["path", "label"]].to_csv(
        test_csv_720, index=False
    )

    return label_dict


# Define your paths for both FSL_1080 and FSL_720 datasets
fmt_1080 = "FSL_1080"
fmt_720 = "FSL_720"
root = "/Users/blurridge/Desktop"
root_dir_1080 = f"{root}/{fmt_1080}_SDR"
root_dir_720 = f"{root}/{fmt_720}_SDR"

train_dir_1080 = f"{root}/{fmt_1080}/train"
val_dir_1080 = f"{root}/{fmt_1080}/val"
test_dir_1080 = f"{root}/{fmt_1080}/test"
train_csv_1080 = f"{root}/{fmt_1080}/train.csv"
val_csv_1080 = f"{root}/{fmt_1080}/val.csv"
test_csv_1080 = f"{root}/{fmt_1080}/test.csv"

train_dir_720 = f"{root}/{fmt_720}/train"
val_dir_720 = f"{root}/{fmt_720}/val"
test_dir_720 = f"{root}/{fmt_720}/test"
train_csv_720 = f"{root}/{fmt_720}/train.csv"
val_csv_720 = f"{root}/{fmt_720}/val.csv"
test_csv_720 = f"{root}/{fmt_720}/test.csv"

if __name__ == "__main__":
    # Run the split with augmentation for both FSL_1080 and FSL_720 datasets
    label_dict = split_dataset(
        root_dir_1080,
        root_dir_720,
        train_dir_1080,
        val_dir_1080,
        test_dir_1080,
        train_csv_1080,
        val_csv_1080,
        test_csv_1080,
        train_dir_720,
        val_dir_720,
        test_dir_720,
        train_csv_720,
        val_csv_720,
        test_csv_720,
        target_videos_per_class=1050,
        random_seed=42,
        apply_aug=True,
    )
    # Save the label dictionary to a CSV or JSON file for future reference
    label_dict_df_1080 = pd.DataFrame(
        list(label_dict.items()), columns=["Label", "Encoded"]
    )
    label_dict_df_1080.to_csv(f"data/{fmt_1080}/label_dict.csv", index=False)

    label_dict_df_720 = pd.DataFrame(
        list(label_dict.items()), columns=["Label", "Encoded"]
    )
    label_dict_df_720.to_csv(f"data/{fmt_720}/label_dict.csv", index=False)
