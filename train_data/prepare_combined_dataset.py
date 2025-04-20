import os
import json
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image
import io
from sklearn.model_selection import train_test_split
import shutil
import yaml
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
BASE_DATA_DIR = ""
OUTPUT_DIR = ".."  # Output train/valid/test datasets in the workspace root
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1  # Ensure these sum to 1.0

# Define target class mapping (source_class_name -> target_class_index)
# We only care about helmets for this task.
CLASS_MAP = {
    'helmet': 0,
    'hardhat': 0,
    # Add other variations if found, mapping to 0
}
TARGET_CLASSES = ['helmet']  # Final class names for data.yaml

# Source Dataset Paths
HELMET_DETECTION_PATH = os.path.join(BASE_DATA_DIR, "HelmetDetection")
HARD_HAT_PATH = os.path.join(BASE_DATA_DIR, "hard-hat-detection", "data")
CONSTRUCTION_SAFETY_PATH = os.path.join(
    BASE_DATA_DIR, "construction-safety-gsnvb", "data"
)

# Output Directories
TRAIN_IMG_DIR = os.path.join(OUTPUT_DIR, "train_dataset", "images")
TRAIN_LBL_DIR = os.path.join(OUTPUT_DIR, "train_dataset", "labels")
VALID_IMG_DIR = os.path.join(OUTPUT_DIR, "valid_dataset", "images")
VALID_LBL_DIR = os.path.join(OUTPUT_DIR, "valid_dataset", "labels")
TEST_IMG_DIR = os.path.join(OUTPUT_DIR, "test_dataset", "images")
TEST_LBL_DIR = os.path.join(OUTPUT_DIR, "test_dataset", "labels")

# --- Helper Functions ---

def clear_or_create_dir(directory):
    """Removes directory if it exists, then creates it."""
    if os.path.exists(directory):
        logging.info(f"Removing existing directory: {directory}")
        shutil.rmtree(directory)
    logging.info(f"Creating directory: {directory}")
    os.makedirs(directory, exist_ok=True)

def convert_voc_to_yolo(xml_path, src_img_dir, target_lbl_dir):
    """Parses VOC XML, converts annotations, and saves YOLO txt file.

    Args:
        xml_path (str): Path to the VOC XML annotation file.
        src_img_dir (str): Directory containing the source images.
        target_lbl_dir (str): Directory to save the output YOLO label file.

    Returns:
        tuple: (image_filename, label_filename) or (None, None) if failed.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        if size is None:
            logging.warning(f"Skipping {xml_path}: <size> tag not found.")
            return None, None
        img_width_elem = size.find('width')
        img_height_elem = size.find('height')
        img_filename_elem = root.find('filename')

        if None in [img_width_elem, img_height_elem, img_filename_elem]:
            logging.warning(
                f"Skipping {xml_path}: Missing width, height, or filename tag."
            )
            return None, None

        img_width = int(img_width_elem.text)
        img_height = int(img_height_elem.text)
        img_filename = img_filename_elem.text

        # Sometimes image extension might be missing or wrong in XML
        base_img_filename = os.path.splitext(img_filename)[0]
        potential_img_paths = [
            os.path.join(src_img_dir, f"{base_img_filename}.jpg"),
            os.path.join(src_img_dir, f"{base_img_filename}.png"),
            os.path.join(src_img_dir, f"{base_img_filename}.jpeg"),
            os.path.join(src_img_dir, img_filename)  # Original name
        ]

        actual_img_path = None
        for p_path in potential_img_paths:
            if os.path.exists(p_path):
                actual_img_path = p_path
                img_filename = os.path.basename(actual_img_path)  # Use actual filename
                break

        if actual_img_path is None:
            logging.warning(
                f"Skipping {xml_path}: Corresponding image not found "
                f"for {base_img_filename} in {src_img_dir}"
            )
            return None, None

        # Check image dimensions if possible (optional, but good practice)
        # try:
        #     with Image.open(actual_img_path) as img:
        #         actual_width, actual_height = img.size
        #     if actual_width != img_width or actual_height != img_height:
        #         logging.warning(
        #               f"XML size mismatch for {img_filename}: "
        #               f"XML=({img_width},{img_height}), Img=({actual_width},{actual_height}). Using XML size."
        #         )
        # except Exception as e:  # noqa
        #     logging.warning(
        #          f"Could not verify image dimensions for {img_filename}: {e}"
        #     )

        yolo_lines = []
        obj_count = 0
        for obj in root.findall('object'):
            name = obj.find('name')
            if name is None:
                continue
            class_name = name.text.lower().strip()

            if class_name in CLASS_MAP:
                target_class_id = CLASS_MAP[class_name]
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue

                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # Clamp coordinates to image bounds
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_width, xmax)
                ymax = min(img_height, ymax)

                # Convert to YOLO format (center_x, center_y, width, height) normalized
                if img_width == 0 or img_height == 0:
                    logging.warning(
                        f"Skipping object in {xml_path} "
                        f"due to zero image dimension."
                    )
                    continue

                dw = 1.0 / img_width
                dh = 1.0 / img_height
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                width = xmax - xmin
                height = ymax - ymin

                x_center_norm = x_center * dw
                y_center_norm = y_center * dh
                width_norm = width * dw
                height_norm = height * dh

                # Clamp normalized values to [0.0, 1.0]
                x_center_norm = max(0.0, min(1.0, x_center_norm))
                y_center_norm = max(0.0, min(1.0, y_center_norm))
                width_norm = max(0.0, min(1.0, width_norm))
                height_norm = max(0.0, min(1.0, height_norm))

                yolo_lines.append(
                    f"{target_class_id} {x_center_norm:.6f} "
                    f"{y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                )
                obj_count += 1

        if yolo_lines:
            label_filename = f"{base_img_filename}.txt"
            label_path = os.path.join(target_lbl_dir, label_filename)
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_lines) + "\n")
            # logging.debug(f"Saved {obj_count} objects to {label_path}")
            return img_filename, label_filename
        else:
            # logging.debug(f"No target objects found in {xml_path}")
            return img_filename, None  # Return image name even if no labels

    except ET.ParseError as e:  # noqa
        logging.error(f"Failed to parse XML {xml_path}: {e}")
        return None, None
    except Exception as e:  # noqa
        logging.error(f"Error processing {xml_path}: {e}")
        return None, None

def convert_coco_json_to_yolo(
    json_path, src_img_dir, target_img_dir, target_lbl_dir
):
    """Parses COCO JSON, converts annotations, saves YOLO txt files, copies images.

    Args:
        json_path (str): Path to the COCO JSON annotation file.
        src_img_dir (str): Directory containing the source images for this split.
        target_img_dir (str): Directory to save the copied images.
        target_lbl_dir (str): Directory to save the output YOLO label files.

    Returns:
        int: Count of images successfully processed and copied.
    """
    processed_count = 0
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Create category map: coco_cat_id -> name
        coco_categories = {cat['id']: cat['name'].lower().strip() for cat in data['categories']}

        # Create target class map: coco_cat_id -> yolo_class_id
        target_category_map = {}
        for coco_id, coco_name in coco_categories.items():
            if coco_name in CLASS_MAP:
                target_category_map[coco_id] = CLASS_MAP[coco_name]

        if not target_category_map:
            logging.warning(f"No target classes found in COCO categories for {json_path}")
            return 0

        # Create image map: image_id -> image_info (filename, width, height)
        image_map = {img['id']: img for img in data['images']}

        # Group annotations by image_id
        annotations_by_image = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        # Process each image
        for img_id, annotations in tqdm(annotations_by_image.items(), desc=f"Processing {os.path.basename(json_path)}"):
            if img_id not in image_map:
                logging.warning(f"Image ID {img_id} found in annotations but not in images section of {json_path}")
                continue

            img_info = image_map[img_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']

            src_img_path = os.path.join(src_img_dir, img_filename)
            if not os.path.exists(src_img_path):
                logging.warning(f"Source image not found, skipping: {src_img_path}")
                continue

            yolo_lines = []
            for ann in annotations:
                category_id = ann['category_id']
                if category_id in target_category_map:
                    yolo_class_id = target_category_map[category_id]
                    bbox = ann['bbox']  # COCO format: [x_min, y_min, width, height]
                    x_min, y_min, width, height = bbox

                    if img_width == 0 or img_height == 0:
                        logging.warning(f"Skipping annotation in {img_filename} due to zero image dimension.")
                        continue

                    # Convert COCO bbox to YOLO format
                    x_center = x_min + width / 2.0
                    y_center = y_min + height / 2.0
                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    width_norm = width / img_width
                    height_norm = height / img_height

                    # Clamp normalized values to [0.0, 1.0]
                    x_center_norm = max(0.0, min(1.0, x_center_norm))
                    y_center_norm = max(0.0, min(1.0, y_center_norm))
                    width_norm = max(0.0, min(1.0, width_norm))
                    height_norm = max(0.0, min(1.0, height_norm))

                    yolo_lines.append(
                        f"{yolo_class_id} {x_center_norm:.6f} "
                        f"{y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                    )

            # Save YOLO label file (even if empty, common practice for YOLO)
            base_filename = os.path.splitext(img_filename)[0]
            label_filename = f"{base_filename}.txt"
            label_path = os.path.join(target_lbl_dir, label_filename)
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_lines) + "\n")

            # Copy image file
            target_img_path = os.path.join(target_img_dir, img_filename)
            shutil.copy2(src_img_path, target_img_path) # copy2 preserves metadata
            processed_count += 1

    except FileNotFoundError:
        logging.error(f"COCO JSON file not found: {json_path}")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON file: {json_path}")
    except KeyError as e:  # noqa
        logging.error(f"Missing expected key in COCO JSON {json_path}: {e}")
    except Exception as e:  # noqa
        logging.error(f"An unexpected error occurred processing {json_path}: {e}")

    logging.info(f"Finished processing {json_path}. Copied {processed_count} images/labels.")
    return processed_count

def convert_parquet_to_yolo(
    parquet_path, target_img_dir, target_lbl_dir, split_name
):
    """Reads Parquet, extracts images, converts annotations, saves images & YOLO txt.

    Assumes Parquet file structure based on construction-safety-gsnvb dataset_info.json:
    - 'image': image bytes
    - 'image_id': unique id for image
    - 'width', 'height': image dimensions
    - 'objects': list of dicts, each with:
        - 'category': int (index based on dataset features)
        - 'bbox': list [x_min, y_min, width, height]

    Args:
        parquet_path (str): Path to the input Parquet file.
        target_img_dir (str): Directory to save the extracted images.
        target_lbl_dir (str): Directory to save the output YOLO label files.
        split_name (str): Name of the dataset split (e.g., 'train').

    Returns:
        int: Count of images successfully processed and saved.
    """
    processed_count = 0
    # Assuming the category index for 'helmet' is 1 based on dataset_info.json
    # Features: ["construction-safety", "helmet", "no-helmet", ...]
    # -> helmet is index 1
    source_helmet_class_id = 1
    target_helmet_class_id = CLASS_MAP.get('helmet')  # Should be 0

    if target_helmet_class_id is None:
        logging.error("Target class 'helmet' not found in CLASS_MAP.")
        return 0

    try:
        df = pd.read_parquet(parquet_path)
        logging.info(f"Read {len(df)} records from {parquet_path}")

        required_cols = ['image', 'image_id', 'width', 'height', 'objects']
        if not all(col in df.columns for col in required_cols):
            logging.error(
                f"Missing required columns in {parquet_path}. Need: {required_cols}"
            )
            return 0

        # Add tqdm progress bar
        for index, row in tqdm(df.iterrows(), total=df.shape[0],
                             desc=f"Processing {os.path.basename(parquet_path)}"):
            try:
                image_id = row['image_id']
                img_width = row['width']
                img_height = row['height']
                objects = row['objects']
                image_bytes = row['image']['bytes'] # Access bytes within the image dict

                if not image_bytes or img_width <= 0 or img_height <= 0:
                    logging.warning(
                        f"Skipping record index {index} (ID: {image_id}) due to "
                        f"invalid image data or dimensions."
                    )
                    continue

                # Determine image format and save image
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    img_format = img.format.lower() if img.format else 'jpg' # Default to jpg
                    if img_format not in ['jpeg', 'jpg', 'png']:
                         logging.warning(
                             f"Unsupported image format '{img_format}' for ID {image_id}. "
                             f"Attempting to save as PNG."
                         )
                         img_format = 'png'

                    # Use image_id for filename uniqueness across datasets
                    img_filename = f"cs_{image_id}.{img_format}"
                    target_img_path = os.path.join(target_img_dir, img_filename)
                    img.save(target_img_path)
                except Exception as img_err:
                    logging.error(f"Failed to process or save image for ID {image_id}: {img_err}")
                    continue

                # Process annotations
                yolo_lines = []
                if isinstance(objects, dict) and 'bbox' in objects and 'category' in objects:
                    # Handle case where 'objects' might be a dict with lists
                    num_annotations = len(objects.get('category', []))
                    for i in range(num_annotations):
                        try:
                            category_id = objects['category'][i]
                            bbox = objects['bbox'][i]

                            if category_id == source_helmet_class_id:
                                x_min, y_min, width, height = bbox

                                # Convert COCO bbox to YOLO format
                                x_center = x_min + width / 2.0
                                y_center = y_min + height / 2.0
                                x_center_norm = x_center / img_width
                                y_center_norm = y_center / img_height
                                width_norm = width / img_width
                                height_norm = height / img_height

                                # Clamp normalized values to [0.0, 1.0]
                                x_center_norm = max(0.0, min(1.0, x_center_norm))
                                y_center_norm = max(0.0, min(1.0, y_center_norm))
                                width_norm = max(0.0, min(1.0, width_norm))
                                height_norm = max(0.0, min(1.0, height_norm))

                                yolo_lines.append(
                                    f"{target_helmet_class_id} {x_center_norm:.6f} "
                                    f"{y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                                )
                        except IndexError:
                             logging.warning(
                                 f"Index error processing annotation {i} for "
                                 f"image ID {image_id} in {parquet_path}"
                             )
                        except Exception as ann_err:
                            logging.warning(
                                f"Error processing annotation {i} for "
                                f"image ID {image_id} in {parquet_path}: {ann_err}"
                            )
                else:
                    logging.warning(
                        f"Unexpected format for 'objects' column for image ID {image_id} "
                        f"in {parquet_path}. Expected dict of lists."
                    )

                # Save YOLO label file
                base_filename = os.path.splitext(img_filename)[0]
                label_filename = f"{base_filename}.txt"
                label_path = os.path.join(target_lbl_dir, label_filename)
                with open(label_path, 'w') as f:
                    f.write("\n".join(yolo_lines) + "\n")

                processed_count += 1

            except KeyError as key_err:
                logging.warning(f"Skipping record index {index} due to missing key: {key_err}")
            except Exception as row_err:
                logging.warning(f"Skipping record index {index} due to error: {row_err}")

    except ImportError:
        logging.error(
            "Failed to process Parquet: pandas or pyarrow not installed. "
            "Try `pip install pandas pyarrow`"
        )
    except FileNotFoundError:
        logging.error(f"Parquet file not found: {parquet_path}")
    except Exception as e:  # noqa
        logging.error(f"An unexpected error occurred processing {parquet_path}: {e}")

    logging.info(f"Finished processing {parquet_path}. Saved {processed_count} images/labels.")
    return processed_count

def split_and_copy_files(
    file_pairs, train_img_dir, train_lbl_dir,
    valid_img_dir, valid_lbl_dir, test_img_dir, test_lbl_dir
):
    """Splits files into train/valid/test sets and copies them to final directories.

    Args:
        file_pairs (list): List of tuples, where each tuple is (src_image_path, src_label_path).
        train_img_dir (str): Target directory for training images.
        train_lbl_dir (str): Target directory for training labels.
        valid_img_dir (str): Target directory for validation images.
        valid_lbl_dir (str): Target directory for validation labels.
        test_img_dir (str): Target directory for testing images.
        test_lbl_dir (str): Target directory for testing labels.
    """
    if not file_pairs:
        logging.warning("No file pairs provided for splitting.")
        return

    # Ensure splits add up to 1.0 for train_test_split
    if not (0.999 < (TRAIN_SPLIT + VALID_SPLIT + TEST_SPLIT) < 1.001):
        logging.error(
            f"Train/Valid/Test splits ({TRAIN_SPLIT}/{VALID_SPLIT}/{TEST_SPLIT}) "
            f"do not sum to 1.0. Aborting split."
        )
        return

    # Split into train and remaining (valid + test)
    train_files, remaining_files = train_test_split(
        file_pairs,
        train_size=TRAIN_SPLIT,
        random_state=42  # for reproducibility
    )

    # Calculate split ratio for valid vs test from the remainder
    if len(remaining_files) > 0:
        relative_test_split = TEST_SPLIT / (VALID_SPLIT + TEST_SPLIT)
        if relative_test_split >= 1.0 or relative_test_split <= 0.0:
            # Handle cases where valid or test split is 0
             if VALID_SPLIT == 0:
                 valid_files = []
                 test_files = remaining_files
             elif TEST_SPLIT == 0:
                 valid_files = remaining_files
                 test_files = []
             else:
                 # Should not happen with the sum check, but as a safeguard
                 logging.error("Invalid relative split calculation.")
                 valid_files, test_files = [], []
        else:
            valid_files, test_files = train_test_split(
                remaining_files,
                test_size=relative_test_split,
                random_state=42  # use same random state
            )
    else:
        valid_files, test_files = [], []

    logging.info(
        f"Splitting into: Train={len(train_files)}, "
        f"Valid={len(valid_files)}, Test={len(test_files)}"
    )

    # Function to copy a list of file pairs to target directories
    def copy_files(files, target_img_dir, target_lbl_dir, split_name):
        copied_count = 0
        logging.info(f"Copying {len(files)} files to {split_name}...")
        for img_path, lbl_path in tqdm(files, desc=f"Copying {split_name}"):
            try:
                if not os.path.exists(img_path):
                    logging.warning(f"Source image missing during copy: {img_path}")
                    continue
                if not os.path.exists(lbl_path):
                    logging.warning(f"Source label missing during copy: {lbl_path}")
                    continue

                img_filename = os.path.basename(img_path)
                lbl_filename = os.path.basename(lbl_path)

                # Add a prefix to avoid potential collisions if filenames are
                # identical across source datasets. Although HelmetDetection
                # seems to have unique names, it's safer.
                prefixed_img_filename = f"hd_{img_filename}"
                prefixed_lbl_filename = f"hd_{lbl_filename}"

                target_img_path = os.path.join(target_img_dir, prefixed_img_filename)
                target_lbl_path = os.path.join(target_lbl_dir, prefixed_lbl_filename)

                shutil.copy2(img_path, target_img_path)
                shutil.copy2(lbl_path, target_lbl_path)
                copied_count += 1
            except Exception as e:  # noqa
                logging.error(f"Error copying {img_path} or {lbl_path}: {e}")
        return copied_count

    # Copy files for each split
    copy_files(train_files, train_img_dir, train_lbl_dir, "train")
    copy_files(valid_files, valid_img_dir, valid_lbl_dir, "valid")
    copy_files(test_files, test_img_dir, test_lbl_dir, "test")

# --- Main Processing Logic ---

def main():
    logging.info("Starting dataset preparation...")

    # 1. Create output directories
    clear_or_create_dir(TRAIN_IMG_DIR)
    clear_or_create_dir(TRAIN_LBL_DIR)
    clear_or_create_dir(VALID_IMG_DIR)
    clear_or_create_dir(VALID_LBL_DIR)
    clear_or_create_dir(TEST_IMG_DIR)
    clear_or_create_dir(TEST_LBL_DIR)

    # List to hold (image_path, label_path) tuples for splitting HelmetDetection
    helmet_detection_files = []
    processed_files_count = 0

    # 2. Process HelmetDetection (VOC XML) -> Needs splitting
    logging.info("Processing HelmetDetection dataset...")
    hd_xml_dir = os.path.join(HELMET_DETECTION_PATH, "annotations")
    hd_img_dir = os.path.join(HELMET_DETECTION_PATH, "images")
    # Create a temporary directory for labels before splitting
    temp_lbl_dir = os.path.join(OUTPUT_DIR, "temp_helmet_detection_labels")
    clear_or_create_dir(temp_lbl_dir)

    if os.path.exists(hd_xml_dir) and os.path.exists(hd_img_dir):
        xml_files = [f for f in os.listdir(hd_xml_dir) if f.endswith(".xml")]
        logging.info(f"Found {len(xml_files)} XML files in {hd_xml_dir}")
        for xml_file in tqdm(xml_files, desc="Processing HelmetDetection"):
            xml_path = os.path.join(hd_xml_dir, xml_file)
            img_filename, lbl_filename = convert_voc_to_yolo(
                xml_path, hd_img_dir, temp_lbl_dir
            )
            if img_filename and lbl_filename:
                # Use the actual source image path and temp label path for splitting
                # Need to reconstruct the source img path based on the returned filename
                base_img_filename = os.path.splitext(img_filename)[0]
                potential_img_paths = [
                    os.path.join(hd_img_dir, f"{base_img_filename}.jpg"),
                    os.path.join(hd_img_dir, f"{base_img_filename}.png"),
                    os.path.join(hd_img_dir, f"{base_img_filename}.jpeg"),
                ]
                actual_img_path = None
                for p_path in potential_img_paths:
                    if os.path.exists(p_path):
                        actual_img_path = p_path
                        break

                if actual_img_path:
                    lbl_path = os.path.join(temp_lbl_dir, lbl_filename)
                    helmet_detection_files.append((actual_img_path, lbl_path))
                    # Don't increment processed_files_count here yet, count after splitting
                else:
                    logging.warning(
                        f"Could not find source image for converted label {lbl_filename}"
                    )
            elif img_filename and not lbl_filename:
                # Image processed, but no relevant labels found
                pass # Don't add to files for splitting
            else:
                logging.warning(f"Skipped processing for XML: {xml_file}")
    else:
        logging.warning(f"HelmetDetection source directories not found: {hd_xml_dir} or {hd_img_dir}")

    # 3. Process hard-hat-detection (COCO JSON) -> Uses pre-split
    logging.info("Processing hard-hat-detection dataset...")
    for split in ["train", "valid", "test"]:
        json_file = os.path.join(
            HARD_HAT_PATH, split, "_annotations.coco.json"
        )
        img_dir_src = os.path.join(HARD_HAT_PATH, split)
        if os.path.exists(json_file) and os.path.exists(img_dir_src):
            # Determine output dirs based on split
            if split == "train":
                img_out_dir, lbl_out_dir = TRAIN_IMG_DIR, TRAIN_LBL_DIR
            elif split == "valid":
                img_out_dir, lbl_out_dir = VALID_IMG_DIR, VALID_LBL_DIR
            else:  # test
                img_out_dir, lbl_out_dir = TEST_IMG_DIR, TEST_LBL_DIR

            count = convert_coco_json_to_yolo(
                json_file, img_dir_src, img_out_dir, lbl_out_dir
            )
            processed_files_count += count
        else:
            logging.warning(
                f"COCO JSON file or image dir not found for split '{split}': {json_file} or {img_dir_src}"
            )

    # 4. Process construction-safety-gsnvb (Parquet) -> Uses pre-split
    logging.info("Processing construction-safety-gsnvb dataset...")
    if os.path.exists(CONSTRUCTION_SAFETY_PATH):
        for filename in os.listdir(CONSTRUCTION_SAFETY_PATH):
            if filename.endswith(".parquet"):
                parquet_file = os.path.join(CONSTRUCTION_SAFETY_PATH, filename)
                split_name = None
                img_out_dir, lbl_out_dir = None, None

                if "train" in filename:
                    split_name = "train"
                    img_out_dir, lbl_out_dir = TRAIN_IMG_DIR, TRAIN_LBL_DIR
                elif "validation" in filename:
                    split_name = "valid"
                    img_out_dir, lbl_out_dir = VALID_IMG_DIR, VALID_LBL_DIR
                elif "test" in filename:
                    split_name = "test"
                    img_out_dir, lbl_out_dir = TEST_IMG_DIR, TEST_LBL_DIR

                if split_name:
                    count = convert_parquet_to_yolo(
                        parquet_file, img_out_dir, lbl_out_dir, split_name
                    )
                    processed_files_count += count
                else:
                    logging.warning(
                        f"Could not determine split for Parquet file: {filename}"
                    )
    else:
        logging.warning(f"Construction Safety path not found: {CONSTRUCTION_SAFETY_PATH}")

    # 5. Split HelmetDetection files and copy all
    logging.info("Splitting and copying HelmetDetection files...")
    split_and_copy_files(
        helmet_detection_files, TRAIN_IMG_DIR, TRAIN_LBL_DIR,
        VALID_IMG_DIR, VALID_LBL_DIR, TEST_IMG_DIR, TEST_LBL_DIR
    )
    # We don't know exactly how many were copied successfully by split_and_copy
    # but we can add the number intended for splitting
    processed_files_count += len(helmet_detection_files)

    # Cleanup temp dir
    if os.path.exists(temp_lbl_dir):
        logging.info(f"Removing temporary directory: {temp_lbl_dir}")
        shutil.rmtree(temp_lbl_dir)

    # 6. Create data.yaml
    logging.info("Creating data.yaml file...")
    data_yaml_content = {
        'train': os.path.relpath(TRAIN_IMG_DIR, OUTPUT_DIR),
        'val': os.path.relpath(VALID_IMG_DIR, OUTPUT_DIR),
        'test': os.path.relpath(TEST_IMG_DIR, OUTPUT_DIR),
        'nc': len(TARGET_CLASSES),
        'names': TARGET_CLASSES
    }
    data_yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    try:
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f, default_flow_style=False,
                      sort_keys=False)
        logging.info(f"Successfully created {data_yaml_path}")
    except Exception as e:  # noqa
        logging.error(f"Failed to create data.yaml: {e}")

    logging.info(
        f"Dataset preparation finished. Processed approximately "
        f"{processed_files_count} files."  # Update this count
    )

if __name__ == "__main__":
    # Check for dependencies
    try:
        import pandas
        import PIL
        import pyarrow
        import sklearn
        # yaml and tqdm checked by top-level imports
    except ImportError as e:  # noqa
        logging.error(
            f"Missing required package: {e}. Please install dependencies."
        )
        exit(1)

    main()