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

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
# Base directory where individual dataset folders are located
BASE_DATA_DIR = "train_data"
# Output directory for the combined YOLO dataset
OUTPUT_DIR = "dataset"
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1  # Ensure these sum to 1.0

# Define target class mapping (source_class_name/id -> target_class_index)
# Map to just two classes: hat (0) and helmet (1)
CLASS_MAP = {
    # HelmetDetection (VOC) - XML uses 'n' tag with values 'helmet' and 'head'
    'head': 0,     # Head maps to hat (class 0)
    'helmet': 1,   # Helmet maps to helmet (class 1)
    
    # hard-hat-detection (COCO) - hardhat=0, no-hardhat=1
    'hardhat': 1,   # Hardhat maps to helmet (class 1)
    'no-hardhat': 0,  # No-hardhat maps to hat (class 0)
    'hh_0': 1,      # hardhat id in hard-hat-detection maps to helmet
    'hh_1': 0,      # no-hardhat id in hard-hat-detection maps to hat
    
    # construction-safety-gsnvb (Parquet) - helmet=1, no-helmet=2
    'cs_1': 1,      # Helmet ID (1) maps to helmet (class 1)
    'cs_2': 0,      # No-helmet ID (2) maps to hat (class 0)
}
TARGET_CLASSES = ['hat', 'helmet']  # Final class names for data.yaml

# Source Dataset Paths
HELMET_DETECTION_PATH = os.path.join(BASE_DATA_DIR, "HelmetDetection")
HARD_HAT_PATH = os.path.join(BASE_DATA_DIR, "hard-hat-detection", "data")
CONSTRUCTION_SAFETY_PATH = os.path.join(
    BASE_DATA_DIR, "construction-safety-gsnvb", "data"
)

# Output Directories (YOLO structure)
TRAIN_IMG_DIR = os.path.join(OUTPUT_DIR, "images", "train")
TRAIN_LBL_DIR = os.path.join(OUTPUT_DIR, "labels", "train")
VALID_IMG_DIR = os.path.join(OUTPUT_DIR, "images", "valid")
VALID_LBL_DIR = os.path.join(OUTPUT_DIR, "labels", "valid")
TEST_IMG_DIR = os.path.join(OUTPUT_DIR, "images", "test")
TEST_LBL_DIR = os.path.join(OUTPUT_DIR, "labels", "test")

# --- Helper Functions ---

def clear_or_create_dir(directory):
    """Removes directory if it exists, then creates it."""
    if os.path.exists(directory):
        logging.info(f"Removing existing directory: {directory}")
        shutil.rmtree(directory)
    logging.info(f"Creating directory: {directory}")
    os.makedirs(directory, exist_ok=True)

def convert_voc_to_yolo(xml_path, src_img_dir, temp_lbl_dir):
    """Parses VOC XML, converts annotations, and saves YOLO txt file.

    Args:
        xml_path (str): Path to the VOC XML annotation file.
        src_img_dir (str): Directory containing the source images.
        temp_lbl_dir (str): Temp directory to save the output YOLO label file.

    Returns:
        tuple: (image_filename, label_filename) or (None, None) if failed.
               Label filename is the base name without prefix.
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
        img_filename_original = img_filename_elem.text

        # Find the actual image file, handling potential extension issues
        base_img_filename = os.path.splitext(img_filename_original)[0]
        potential_img_paths = [
            os.path.join(src_img_dir, f"{base_img_filename}.jpg"),
            os.path.join(src_img_dir, f"{base_img_filename}.png"),
            os.path.join(src_img_dir, f"{base_img_filename}.jpeg"),
            os.path.join(src_img_dir, img_filename_original)  # Original name
        ]

        actual_img_path = None
        actual_img_filename = None
        for p_path in potential_img_paths:
            if os.path.exists(p_path):
                actual_img_path = p_path
                actual_img_filename = os.path.basename(actual_img_path)
                break

        if actual_img_path is None:
            logging.warning(
                f"Skipping {xml_path}: Corresponding image not found "
                f"for {base_img_filename} in {src_img_dir}"
            )
            return None, None

        # Verify image dimensions (optional but recommended)
        try:
            with Image.open(actual_img_path) as img:
                actual_width, actual_height = img.size
            if actual_width != img_width or actual_height != img_height:
                logging.warning(
                    f"XML size mismatch for {actual_img_filename}: "
                    f"XML=({img_width},{img_height}), "
                    f"Img=({actual_width},{actual_height}). "
                    f"Using XML size."
                )
        except Exception as e:
            logging.warning(
                f"Could not verify image dimensions for "
                f"{actual_img_filename}: {e}"
            )

        yolo_lines = []
        obj_count = 0
        for obj in root.findall('object'):
            # Check for both 'name' and 'n' tags as the HelmetDetection dataset uses 'n'
            name_elem = obj.find('name')
            if name_elem is None:
                name_elem = obj.find('n')  # Try alternative tag
            
            if name_elem is None:
                continue
            
            class_name = name_elem.text.lower().strip()

            # Check if the class is one we want to map
            if class_name in CLASS_MAP:
                target_class_id = CLASS_MAP[class_name]
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue

                # Use original XML bounds
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # Clamp coordinates to image bounds (using XML dimensions)
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_width, xmax)
                ymax = min(img_height, ymax)

                # Prevent division by zero and invalid boxes
                if (img_width <= 0 or img_height <= 0 or
                        xmax <= xmin or ymax <= ymin):
                    logging.warning(
                        f"Skipping object in {xml_path} due to zero image "
                        f"dimension or invalid box coordinates."
                    )
                    continue

                # Convert to YOLO format (center_x, center_y, w, h) normalized
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

        # Save label file to temporary location IF annotations were found
        if yolo_lines:
            # Use the base name from the image file found
            base_label_filename = os.path.splitext(actual_img_filename)[0]
            label_filename = f"{base_label_filename}.txt"
            temp_label_path = os.path.join(temp_lbl_dir, label_filename)
            with open(temp_label_path, 'w') as f:
                f.write("\n".join(yolo_lines) + "\n")
            # Return the actual image filename and the base label filename
            return actual_img_filename, label_filename
        else:
            # Return image name even if no target labels found
            return actual_img_filename, None

    except ET.ParseError as e:
        logging.error(f"Failed to parse XML {xml_path}: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error processing {xml_path}: {e}", exc_info=True)
        return None, None

def convert_coco_json_to_yolo(
    json_path, src_img_dir, target_img_dir, target_lbl_dir
):
    """Parses COCO JSON, saves YOLO labels & copies images to final dirs.

    Converts relevant annotations (mapped via CLASS_MAP) and copies
    corresponding images to the final target directories, applying the
    'hh_' prefix.

    Args:
        json_path (str): Path to the COCO JSON annotation file.
        src_img_dir (str): Directory containing the source images.
        target_img_dir (str): Final directory to save the copied images.
        target_lbl_dir (str): Final directory to save YOLO label files.

    Returns:
        int: Count of images successfully processed and copied.
    """
    processed_count = 0
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Create category map: coco_cat_id -> name
        coco_categories = {
            cat['id']: cat['name'].lower().strip()
            for cat in data.get('categories', [])
        }

        # Create mapping from COCO category ID to target YOLO class ID
        target_category_map = {}
        ignored_categories = set()
        for coco_id, coco_name in coco_categories.items():
            # Try direct mapping by name first
            if coco_name in CLASS_MAP:
                target_yolo_id = CLASS_MAP[coco_name]
                target_category_map[coco_id] = target_yolo_id
                logging.debug(
                    f"Mapping COCO category '{coco_name}' (ID: {coco_id}) "
                    f"to YOLO ID: {target_yolo_id}"
                )
            # Try mapping by prefixed ID for hard-hat-detection
            elif f'hh_{coco_id}' in CLASS_MAP:
                target_yolo_id = CLASS_MAP[f'hh_{coco_id}']
                target_category_map[coco_id] = target_yolo_id
                logging.debug(
                    f"Mapping COCO category ID {coco_id} ('{coco_name}') "
                    f"to YOLO ID: {target_yolo_id}"
                )
            elif coco_name not in ignored_categories:
                # Log ignored categories only once per file
                logging.info(  # Changed from warning to info as it's expected
                    f"Ignoring category '{coco_name}' (ID: {coco_id}) in "
                    f"{os.path.basename(json_path)} as it's not in CLASS_MAP."
                )
                ignored_categories.add(coco_name)

        if not target_category_map:
            logging.warning(
                f"No target categories found in {json_path} based on "
                f"CLASS_MAP keys: {list(CLASS_MAP.keys())}. "
                f"Available COCO categories: {list(coco_categories.values())}"
            )
            # Continue processing other files, but this one yields nothing
            return 0

        # Create image map: image_id -> image_info (filename, width, height)
        image_map = {img['id']: img for img in data.get('images', [])}

        # Group annotations by image_id
        annotations_by_image = {}
        # Handle missing 'annotations' key gracefully
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        # Process each image that has annotations
        # Using list(annotations_by_image.items()) for tqdm compatibility
        for img_id, annotations in annotations_by_image.items():
            if img_id not in image_map:
                logging.warning(
                    f"Image ID {img_id} in annotations but not in images "
                    f"section of {json_path}. Skipping."
                )
                continue

            img_info = image_map[img_id]
            img_filename_original = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']

            src_img_path = os.path.join(src_img_dir, img_filename_original)
            if not os.path.exists(src_img_path):
                logging.warning(
                    f"Source image not found, skipping: {src_img_path}"
                )
                continue

            if img_width <= 0 or img_height <= 0:
                logging.warning(
                    f"Skipping image {img_filename_original} (ID: {img_id}) "
                    f"due to invalid dimensions (W:{img_width}, H:{img_height})."
                )
                continue

            yolo_lines = []
            for ann in annotations:
                category_id = ann['category_id']
                # Process only if it's a category we mapped
                if category_id in target_category_map:
                    yolo_class_id = target_category_map[category_id]
                    # COCO format: [x_min, y_min, width, height]
                    bbox = ann['bbox']
                    x_min, y_min, width, height = bbox

                    # Validate bbox dimensions
                    if width <= 0 or height <= 0:
                        logging.warning(
                            f"Skipping annotation in {img_filename_original} "
                            f"(ID: {img_id}) due to zero bbox dimension."
                        )
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

            # Apply prefix
            prefixed_img_filename = f"hh_{img_filename_original}"
            base_filename = os.path.splitext(prefixed_img_filename)[0]
            label_filename = f"{base_filename}.txt"

            # Save YOLO label file (even if empty if image processed)
            label_path = os.path.join(target_lbl_dir, label_filename)
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_lines) + "\n")

            # Copy image file with prefix
            target_img_path = os.path.join(
                target_img_dir, prefixed_img_filename
            )
            try:
                # copy2 preserves metadata
                shutil.copy2(src_img_path, target_img_path)
                processed_count += 1
            except Exception as copy_err:
                logging.error(
                    f"Failed to copy {src_img_path} to {target_img_path}: "
                    f"{copy_err}"
                )
                # If copy fails, remove the potentially created label file
                if os.path.exists(label_path):
                    os.remove(label_path)

    except FileNotFoundError:
        logging.error(f"COCO JSON file not found: {json_path}")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON file: {json_path}")
    except KeyError as e:
        logging.error(f"Missing expected key in COCO JSON {json_path}: {e}")
    except Exception as e:
        logging.error(
            f"Unexpected error processing {json_path}: {e}", exc_info=True
        )

    logging.info(
        f"Finished processing {os.path.basename(json_path)}. "
        f"Copied {processed_count} images/labels."
    )
    return processed_count

def convert_parquet_to_yolo(
    parquet_path, target_img_dir, target_lbl_dir, split_name
):
    """Reads Parquet, extracts images, converts annotations, saves images & YOLO.

    Saves files to the final target directories with 'cs_' prefix.

    Args:
        parquet_path (str): Path to the input Parquet file.
        target_img_dir (str): Final directory to save the extracted images.
        target_lbl_dir (str): Final directory to save YOLO label files.
        split_name (str): Name of the dataset split (e.g., 'train').

    Returns:
        int: Count of images successfully processed and saved.
    """
    processed_count = 0
    # In construction-safety dataset: helmet=1, no-helmet=2
    source_helmet_class_id = 1
    source_no_helmet_class_id = 2

    # Use prefixed ids for construction-safety dataset
    prefixed_helmet_id = f'cs_{source_helmet_class_id}'
    prefixed_no_helmet_id = f'cs_{source_no_helmet_class_id}'
    
    target_helmet_class_id = CLASS_MAP.get(prefixed_helmet_id)  # Should be 1
    target_no_helmet_class_id = CLASS_MAP.get(prefixed_no_helmet_id)  # Should be 0

    if target_helmet_class_id is None or target_no_helmet_class_id is None:
        logging.error(
            f"Target class mapping not found for prefixed IDs "
            f"'{prefixed_helmet_id}' or '{prefixed_no_helmet_id}' in CLASS_MAP."
        )
        return 0

    try:
        df = pd.read_parquet(parquet_path)
        logging.info(f"Read {len(df)} records from {parquet_path}")

        required_cols = ['image', 'image_id', 'width', 'height', 'objects']
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            logging.error(
                f"Missing required columns in {parquet_path}. Need: "
                f"{required_cols}, Missing: {missing}"
            )
            return 0

        # Process each row
        # Using df.iterrows() and tqdm if progress bar is desired
        for index, row in df.iterrows():
            try:
                image_id = row['image_id']  # Keep original ID for filename
                img_width = row['width']
                img_height = row['height']
                # Format: {'category': [id1,...], 'bbox': [[...],...]}
                objects_data = row['objects']
                image_info = row['image']  # Expecting {'bytes': b'...'}

                # Validate basic info
                if not isinstance(image_info, dict) or 'bytes' not in image_info:
                    logging.warning(
                        f"Skipping record index {index} (ID: {image_id}) "
                        f"due to missing or invalid 'image' structure."
                    )
                    continue
                image_bytes = image_info['bytes']
                if not image_bytes or img_width <= 0 or img_height <= 0:
                    logging.warning(
                        f"Skipping record index {index} (ID: {image_id}) "
                        f"due to empty image bytes or invalid dimensions."
                    )
                    continue

                # Attempt to save image and determine format
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    # Convert potentially problematic modes like P to RGB
                    if img.mode == 'P':
                        img = img.convert('RGB')
                    elif img.mode == 'RGBA':
                        img = img.convert('RGB')  # Handle alpha if needed

                    img_format = img.format.lower() if img.format else 'jpg'
                    # Ensure safe format
                    if img_format not in ['jpeg', 'jpg', 'png']:
                        logging.warning(
                            f"Unsupported format '{img_format}' for ID "
                            f"{image_id}. Saving as PNG."
                        )
                        img_format = 'png'

                    # Use prefix and image_id for filename
                    prefixed_img_filename = f"cs_{image_id}.{img_format}"
                    target_img_path = os.path.join(
                        target_img_dir, prefixed_img_filename
                    )
                    img.save(target_img_path)
                except Exception as img_err:
                    logging.error(
                        f"Failed to save image for ID {image_id}: {img_err}"
                    )
                    continue  # Skip this record if image fails

                # Process annotations
                yolo_lines = []
                if (isinstance(objects_data, dict) and
                        'bbox' in objects_data and
                        'category' in objects_data and
                        isinstance(objects_data['category'], list) and
                        isinstance(objects_data['bbox'], list) and
                        len(objects_data['category']) ==
                        len(objects_data['bbox'])):

                    num_annotations = len(objects_data['category'])
                    for i in range(num_annotations):
                        try:
                            category_id = objects_data['category'][i]
                            bbox = objects_data['bbox'][i]  # COCO [xmin,ymin,w,h]

                            # Check if it's either helmet (1) or no-helmet (2) category
                            if category_id == source_helmet_class_id:
                                target_class_id = target_helmet_class_id
                                x_min, y_min, width, height = bbox

                                if width <= 0 or height <= 0:
                                    logging.warning(
                                        f"Skipping annotation {i} for image "
                                        f"ID {image_id} due to zero bbox dim."
                                    )
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
                                    f"{target_class_id} {x_center_norm:.6f} "
                                    f"{y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                                )
                            elif category_id == source_no_helmet_class_id:
                                target_class_id = target_no_helmet_class_id
                                x_min, y_min, width, height = bbox

                                if width <= 0 or height <= 0:
                                    logging.warning(
                                        f"Skipping annotation {i} for image "
                                        f"ID {image_id} due to zero bbox dim."
                                    )
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
                                    f"{target_class_id} {x_center_norm:.6f} "
                                    f"{y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                                )
                        except (IndexError, TypeError) as ann_idx_err:
                            logging.warning(
                                f"Format error processing annotation {i} for ID "
                                f"{image_id} in {parquet_path}: {ann_idx_err}"
                            )
                        except Exception as ann_err:
                            logging.warning(
                                f"Unexpected error processing annotation {i} for "
                                f"ID {image_id} in {parquet_path}: {ann_err}"
                            )
                # Check if objects_data exists but is wrong format
                elif objects_data is not None:
                    logging.warning(
                        f"Unexpected format/empty 'objects' data for ID "
                        f"{image_id} in {parquet_path}. Expected dict with "
                        f"'category' and 'bbox' lists."
                    )

                # Save YOLO label file (even if empty) with prefix
                base_filename = os.path.splitext(prefixed_img_filename)[0]
                label_filename = f"{base_filename}.txt"
                label_path = os.path.join(target_lbl_dir, label_filename)
                with open(label_path, 'w') as f:
                    f.write("\n".join(yolo_lines) + "\n")

                # Count successful image save + label save attempt
                processed_count += 1

            except KeyError as key_err:
                logging.warning(
                    f"Skipping record index {index} due to missing key: "
                    f"{key_err}"
                )
            except Exception as row_err:
                logging.warning(
                    f"Skipping record index {index} due to error: {row_err}",
                    exc_info=True
                )

    except ImportError:
        logging.error(
            "Failed to process Parquet: pandas or pyarrow not installed. "
            "Try `pip install pandas pyarrow`"
        )
    except FileNotFoundError:
        logging.error(f"Parquet file not found: {parquet_path}")
    except Exception as e:
        logging.error(
            f"Unexpected error processing {parquet_path}: {e}", exc_info=True
        )

    logging.info(
        f"Finished processing {os.path.basename(parquet_path)}. "
        f"Saved {processed_count} images/labels."
    )
    return processed_count

def split_and_copy_files(
    file_pairs, train_img_dir, train_lbl_dir,
    valid_img_dir, valid_lbl_dir, test_img_dir, test_lbl_dir,
    prefix
):
    """Splits file pairs (img_path, lbl_path) and copies them to final dirs.

    Args:
        file_pairs (list): List of tuples: (src_image_path, temp_label_path).
        train_img_dir (str): Target directory for training images.
        train_lbl_dir (str): Target directory for training labels.
        valid_img_dir (str): Target directory for validation images.
        valid_lbl_dir (str): Target directory for validation labels.
        test_img_dir (str): Target directory for testing images.
        test_lbl_dir (str): Target directory for testing labels.
        prefix (str): Prefix to add to filenames (e.g., 'hd_').
    """
    if not file_pairs:
        logging.warning("No file pairs provided for splitting.")
        return 0

    total_files = len(file_pairs)
    logging.info(
        f"Attempting to split {total_files} file pairs with prefix "
        f"'{prefix}'..."
    )

    # Ensure splits add up to 1.0 for train_test_split
    if not (0.999 < (TRAIN_SPLIT + VALID_SPLIT + TEST_SPLIT) < 1.001):
        logging.error(
            f"Train/Valid/Test splits ({TRAIN_SPLIT}/{VALID_SPLIT}/"
            f"{TEST_SPLIT}) do not sum to 1.0. Aborting split."
        )
        return 0

    # Split into train and remaining (valid + test)
    train_files, remaining_files = train_test_split(
        file_pairs,
        train_size=TRAIN_SPLIT,
        random_state=42  # for reproducibility
    )

    # Calculate split ratio for valid vs test from the remainder
    valid_files, test_files = [], []
    if remaining_files:
        denominator = VALID_SPLIT + TEST_SPLIT
        # Use tolerance for float comparison
        if denominator > 1e-9:
            relative_test_split = TEST_SPLIT / denominator
            if relative_test_split >= 1.0:  # Only test files remain
                valid_files = []
                test_files = remaining_files
            elif relative_test_split <= 0.0:  # Only valid files remain
                valid_files = remaining_files
                test_files = []
            else:  # Split remaining into valid and test
                valid_files, test_files = train_test_split(
                    remaining_files,
                    test_size=relative_test_split,
                    random_state=42  # use same random state
                )
        elif TRAIN_SPLIT >= 1.0:  # Only train split requested
            pass  # valid_files and test_files remain empty
        else:  # Should not happen if sum is 1, but safeguard
            logging.warning(
                "Could not split remaining files due to zero denominator."
            )
            # Put all remaining in validation as fallback
            valid_files = remaining_files

    logging.info(
        f"Splitting into: Train={len(train_files)}, "
        f"Valid={len(valid_files)}, Test={len(test_files)}"
    )

    # Function to copy a list of file pairs to target directories with prefix
    def copy_files_with_prefix(files, target_img_dir, target_lbl_dir,
                               split_name):
        copied_count = 0
        if not files:
            return 0
        logging.info(
            f"Copying {len(files)} '{prefix}' files to {split_name}..."
        )
        # Using tqdm if progress bar is desired
        for img_path, lbl_path in files:
            try:
                if not os.path.exists(img_path):
                    logging.warning(
                        f"Source image missing during copy: {img_path}"
                    )
                    continue
                # Label path might be None if no relevant objects were found
                if lbl_path and not os.path.exists(lbl_path):
                    logging.warning(
                        f"Source label missing during copy: {lbl_path}"
                    )
                    # Still copy image, create empty label file
                    lbl_path = None  # Treat as missing

                img_filename = os.path.basename(img_path)
                lbl_filename_base = (
                    os.path.basename(lbl_path) if lbl_path
                    else os.path.splitext(img_filename)[0] + ".txt"
                )

                prefixed_img_filename = f"{prefix}{img_filename}"
                prefixed_lbl_filename = f"{prefix}{lbl_filename_base}"

                target_img_path = os.path.join(
                    target_img_dir, prefixed_img_filename
                )
                target_lbl_path = os.path.join(
                    target_lbl_dir, prefixed_lbl_filename
                )

                shutil.copy2(img_path, target_img_path)

                # Copy existing label or create empty one
                if lbl_path:
                    shutil.copy2(lbl_path, target_lbl_path)
                else:
                    # Create empty label file if source didn't exist or
                    # wasn't generated (no relevant objects)
                    with open(target_lbl_path, 'w') as _:
                        # No need to write anything, just create the file
                        pass
                    logging.debug(
                        f"Created empty label file: {target_lbl_path}"
                    )

                copied_count += 1
            except Exception as e:
                logging.error(
                    f"Error copying {img_path} or {lbl_path} to {split_name}: "
                    f"{e}", exc_info=True
                )
        logging.info(
            f"Finished copying {copied_count} '{prefix}' files to "
            f"{split_name}."
        )
        return copied_count

    # Copy files for each split
    total_copied = 0
    total_copied += copy_files_with_prefix(
        train_files, train_img_dir, train_lbl_dir, "train"
    )
    total_copied += copy_files_with_prefix(
        valid_files, valid_img_dir, valid_lbl_dir, "valid"
    )
    total_copied += copy_files_with_prefix(
        test_files, test_img_dir, test_lbl_dir, "test"
    )

    return total_copied

# --- Main Processing Logic ---

def main():
    """Main function to orchestrate dataset processing and combination."""
    logging.info("Starting combined dataset preparation...")

    # 1. Create final output directories (YOLO structure)
    clear_or_create_dir(TRAIN_IMG_DIR)
    clear_or_create_dir(TRAIN_LBL_DIR)
    clear_or_create_dir(VALID_IMG_DIR)
    clear_or_create_dir(VALID_LBL_DIR)
    clear_or_create_dir(TEST_IMG_DIR)
    clear_or_create_dir(TEST_LBL_DIR)

    # Temp dir for HelmetDetection labels before splitting
    temp_lbl_dir = os.path.join(OUTPUT_DIR, "temp_hd_labels")
    clear_or_create_dir(temp_lbl_dir)

    # List to hold (image_path, temp_label_path) for HelmetDetection
    helmet_detection_file_pairs = []
    total_processed_count = 0  # Track across all datasets

    # --- Process HelmetDetection (VOC XML) ---
    logging.info("--- Processing HelmetDetection (VOC XML) ---")
    hd_xml_dir = os.path.join(HELMET_DETECTION_PATH, "annotations")
    hd_img_dir = os.path.join(HELMET_DETECTION_PATH, "images")

    hd_processed_count = 0
    if os.path.exists(hd_xml_dir) and os.path.exists(hd_img_dir):
        xml_files = [f for f in os.listdir(hd_xml_dir) if f.endswith(".xml")]
        logging.info(f"Found {len(xml_files)} XML files in {hd_xml_dir}")

        # Using tqdm if progress bar is desired
        for xml_file in xml_files:
            xml_path = os.path.join(hd_xml_dir, xml_file)
            # Convert -> saves label temporarily, returns img/label filenames
            img_filename, base_lbl_filename = convert_voc_to_yolo(
                xml_path, hd_img_dir, temp_lbl_dir
            )

            # If successful (even if no target labels found)
            if img_filename:
                hd_processed_count += 1
                # Find source image path again based on returned filename
                src_img_path = os.path.join(hd_img_dir, img_filename)
                if os.path.exists(src_img_path):
                    # If label file was created, construct its temp path
                    temp_lbl_path = None
                    if base_lbl_filename:
                        temp_lbl_path = os.path.join(
                            temp_lbl_dir, base_lbl_filename
                        )
                        # Sanity check if label file actually exists
                        if not os.path.exists(temp_lbl_path):
                            logging.warning(
                                f"Label file {base_lbl_filename} reported "
                                f"but not found in temp dir for {xml_file}"
                            )
                            temp_lbl_path = None  # Treat as if no label file

                    helmet_detection_file_pairs.append(
                        (src_img_path, temp_lbl_path)
                    )
                else:
                    logging.warning(
                        f"Could not find source image {img_filename} after "
                        f"conversion for XML: {xml_file}"
                    )
            else:
                logging.warning(f"Skipped processing for XML: {xml_file}")
        logging.info(
            f"Processed {hd_processed_count} images from HelmetDetection."
        )
    else:
        logging.warning(
            f"HelmetDetection source directories not found or missing: "
            f"XML='{hd_xml_dir}', Images='{hd_img_dir}'"
        )

    # --- Process hard-hat-detection (COCO JSON) ---
    logging.info("--- Processing hard-hat-detection (COCO JSON) ---")
    hh_processed_count = 0
    for split in ["train", "valid", "test"]:
        json_file = os.path.join(
            HARD_HAT_PATH, split, "_annotations.coco.json"
        )
        img_dir_src = os.path.join(HARD_HAT_PATH, split)

        if os.path.exists(json_file) and os.path.exists(img_dir_src):
            logging.info(f"Processing {split} split from {json_file}...")
            # Determine final output dirs based on split
            if split == "train":
                img_out_dir, lbl_out_dir = TRAIN_IMG_DIR, TRAIN_LBL_DIR
            elif split == "valid":
                img_out_dir, lbl_out_dir = VALID_IMG_DIR, VALID_LBL_DIR
            else:  # test
                img_out_dir, lbl_out_dir = TEST_IMG_DIR, TEST_LBL_DIR

            # Convert -> copies images/saves labels directly to final dirs
            count = convert_coco_json_to_yolo(
                json_file, img_dir_src, img_out_dir, lbl_out_dir
            )
            hh_processed_count += count
        else:
            logging.warning(
                f"COCO JSON or image dir not found for hard-hat '{split}' "
                f"split: JSON='{json_file}', Images='{img_dir_src}'"
            )
    total_processed_count += hh_processed_count
    logging.info(
        f"Processed {hh_processed_count} images/labels from hard-hat-detection."
    )

    # --- Process construction-safety-gsnvb (Parquet) ---
    logging.info("--- Processing construction-safety-gsnvb (Parquet) ---")
    cs_processed_count = 0
    if os.path.exists(CONSTRUCTION_SAFETY_PATH):
        for filename in os.listdir(CONSTRUCTION_SAFETY_PATH):
            if filename.endswith(".parquet"):
                parquet_file = os.path.join(CONSTRUCTION_SAFETY_PATH, filename)
                split_name = None
                img_out_dir, lbl_out_dir = None, None

                # Determine split and target dirs from filename
                if "train" in filename:
                    split_name = "train"
                    img_out_dir, lbl_out_dir = TRAIN_IMG_DIR, TRAIN_LBL_DIR
                elif "validation" in filename:  # Matches 'validation.parquet'
                    split_name = "valid"
                    img_out_dir, lbl_out_dir = VALID_IMG_DIR, VALID_LBL_DIR
                elif "test" in filename:
                    split_name = "test"
                    img_out_dir, lbl_out_dir = TEST_IMG_DIR, TEST_LBL_DIR

                if split_name and img_out_dir and lbl_out_dir:
                    logging.info(
                        f"Processing {split_name} split from {filename}..."
                    )
                    # Convert -> saves images/labels directly to final dirs
                    count = convert_parquet_to_yolo(
                        parquet_file, img_out_dir, lbl_out_dir, split_name
                    )
                    cs_processed_count += count
                else:
                    logging.warning(
                        f"Could not determine split or target dirs for "
                        f"Parquet file: {filename}"
                    )
    else:
        logging.warning(
            f"Construction Safety path not found: {CONSTRUCTION_SAFETY_PATH}"
        )
    total_processed_count += cs_processed_count
    logging.info(
        f"Processed {cs_processed_count} images/labels from construction-safety."
    )

    # --- Split and Copy HelmetDetection files ---
    logging.info("--- Splitting and Copying HelmetDetection files ---")
    hd_copied_count = split_and_copy_files(
        helmet_detection_file_pairs,
        TRAIN_IMG_DIR, TRAIN_LBL_DIR,
        VALID_IMG_DIR, VALID_LBL_DIR,
        TEST_IMG_DIR, TEST_LBL_DIR,
        prefix="hd_"  # Add 'hd_' prefix
    )
    total_processed_count += hd_copied_count  # Add successfully copied count
    logging.info(
        f"Copied {hd_copied_count} images/labels from HelmetDetection."
    )

    # --- Cleanup ---
    if os.path.exists(temp_lbl_dir):
        logging.info(f"Removing temporary directory: {temp_lbl_dir}")
        shutil.rmtree(temp_lbl_dir)

    # --- Create data.yaml ---
    logging.info("--- Creating data.yaml file ---")
    # Use relative paths from the location of data.yaml (OUTPUT_DIR)
    data_yaml_content = {
        # Absolute path to dataset root (recommended by YOLO)
        'path': os.path.abspath(OUTPUT_DIR),
        'train': os.path.relpath(TRAIN_IMG_DIR, OUTPUT_DIR),  # images/train
        'val': os.path.relpath(VALID_IMG_DIR, OUTPUT_DIR),    # images/valid
        'test': os.path.relpath(TEST_IMG_DIR, OUTPUT_DIR),   # images/test
        'nc': len(TARGET_CLASSES),
        'names': TARGET_CLASSES
    }
    data_yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    try:
        with open(data_yaml_path, 'w') as f_yaml:
            yaml.dump(data_yaml_content, f_yaml, default_flow_style=False,
                      sort_keys=False)
        logging.info(f"Successfully created {data_yaml_path}")
    except Exception as e:
        logging.error(f"Failed to create data.yaml: {e}")

    logging.info(
        f"Dataset preparation finished. Combined approximately "
        f"{total_processed_count} images/labels into '{OUTPUT_DIR}'."
    )


if __name__ == "__main__":
    # Check for dependencies (optional, can be removed if managed externally)
    try:
        # Imports are checked for availability, not direct usage here
        # pylint: disable=unused-import
        import pandas
        import PIL  # Needed by Image.open
        import pyarrow
        import sklearn
        import yaml
        # import tqdm # Import if progress bars are desired
    except ImportError as e:
        logging.error(
            f"Missing required package: {e}. Please install dependencies "
            f"(e.g., pip install pandas pyarrow scikit-learn PyYAML Pillow)"
        )
        exit(1)

    main()