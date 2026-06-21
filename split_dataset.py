"""
split_dataset.py
-----------------
What this script does (in plain terms):

1. Looks through your S2TLD dataset folder and finds every photo + its matching
   XML label file (even if they're nested inside sub-folders like normal_1, normal_2).
2. Converts each XML label into the YOLO .txt format that YOLOv8 actually needs.
3. Splits everything fairly into three piles: train (70%), val (15%), test (15%) -
   making sure each pile gets a fair mix of red/green/yellow/off, not random luck.
4. Writes everything into a new clean folder called "dataset_split/" with the
   exact structure YOLOv8 expects, plus a data.yaml file to point YOLOv8 at it.

HOW TO RUN:
1. Install the two things this script needs (one-time):
     pip install scikit-learn

2. Edit the ROOT_DIR path below to point at your S2TLD720x1280 folder
   (the one with normal_1 / normal_2 inside it).

3. Run:
     python3 split_dataset.py
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from collections import Counter
from sklearn.model_selection import train_test_split

# ── EDIT THIS ────────────────────────────────────────────────────────────────
# Point this at the folder that CONTAINS your dataset (the one with normal_1,
# normal_2 inside it, or Annotations/JPEGImages directly inside it).
ROOT_DIR = "S2TLD720x1280"

# Where the clean, split, YOLO-ready dataset will be written.
OUTPUT_DIR = "dataset_split"

# Split ratios — must add up to 1.0
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)  # makes the split reproducible — same split every time you run it


# ── STEP 1: Find every image + matching XML, no matter how deep it's nested ──
def find_image_label_pairs(root_dir):
    pairs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == "JPEGImages":
            ann_dir = os.path.join(os.path.dirname(dirpath), "Annotations")
            if not os.path.isdir(ann_dir):
                continue
            for fname in filenames:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    name = os.path.splitext(fname)[0]
                    xml_path = os.path.join(ann_dir, name + ".xml")
                    img_path = os.path.join(dirpath, fname)
                    if os.path.isfile(xml_path):
                        pairs.append((img_path, xml_path))
    return pairs


# ── STEP 2: Read one XML file, return image size + list of (class, box) ──────
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)
        objects.append((cls_name, xmin, ymin, xmax, ymax))
    return width, height, objects


# ── STEP 3: Convert one box from VOC (pixel corners) to YOLO (normalized) ────
def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    box_w = (xmax - xmin) / img_w
    box_h = (ymax - ymin) / img_h
    return x_center, y_center, box_w, box_h


# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    print(f"Scanning '{ROOT_DIR}' for image/label pairs...")
    pairs = find_image_label_pairs(ROOT_DIR)
    print(f"Found {len(pairs)} image+label pairs.\n")

    if len(pairs) == 0:
        print("No pairs found. Check that ROOT_DIR points to the right folder.")
        return

    # Discover every class name actually used, so we don't hardcode it
    all_classes = set()
    parsed_cache = {}  # avoid re-parsing XML twice
    dominant_class_per_image = []

    for img_path, xml_path in pairs:
        width, height, objects = parse_xml(xml_path)
        parsed_cache[img_path] = (width, height, objects, xml_path)
        classes_in_image = [o[0] for o in objects]
        all_classes.update(classes_in_image)
        # Use the most common class in this image as its "label" for stratifying
        dominant = Counter(classes_in_image).most_common(1)[0][0] if classes_in_image else "none"
        dominant_class_per_image.append(dominant)

    class_list = sorted(all_classes)
    class_to_id = {name: i for i, name in enumerate(class_list)}
    print(f"Classes found: {class_list}\n")

    # ── STEP 4: Stratified split — fair mix of classes in each pile ──────────
    image_paths = [p[0] for p in pairs]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, dominant_class_per_image,
        train_size=TRAIN_RATIO, stratify=dominant_class_per_image, random_state=42
    )
    val_size_within_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_paths, test_paths, _, _ = train_test_split(
        temp_paths, temp_labels,
        train_size=val_size_within_temp, stratify=temp_labels, random_state=42
    )

    splits = {"train": train_paths, "val": val_paths, "test": test_paths}

    # ── STEP 5: Write out images + YOLO-format labels into clean folders ────
    for split_name, split_imgs in splits.items():
        img_out_dir = os.path.join(OUTPUT_DIR, split_name, "images")
        lbl_out_dir = os.path.join(OUTPUT_DIR, split_name, "labels")
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)

        for img_path in split_imgs:
            width, height, objects, xml_path = parsed_cache[img_path]
            fname = os.path.basename(img_path)
            name_no_ext = os.path.splitext(fname)[0]

            # Copy the image
            shutil.copy2(img_path, os.path.join(img_out_dir, fname))

            # Write the YOLO label file (one line per object in the image)
            label_lines = []
            for cls_name, xmin, ymin, xmax, ymax in objects:
                cls_id = class_to_id[cls_name]
                xc, yc, w, h = voc_to_yolo(xmin, ymin, xmax, ymax, width, height)
                label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            with open(os.path.join(lbl_out_dir, name_no_ext + ".txt"), "w") as f:
                f.write("\n".join(label_lines))

        print(f"{split_name}: {len(split_imgs)} images written to {img_out_dir}")

    # ── STEP 6: Write data.yaml — tells YOLOv8 where everything is ──────────
    yaml_content = f"""train: {os.path.abspath(os.path.join(OUTPUT_DIR, 'train', 'images'))}
val: {os.path.abspath(os.path.join(OUTPUT_DIR, 'val', 'images'))}
test: {os.path.abspath(os.path.join(OUTPUT_DIR, 'test', 'images'))}

nc: {len(class_list)}
names: {class_list}
"""
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"\ndata.yaml written to {OUTPUT_DIR}/data.yaml")
    print("\nDone! Your dataset is now split and YOLO-ready.")
    print(f"Total: {len(pairs)} | Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")


if __name__ == "__main__":
    main()