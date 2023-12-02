import random
import shutil
from hashlib import md5
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import cv2

from tqdm import tqdm

random.seed(42)

# DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATASET_URL = (
    "https://github.com/tjmoon0104/pytorch-tiny-imagenet/releases/download/tiny-imagenet-dataset/tiny-imagenet-200.zip"
)
DATASET_ZIP = Path("./tiny-imagenet-200.zip")
DATASET_MD5_HASH = "90528d7ca1a48142e341f4ef8d21d0de"

# Download Dataset if needed
if not DATASET_ZIP.exists():
    print("Downloading the dataset, this may take a while...")

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=DATASET_URL.split("/")[-1]) as t:

        def show_progress(block_num, block_size, total_size):
            t.total = total_size
            t.update(block_num * block_size - t.n)

        urlretrieve(url=DATASET_URL, filename=DATASET_ZIP, reporthook=show_progress)

# Check MD5 Hash
with DATASET_ZIP.open("rb") as f:
    assert (
        md5(f.read()).hexdigest() == DATASET_MD5_HASH
    ), "The dataset zip file seems corrupted. Try to download it again."


# Remove existing data set
ORIGINAL_DATASET_DIR = Path("./original")
if ORIGINAL_DATASET_DIR.exists():
    shutil.rmtree(ORIGINAL_DATASET_DIR)

if not ORIGINAL_DATASET_DIR.exists():
    print("Extracting the dataset, this may take a while...")

    # Unzip the dataset
    with ZipFile(DATASET_ZIP, "r") as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc="Extracting"):
            zip_ref.extract(member, ORIGINAL_DATASET_DIR)

# Remove existing data set
DATASET_DIR = Path("./tiny-imagenet-200")
if DATASET_DIR.exists():
    shutil.rmtree(DATASET_DIR)

# Create the dataset directory
if not DATASET_DIR.exists():
    print("Creating the dataset directory...")
    DATASET_DIR.mkdir()

# Move train images to dataset directory
ORIGINAL_TRAIN_DIR = ORIGINAL_DATASET_DIR / "tiny-imagenet-200" / "train"
if ORIGINAL_TRAIN_DIR.exists():
    print("Moving train images...")
    ORIGINAL_TRAIN_DIR.replace(DATASET_DIR / "train")

# Get validation images and annotations
val_dict = {}
ORIGINAL_VAL_DIR = ORIGINAL_DATASET_DIR / "tiny-imagenet-200" / "val"
with (ORIGINAL_VAL_DIR / "val_annotations.txt").open("r") as f:
    for line in f.readlines():
        split_line = line.split("\t")
        if split_line[1] not in val_dict.keys():
            val_dict[split_line[1]] = [split_line[0]]
        else:
            val_dict[split_line[1]].append(split_line[0])


def split_list_randomly(input_list: list[str], split_ratio=0.5) -> dict[str, list[str]]:
    # Shuffle the input list in-place
    random.shuffle(input_list)

    # Calculate the index to split the list
    split_index = int(len(input_list) * split_ratio)

    # Split the list into two parts
    return {"val": input_list[:split_index], "test": input_list[split_index:]}


# Sample from validation images randomly into validation and test sets (50/50)
print("Splitting original dataset images...")
with tqdm(val_dict.items(), desc="Splitting images", unit="class") as t:
    for image_label, images in t:
        for split_type, split_images in split_list_randomly(images, split_ratio=0.5).items():
            for image in split_images:
                src = ORIGINAL_VAL_DIR / "images" / image
                dest_folder = DATASET_DIR / split_type / image_label / "images"
                dest_folder.mkdir(parents=True, exist_ok=True)
                src.replace(dest_folder / image)
        t.update()

# Remove original directory
shutil.rmtree(ORIGINAL_DATASET_DIR)

# Remove resized data set directory
RESIZED_DIR = Path("./tiny-224")
if RESIZED_DIR.exists():
    shutil.rmtree(RESIZED_DIR)

# Copy processed dataset to tiny-224
print("Copying processed dataset to tiny-224...")
shutil.copytree(DATASET_DIR, RESIZED_DIR)


# Resize images to 224x224
def resize_img(image_path: Path, size: int = 224) -> None:
    img = cv2.imread(image_path.as_posix())
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(image_path.as_posix(), img)


all_images = [*Path("tiny-224").glob("**/*.JPEG")]
print("Resizing images...")
with tqdm(all_images, desc="Resizing images", unit="file") as t:
    for image in t:
        resize_img(image, 224)
