from pathlib import Path
from PIL import Image, ImageEnhance
import random
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
BASE_DIR = Path("Sugarcane Leaf Image Dataset")

INPUT_FOLDERS = {
    "Healthy Leaves": BASE_DIR / "Healthy Leaves",
}

OUTPUT_BASE = Path("Augmented Sugarcane Leaf Dataset")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Choose one value per quality for preview clarity
BRIGHTNESS_FACTOR = 1.2
CONTRAST_FACTOR = 1.2
SATURATION_FACTOR = 1.2
SHARPNESS_FACTOR = 1.5
BLACK_POINT_SHIFT = -20

# How many original images to preview
NUM_PREVIEW_IMAGES = 4


# =========================
# HELPERS
# =========================
def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def apply_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def apply_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def apply_saturation(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(factor)


def apply_sharpness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Sharpness(img).enhance(factor)


def apply_black_point_shift(img: Image.Image, shift: int) -> Image.Image:
    return img.point(lambda p: max(0, min(255, p + shift)))


def save_image(img: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=95)


# =========================
# AUGMENT AND SAVE
# =========================
def augment_and_save(image_path: Path, output_dir: Path) -> int:
    saved_count = 0

    with Image.open(image_path) as img:
        img = ensure_rgb(img)
        stem = image_path.stem

        # Original
        save_image(img, output_dir / f"{stem}_original.jpg")
        saved_count += 1

        # Brightness
        bright = apply_brightness(img, BRIGHTNESS_FACTOR)
        save_image(bright, output_dir / f"{stem}_brightness.jpg")
        saved_count += 1

        # Contrast
        contrast = apply_contrast(img, CONTRAST_FACTOR)
        save_image(contrast, output_dir / f"{stem}_contrast.jpg")
        saved_count += 1

        # Saturation
        saturation = apply_saturation(img, SATURATION_FACTOR)
        save_image(saturation, output_dir / f"{stem}_saturation.jpg")
        saved_count += 1

        # Sharpness
        sharp = apply_sharpness(img, SHARPNESS_FACTOR)
        save_image(sharp, output_dir / f"{stem}_sharpness.jpg")
        saved_count += 1

        # Black point / shadow shift
        blackpoint = apply_black_point_shift(img, BLACK_POINT_SHIFT)
        save_image(blackpoint, output_dir / f"{stem}_blackpoint.jpg")
        saved_count += 1

    return saved_count


def process_folder(class_name: str, input_dir: Path, output_base: Path) -> None:
    if not input_dir.exists():
        print(f"[WARNING] Folder not found: {input_dir}")
        return

    output_dir = output_base / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"[INFO] No supported images found in: {input_dir}")
        return

    total_saved = 0

    for image_file in image_files:
        try:
            count = augment_and_save(image_file, output_dir)
            total_saved += count
            print(f"[DONE] {class_name} -> {image_file.name} | saved {count} files")
        except Exception as e:
            print(f"[ERROR] Could not process {image_file.name}: {e}")

    print(f"\n[SUMMARY] {class_name}: {total_saved} files saved in {output_dir}\n")


# =========================
# PREVIEW: SIDE BY SIDE FOR EACH QUALITY
# =========================
def show_side_by_side_per_quality(output_dir: Path, num_images: int = 4) -> None:
    original_files = list(output_dir.glob("*_original.jpg"))

    if not original_files:
        print("[INFO] No original images found for preview.")
        return

    chosen_originals = random.sample(original_files, min(num_images, len(original_files)))

    columns = [
        ("Original", "_original.jpg"),
        ("Brightness", "_brightness.jpg"),
        ("Contrast", "_contrast.jpg"),
        ("Saturation", "_saturation.jpg"),
        ("Sharpness", "_sharpness.jpg"),
        ("Black Point", "_blackpoint.jpg"),
    ]

    fig, axes = plt.subplots(
        len(chosen_originals),
        len(columns),
        figsize=(3.2 * len(columns), 3.5 * len(chosen_originals))
    )

    if len(chosen_originals) == 1:
        axes = [axes]

    for row, original_path in enumerate(chosen_originals):
        stem = original_path.name.replace("_original.jpg", "")

        for col, (title, suffix) in enumerate(columns):
            img_path = output_dir / f"{stem}{suffix}"
            ax = axes[row][col]

            if img_path.exists():
                with Image.open(img_path) as img:
                    img = ensure_rgb(img)
                    ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Not found", ha="center", va="center")
            if row == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# MAIN
# =========================
def main():
    print("Starting augmentation...\n")

    for class_name, input_dir in INPUT_FOLDERS.items():
        process_folder(class_name, input_dir, OUTPUT_BASE)

    print("All done.")
    print(f"Augmented dataset saved in: {OUTPUT_BASE.resolve()}")

    show_side_by_side_per_quality(
        OUTPUT_BASE / "Healthy Leaves",
        num_images=NUM_PREVIEW_IMAGES
    )


if __name__ == "__main__":
    main()