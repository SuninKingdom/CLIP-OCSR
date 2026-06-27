"""Image cropping using pre-computed local MinerU pipeline output.

Reads MinerU output files (content_list.json + extracted images) from a local
directory instead of calling the MinerU API. This ensures deterministic,
reproducible results.

Input: image path + MinerU output directory
Output: cropped structure region + cropped text region + OCR text + chemical image

MinerU returns content_list with bbox [x1, y1, x2, y2] in pixel coordinates.
We use this to find the split point between structure and text regions.
"""

import json
import logging
import os

from PIL import Image, ImageDraw

from config import Config

logger = logging.getLogger(__name__)


class MinerULayout:
    """MinerU layout analysis result for a single image."""

    def __init__(self, content_list: list, images_dir: str):
        self.content_list = content_list
        self.images_dir = images_dir

        # Find chemical structure image (type=image, first one)
        self.structure_image = None
        self.structure_bbox = None
        self.text_items = []

        for item in content_list:
            item_type = item.get("type", "")
            bbox = item.get("bbox", [])

            if item_type == "image" and self.structure_image is None:
                self.structure_bbox = bbox
                img_path = item.get("img_path", "")
                if img_path:
                    full_path = os.path.join(images_dir, os.path.basename(img_path))
                    if os.path.exists(full_path):
                        self.structure_image = Image.open(full_path)
            elif item_type == "text":
                self.text_items.append(
                    {"text": item.get("text", ""), "bbox": bbox}
                )

    @property
    def ocr_text(self) -> str:
        """Combine all text items into a single string."""
        return "\n".join(item.get("text", "") for item in self.content_list if item.get("type") == "text")


class CropResult:
    """Result of cropping a Markush image."""

    def __init__(
        self,
        structure_image: Image.Image,
        text_image: Image.Image,
        y_threshold: float,
        layout: MinerULayout,
        mineru_structure_image: Image.Image | None = None,
        ocr_text: str = "",
    ):
        self.structure_image = structure_image
        self.text_image = text_image
        self.y_threshold = y_threshold
        self.layout = layout
        self.mineru_structure_image = mineru_structure_image
        self.ocr_text = ocr_text


def draw_bbox_on_image(image: Image.Image, content_list: list) -> Image.Image:
    """Draw red bounding boxes on the image based on MinerU content_list."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for item in content_list:
        bbox = item.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        item_type = item.get("type", "")

        if item_type == "image":
            color = "red"
            width = 3
        else:
            color = "blue"
            width = 2

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = item_type
        draw.text((x1, y1 - 12), label, fill=color)

    return annotated


def load_mineru_output(image_name: str, mineru_output_dir: str) -> MinerULayout | None:
    """Load pre-computed MinerU output for a given image.

    Args:
        image_name: e.g. "markush_representation_01.png"
        mineru_output_dir: root MinerU output directory

    Returns:
        MinerULayout or None if not found
    """
    stem = os.path.splitext(image_name)[0]
    sample_dir = os.path.join(mineru_output_dir, stem, "auto")

    # Try content_list.json first, then content_list_v2.json
    for fname in [f"{stem}_content_list.json", f"{stem}_content_list_v2.json"]:
        cl_path = os.path.join(sample_dir, fname)
        if os.path.exists(cl_path):
            with open(cl_path, "r", encoding="utf-8") as f:
                content_list = json.load(f)
            images_dir = os.path.join(sample_dir, "images")
            return MinerULayout(content_list, images_dir)

    logger.warning(f"MinerU output not found for {image_name} in {mineru_output_dir}")
    return None


def crop_markush_image(
    image_path: str, config: Config, save_dir: str | None = None
) -> CropResult:
    """Crop a Markush image using pre-computed MinerU layout.

    Args:
        image_path: Path to the input image
        config: Configuration (must have mineru_output_dir set)
        save_dir: If provided, save intermediate results

    Returns:
        CropResult with structure_image, text_image, y_threshold, and layout
    """
    image = Image.open(image_path)
    width, height = image.size
    image_name = os.path.basename(image_path)

    # Load pre-computed MinerU layout
    layout = load_mineru_output(image_name, config.mineru_output_dir)
    if layout is None:
        raise FileNotFoundError(
            f"MinerU output not found for {image_name}. "
            f"Run MinerU pipeline first: mineru -p {image_path} -o {config.mineru_output_dir} -b pipeline"
        )

    content_list = layout.content_list

    # Find split point: use structure bbox bottom, or text items, or fallback
    y_threshold = config.text_y_threshold
    if layout.structure_bbox:
        _, _, _, y2 = layout.structure_bbox
        y_threshold = min((y2 + 50) / height, 0.95)
    elif layout.text_items:
        # Find "wherein" or similar keywords
        keywords = ["wherein", "where each", "represent"]
        for item in layout.text_items:
            text = item.get("text", "").lower()
            for keyword in keywords:
                if keyword in text:
                    y1 = item["bbox"][1]
                    y_threshold = y1 / height
                    break

    y_split_px = int(y_threshold * height)

    # Vertical margin
    v_margin = int(0.05 * height)

    # Horizontal bounds from all items
    all_x1 = []
    all_x2 = []
    for item in content_list:
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            all_x1.append(bbox[0])
            all_x2.append(bbox[2])

    if all_x1 and all_x2:
        h_margin = int(0.05 * width)
        x_left = max(min(all_x1) - h_margin, 0)
        x_right = min(max(all_x2) + h_margin, width)
    else:
        x_left = 0
        x_right = width

    # Crop
    structure_image = image.crop((x_left, 0, x_right, max(y_split_px - v_margin, 1)))
    text_image = image.crop((x_left, max(y_split_px - v_margin, 0), x_right, height))

    # Get MinerU extracted chemical image
    mineru_structure_image = layout.structure_image

    # Save intermediate results if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Save MinerU layout
        layout_path = os.path.join(save_dir, "mineru_layout.json")
        with open(layout_path, "w", encoding="utf-8") as f:
            json.dump(content_list, f, ensure_ascii=False, indent=2)

        # Save annotated image with bounding boxes
        annotated = draw_bbox_on_image(image, content_list)
        annotated.save(os.path.join(save_dir, "annotated.png"))

        # Save cropped images
        structure_image.save(os.path.join(save_dir, "structure.png"))
        text_image.save(os.path.join(save_dir, "text.png"))

        # Save MinerU extracted chemical image (if available)
        if mineru_structure_image:
            mineru_structure_image.save(os.path.join(save_dir, "mineru_chemical.png"))

        # Save OCR text
        with open(os.path.join(save_dir, "ocr_text.txt"), "w", encoding="utf-8") as f:
            f.write(layout.ocr_text)

        logger.info(f"Saved intermediate results to {save_dir}")

    return CropResult(
        structure_image=structure_image,
        text_image=text_image,
        y_threshold=y_threshold,
        layout=layout,
        mineru_structure_image=mineru_structure_image,
        ocr_text=layout.ocr_text,
    )
