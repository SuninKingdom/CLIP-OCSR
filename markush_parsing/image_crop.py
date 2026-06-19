"""Image cropping using MinerU for layout analysis.

Input: a single Markush structure image
Output: cropped structure region + cropped text region

MinerU returns content_list with bbox [x1, y1, x2, y2] in pixel coordinates.
We use this to find the split point between structure and text regions.
"""

import io
import json
import logging
import os
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont
from mineru import MinerU

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class MinerULayout:
    """MinerU layout analysis result for a single image."""
    content_list: list  # Raw content_list from MinerU
    structure_bbox: list | None  # [x1, y1, x2, y2] of the chemical structure
    text_items: list  # List of text items with bbox and text
    extracted_images: list  # List of Image objects extracted by MinerU


@dataclass
class CropResult:
    structure_image: Image.Image
    text_image: Image.Image
    y_threshold: float
    layout: MinerULayout
    mineru_structure_image: Image.Image | None = None  # MinerU extracted chemical image
    ocr_text: str = ""  # OCR text from MinerU text items


def draw_bbox_on_image(
    image: Image.Image,
    content_list: list,
    bbox_offset: tuple[int, int] = (10, 20),
) -> Image.Image:
    """Draw red bounding boxes on the image based on MinerU content_list.

    Args:
        image: Original PIL Image
        content_list: MinerU content_list with bbox coordinates
        bbox_offset: (dx, dy) pixel offset to correct MinerU systematic bias.
                     Positive values shift right (x) and down (y).

    Returns:
        New Image with red bounding boxes drawn
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    dx, dy = bbox_offset

    for item in content_list:
        bbox = item.get('bbox', [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy
        item_type = item.get('type', '')
        sub_type = item.get('sub_type', '')

        # Color: red for chemical structure, blue for text
        if item_type == 'image' and sub_type == 'chemical':
            color = 'red'
            width = 3
        else:
            color = 'blue'
            width = 2

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        # Add label
        label = f"{item_type}"
        if sub_type:
            label += f" ({sub_type})"
        draw.text((x1, y1 - 12), label, fill=color)

    return annotated


def analyze_layout(image_path: str, config: Config) -> MinerULayout:
    """Use MinerU to analyze image layout.

    Returns MinerULayout with content_list, structure_bbox, and text_items.
    """
    client = MinerU(config.mineru_token)
    try:
        result = client.extract(
            image_path,
            ocr=True,
            model='vlm',
            timeout=120,
        )
    finally:
        client.close()

    content_list = result.content_list
    if not content_list:
        logger.warning(f"MinerU returned empty content_list for {image_path}")
        return MinerULayout(content_list=[], structure_bbox=None, text_items=[], extracted_images=[])

    # Find chemical structure (image with sub_type "chemical")
    structure_bbox = None
    text_items = []

    for item in content_list:
        item_type = item.get('type', '')
        bbox = item.get('bbox', [])

        if item_type == 'image' and item.get('sub_type') == 'chemical':
            structure_bbox = bbox
        elif item_type == 'text':
            text_items.append({
                'text': item.get('text', ''),
                'bbox': bbox,
            })

    return MinerULayout(
        content_list=content_list,
        structure_bbox=structure_bbox,
        text_items=text_items,
        extracted_images=result.images,
    )


def find_split_threshold_from_layout(
    layout: MinerULayout, image_height: int, config: Config,
    y_offset: int = 20,
) -> float:
    """Find y-threshold to split structure from text using MinerU layout.

    Strategy:
    1. If structure_bbox found, use its y2 (bottom) as split point
    2. Otherwise, find "wherein" keyword in text items
    3. Fallback to gap detection between text items

    Args:
        y_offset: Pixel correction applied to MinerU bbox y-coordinates.
    """
    # Strategy 1: Use structure bounding box
    if layout.structure_bbox:
        _, _, _, y2 = layout.structure_bbox
        # Add generous margin below structure + y_offset correction
        y_threshold = (y2 + y_offset + 50) / image_height
        return min(y_threshold, 0.95)

    # Strategy 2: Find "wherein" or similar keywords
    keywords = ['wherein', 'where each', 'represent']
    for item in layout.text_items:
        text = item.get('text', '').lower()
        for keyword in keywords:
            if keyword in text:
                y1 = item['bbox'][1] + y_offset
                return y1 / image_height

    # Strategy 3: Gap detection between text items
    if len(layout.text_items) >= 2:
        midpoints = [
            (item['bbox'][1] + y_offset + item['bbox'][3] + y_offset) / 2
            for item in layout.text_items
        ]
        max_gap = 0
        gap_y = midpoints[0]
        for i in range(len(midpoints) - 1):
            gap = midpoints[i + 1] - midpoints[i]
            if gap > max_gap:
                max_gap = gap
                gap_y = (midpoints[i] + midpoints[i + 1]) / 2

        if max_gap > 30:  # At least 30 pixels gap
            return gap_y / image_height

    # Fallback
    return config.text_y_threshold


def crop_markush_image(
    image_path: str, config: Config, save_dir: str | None = None
) -> CropResult:
    """Crop a Markush image into structure region and text region using MinerU.

    Args:
        image_path: Path to the input image
        config: Configuration
        save_dir: If provided, save intermediate results (layout JSON, cropped images)

    Returns:
        CropResult with structure_image, text_image, y_threshold, and layout
    """
    image = Image.open(image_path)
    width, height = image.size

    # Analyze layout with MinerU
    layout = analyze_layout(image_path, config)

    # Find split point
    y_threshold = find_split_threshold_from_layout(layout, height, config)
    y_split_px = int(y_threshold * height)

    # Vertical: increase margin below y_split to avoid cutting structure bottom
    v_margin = int(0.05 * height)

    # Horizontal: determine left/right crop boundaries from bbox ranges
    all_x1 = []
    all_x2 = []
    for item in layout.content_list:
        bbox = item.get('bbox', [])
        if len(bbox) == 4:
            all_x1.append(bbox[0] + 10)  # apply x_offset
            all_x2.append(bbox[2] + 10)

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

    # Get MinerU extracted chemical image — match by filename to sub_type "chemical"
    mineru_structure_image = None
    if layout.extracted_images and layout.content_list:
        # Build lookup: filename -> extracted image object
        img_by_name = {img_obj.name: img_obj for img_obj in layout.extracted_images}
        # Find content_list entry with sub_type "chemical"
        for item in layout.content_list:
            if item.get('type') == 'image' and item.get('sub_type') == 'chemical':
                fname = os.path.basename(item.get('img_path', ''))
                if fname in img_by_name:
                    mineru_structure_image = Image.open(io.BytesIO(img_by_name[fname].data))
                    break
        # Fallback: use first extracted image if no chemical match
        if mineru_structure_image is None and layout.extracted_images:
            mineru_structure_image = Image.open(io.BytesIO(layout.extracted_images[0].data))

    # Combine OCR text from MinerU text items
    ocr_text = "\n".join(item.get('text', '') for item in layout.content_list if item.get('type') == 'text')

    # Save intermediate results if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Save MinerU layout
        layout_path = os.path.join(save_dir, 'mineru_layout.json')
        with open(layout_path, 'w', encoding='utf-8') as f:
            json.dump(layout.content_list, f, ensure_ascii=False, indent=2)

        # Save MinerU extracted images (chemical structure sub-images)
        for i, img_obj in enumerate(layout.extracted_images):
            img_path = os.path.join(save_dir, f'mineru_extracted_{i}_{img_obj.name}')
            with open(img_path, 'wb') as f:
                f.write(img_obj.data)

        # Save annotated image with bounding boxes
        annotated = draw_bbox_on_image(image, layout.content_list)
        annotated.save(os.path.join(save_dir, 'annotated.png'))

        # Save cropped images
        structure_image.save(os.path.join(save_dir, 'structure.png'))
        text_image.save(os.path.join(save_dir, 'text.png'))

        # Save MinerU extracted chemical image (if available)
        if mineru_structure_image:
            mineru_structure_image.save(os.path.join(save_dir, 'mineru_chemical.png'))

        # Save OCR text
        with open(os.path.join(save_dir, 'ocr_text.txt'), 'w', encoding='utf-8') as f:
            f.write(ocr_text)

        logger.info(f"Saved intermediate results to {save_dir}")

    return CropResult(
        structure_image=structure_image,
        text_image=text_image,
        y_threshold=y_threshold,
        layout=layout,
        mineru_structure_image=mineru_structure_image,
        ocr_text=ocr_text,
    )
