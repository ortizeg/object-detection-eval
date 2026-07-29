"""VLM auto-labeling task: directory of images -> single COCO annotations JSON.

Ported from the source repo's `VLMAnnotationTask` (VLM-03). Unlike the
source (one self-contained `DetectionAnnotation` JSON per image), this task
aggregates every image's detections into ONE COCO file via
`object_detection_eval.annotate.coco_writer.write_coco`, so the output loads
back through `load_coco_gt()` without error -- the round trip VLM-03
requires.

`GeminiInferencer` is imported lazily inside `run_vlm_annotation` so
importing this module stays torch/genai-free (VLM-04): the `[vlm]` extra is
only required at call time, not at module-import time.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from object_detection_eval.annotate.coco_writer import ImageDetections, write_coco
from object_detection_eval.data.image import ImageLoader

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})


def run_vlm_annotation(
    image_dir: Path | str,
    classes: list[str],
    output_path: Path | str,
    model_name: str = "gemini-2.5-pro",
    prompt_template: str | None = None,
) -> Path:
    """Run a VLM over every image in `image_dir` and write one COCO file.

    Args:
        image_dir: Directory containing images to auto-label.
        classes: Ordered class names -- defines both the COCO
            `categories` id->name map and the eval class ids
            `GeminiInferencer` resolves labels to (index in `classes`).
        output_path: Where to write the aggregated COCO annotations JSON.
        model_name: Name of the Gemini model to use.
        prompt_template: Optional custom prompt template.

    Returns:
        The path written by `write_coco` (same as `output_path`).
    """
    # Imported lazily: google.genai/torch only need to be installed under
    # the [vlm] extra at call time, not at module-import time (VLM-04).
    from object_detection_eval.inference.vlm.gemini import GeminiInferencer

    image_dir = Path(image_dir)
    output_path = Path(output_path)
    categories = dict(enumerate(classes))

    logger.info(f"Initializing GeminiInferencer with model: {model_name}")
    inferencer = GeminiInferencer(
        model_name=model_name,
        classes=classes,
        prompt_template=prompt_template,
    )

    image_paths = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not image_paths:
        logger.warning(f"No images found in {image_dir}")

    images: list[ImageDetections] = []
    for img_path in image_paths:
        try:
            loader = ImageLoader(img_path)
            image = loader.read()
            detections = inferencer.predict(
                image, image_width=loader.width, image_height=loader.height
            )
            images.append(
                ImageDetections(
                    filename=loader.filename,
                    width=loader.width,
                    height=loader.height,
                    detections=detections,
                )
            )
            logger.info(f"{loader.filename}: {len(detections)} detections")
        except Exception as exc:
            # One bad file must not abort the batch (T-05-13).
            logger.error(f"Failed to process image {img_path}: {exc}")
            continue

    write_coco(output_path, images, categories)
    logger.info(
        f"VLM auto-labeling complete. {len(images)}/{len(image_paths)} images "
        f"processed -> {output_path}"
    )
    return output_path
