from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import io
import uuid
from typing import Optional, List, Tuple
from PIL import Image, ImageEnhance
import numpy as np
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError
try:
    from boto3.exceptions import S3UploadFailedError
except Exception:  # boto3 older versions may not expose this
    class S3UploadFailedError(Exception):
        pass

image_router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "image_datasets"
MODELS_DIR = PROJECT_ROOT / "models" / "image"
PIPELINE_DIR = PROJECT_ROOT / "image_pipeline"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)


def _save_image_bytes_to_path(content: bytes, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(content)


def _open_image(content: bytes) -> Image.Image:
    return Image.open(io.BytesIO(content)).convert("RGB")


def _augment_image(img: Image.Image, count: int = 12) -> List[Tuple[str, Image.Image]]:
    """
    Produce 'count' augmented images including the original using fast PIL ops.
    If 'count' exceeds the base set, add randomized lightweight transforms.
    """
    import random
    variants: List[Tuple[str, Image.Image]] = []
    base: List[Tuple[str, Image.Image]] = []
    base.append(("orig", img.copy()))
    base.append(("flip", img.transpose(Image.FLIP_LEFT_RIGHT)))
    base.append(("rot15", img.rotate(15, expand=True, fillcolor=(255, 255, 255))))
    base.append(("rot-15", img.rotate(-15, expand=True, fillcolor=(255, 255, 255))))
    enhancer = ImageEnhance.Brightness(img)
    base.append(("bright", enhancer.enhance(1.2)))
    enhancer_ct = ImageEnhance.Contrast(img)
    base.append(("contrast", enhancer_ct.enhance(1.15)))
    variants.extend(base[: min(len(base), max(1, count))])
    i = 0
    while len(variants) < count:
        angle = random.uniform(-20, 20)
        b = random.uniform(0.85, 1.3)
        c = random.uniform(0.85, 1.3)
        tmp = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        tmp = ImageEnhance.Brightness(tmp).enhance(b)
        tmp = ImageEnhance.Contrast(tmp).enhance(c)
        if random.random() < 0.5:
            tmp = tmp.transpose(Image.FLIP_LEFT_RIGHT)
        variants.append((f"rnd{i}", tmp))
        i += 1
    return variants


def _s3_client():
    try:
        return boto3.client("s3")
    except Exception:
        return None


def _upload_files_to_s3(bucket: str, base_key: str, paths: List[Path]) -> List[str]:
    s3 = _s3_client()
    urls = []
    if not s3:
        return urls
    for p in paths:
        key = f"{base_key}/{p.name}"
        try:
            s3.upload_file(str(p), bucket, key)
            urls.append(f"s3://{bucket}/{key}")
        except (BotoCoreError, NoCredentialsError, ClientError, S3UploadFailedError, Exception):
            # best-effort; ignore upload failures
            pass
    return urls


def _tf_available():
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except Exception:
        return False


def _load_or_train_model(user_id: str, dataset_root: Path, model_path: Path, epochs: int = 3) -> Optional[object]:
    """
    If TensorFlow is available and dataset has >=2 class folders with >=5 imgs each,
    train a tiny CNN and save. Otherwise return None.
    """
    if not _tf_available():
        return None
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # Inspect dataset
    class_dirs = [d for d in (dataset_root / user_id).glob("*") if d.is_dir()]
    classes = [d.name for d in class_dirs if len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png"))) >= 5]
    if len(classes) < 2:
        return None

    img_size = (128, 128)
    batch_size = 16
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_root / user_id,
        image_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_root / user_id,
        image_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=42,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(classes)
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(img_size[0], img_size[1], 3)),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=max(1, int(epochs)), verbose=0)
    model.save(model_path)
    return model


def _predict_with_model(model, img: Image.Image, class_names: List[str]) -> Tuple[str, float]:
    import tensorflow as tf
    img_size = (128, 128)
    img_resized = img.resize(img_size)
    arr = tf.keras.preprocessing.image.img_to_array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return class_names[idx], float(preds[idx])


@image_router.post("/api/image/pipeline")
async def image_pipeline(
    mail: str = Form(...),
    file: UploadFile = File(...),
    auto: bool = Form(False),
    augment_count: int = Form(12),
    epochs: int = Form(3),
    class_label: str = Form("auto"),
):
    if file is None or not file.filename:
        raise HTTPException(400, "No image file provided")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
        raise HTTPException(400, "Only image files are supported for pipeline")

    content = await file.read()
    job_id = uuid.uuid4().hex
    user_id = "".join(ch if ch.isalnum() else "_" for ch in (mail or "user"))

    # Workspace folders
    job_dir = PIPELINE_DIR / user_id / job_id
    original_dir = job_dir / "original"
    aug_dir = job_dir / "augmented"
    # Normalize/sanitize class label for folder name
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (class_label or "auto")).strip() or "auto"
    dataset_user_dir = DATASETS_DIR / user_id / safe_label
    original_dir.mkdir(parents=True, exist_ok=True)
    aug_dir.mkdir(parents=True, exist_ok=True)
    dataset_user_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    orig_path = original_dir / file.filename
    _save_image_bytes_to_path(content, orig_path)

    # Decide effective augmentation count on the server (ignore too-small client values)
    try:
        server_min_aug = int(os.getenv("IMAGE_AUG_MIN", "24"))
    except Exception:
        server_min_aug = 24
    effective_aug_count = max(server_min_aug, int(augment_count or 0))

    # Augment and save
    img = _open_image(content)
    variants = _augment_image(img, count=effective_aug_count)
    saved_paths = []
    for suffix, im in variants:
        out_name = f"{Path(file.filename).stem}_{suffix}.jpg"
        out_path = aug_dir / out_name
        im.save(out_path, format="JPEG", quality=90)
        saved_paths.append(out_path)
        # also place into dataset class folder
        ds_path = dataset_user_dir / out_name
        _save_image_bytes_to_path(out_path.read_bytes(), ds_path)

    # Upload to S3 (best effort)
    bucket = os.getenv("AKKIO_S3_BUCKET") or os.getenv("AWS_S3_BUCKET") or "akkio-image-pipeline"
    s3_key_prefix = f"image-pipeline/{user_id}/{job_id}"
    s3_urls = _upload_files_to_s3(bucket, s3_key_prefix, saved_paths)

    # Train and predict if possible and requested
    model_path = MODELS_DIR / f"{user_id}_cnn.h5"
    trained = False
    prediction = None
    confidence = None
    if auto:
        model = _load_or_train_model(user_id=user_id, dataset_root=DATASETS_DIR, model_path=model_path, epochs=max(1, int(epochs)))
        if model is not None:
            trained = True
            # infer class names from directory listing
            class_dirs = sorted([d for d in (DATASETS_DIR / user_id).glob("*") if d.is_dir()])
            class_names = [d.name for d in class_dirs]
            try:
                pred_label, conf = _predict_with_model(model, img, class_names)
                prediction = pred_label
                confidence = conf
            except Exception:
                pass

    return JSONResponse(content={
        "status": "ok",
        "job_id": job_id,
        "trained": trained,
        "prediction": prediction,
        "confidence": confidence,
        "augment_count_requested": int(augment_count),
        "augment_count_effective": int(effective_aug_count),
        "epochs": int(epochs),
        "class_label": safe_label,
        "generated_count": len(saved_paths),
        "augmented_dir": str(aug_dir),
        "generated_files": [p.name for p in saved_paths],
        "s3_bucket": bucket,
        "s3_urls": s3_urls,
        "dataset_dir": str(DATASETS_DIR / user_id)
    })


@image_router.post("/api/image/predict")
async def image_predict(
    mail: str = Form(...),
    file: UploadFile = File(...)
):
    if not _tf_available():
        raise HTTPException(400, "Prediction unavailable: TensorFlow not installed on server")
    if file is None or not file.filename:
        raise HTTPException(400, "No image provided")
    content = await file.read()
    img = _open_image(content)
    user_id = "".join(ch if ch.isalnum() else "_" for ch in (mail or "user"))
    model_path = MODELS_DIR / f"{user_id}_cnn.h5"
    if not model_path.exists():
        raise HTTPException(404, "No trained model found for this user")

    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    class_dirs = sorted([d for d in (DATASETS_DIR / user_id).glob("*") if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    label, conf = _predict_with_model(model, img, class_names)
    return JSONResponse(content={"prediction": label, "confidence": conf})


