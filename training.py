
import gc
import os
import json
import math
import random
import warnings

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks, backend as K
from tensorflow.keras.applications import DenseNet201, EfficientNetV2B2, ConvNeXtTiny

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc as sk_auc,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
from scipy.optimize import minimize

import xgboost as xgb

warnings.filterwarnings("ignore")

import kagglehub

# Download latest version
path = kagglehub.dataset_download("mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset")

print("Path to dataset files:", path)


SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)

IMG_SIZE      = (384, 384)
BATCH_SIZE    = 4
N_FOLDS       = 5
EPOCHS_HEAD   = 15       # Phase 1: head warmup (base frozen)
EPOCHS_FINE1  = 15       # Phase 2a: fine-tune top conv block
EPOCHS_FINE2  = 15       # Phase 2b: fine-tune deeper layers
LR_HEAD       = 1e-3
LR_FINE1      = 1e-4
LR_FINE2      = 5e-6
WARMUP_FRAC   = 0.10     # 10% of steps for linear LR warmup
MIXUP_ALPHA   = 0.2      # MixUp interpolation strength
TTA_STEPS     = 4
CATEGORIES    = ["Normal", "Osteopenia", "Osteoporosis"]
NUM_CLASSES   = len(CATEGORIES)
DATASET_ROOT  = (
    "/home/adarsha/.cache/kagglehub/datasets/"
    "mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset/"
    "versions/1/OS Collected Data" #Use the path where dataset is downloaded
)
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


rows = []
for cat in CATEGORIES:
    folder = os.path.join(DATASET_ROOT, cat)
    for f in os.listdir(folder):
        if f.lower().endswith(IMAGE_EXTS):
            rows.append({"filepath": os.path.join(folder, f), "label": cat})

df = pd.DataFrame(rows)
label2idx = {c: i for i, c in enumerate(CATEGORIES)}
df["label_idx"] = df["label"].map(label2idx)

print(f"Total images: {len(df)}")
print(df["label"].value_counts())

train_val_df, test_df = train_test_split(
    df, test_size=0.15, stratify=df["label_idx"], random_state=SEED
)
print(f"Train+Val: {len(train_val_df)}  |  Test: {len(test_df)}")

def apply_clahe(img_bgr: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """CLAHE in LAB colour space to boost local contrast."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def load_image(path: str, size: tuple = IMG_SIZE) -> np.ndarray:
    """Load → CLAHE → resize → RGB → float32 in [0, 255]."""
    img = cv2.imread(path)
    
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = apply_clahe(img)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)   


def tf_load_sample(path, label):
    img = tf.py_function(
        lambda p: load_image(p.numpy().decode()),
        [path], tf.float32,
    )
    img.set_shape([*IMG_SIZE, 3])
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label

# Training augmentation — heavier, applied to batches
train_aug = keras.Sequential([
    layers.RandomRotation(0.10),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomZoom(0.10),
    layers.RandomFlip("horizontal"),
    layers.RandomContrast(0.15),
], name="train_augmentation")

# TTA augmentation — lighter, used at test time
tta_aug = keras.Sequential([
    layers.RandomRotation(0.07),
    layers.RandomTranslation(0.04, 0.04),
    layers.RandomZoom(0.07),
    layers.RandomFlip("horizontal"),
], name="tta_augmentation")


def mixup_batch(images, labels, sample_weights, alpha=MIXUP_ALPHA):
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform([], alpha, 1.0 - alpha)
    idx = tf.random.shuffle(tf.range(batch_size))
    mixed_x  = lam * images         + (1 - lam) * tf.gather(images, idx)
    mixed_y  = lam * labels         + (1 - lam) * tf.gather(labels, idx)
    mixed_sw = lam * sample_weights + (1 - lam) * tf.gather(sample_weights, idx)
    return mixed_x, mixed_y, mixed_sw


def build_dataset(
    filepaths, labels, training: bool,
    class_weights: dict = None,
    batch: int = BATCH_SIZE,
    use_mixup: bool = False,
):
    ds = tf.data.Dataset.from_tensor_slices(
        (filepaths, labels.astype(np.int32))
    )
    ds = ds.map(tf_load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=1024, seed=SEED)

    ds = ds.batch(batch) 

    if class_weights is not None:
        cw_tensor = tf.constant(
            [class_weights.get(i, 1.0) for i in range(NUM_CLASSES)],
            dtype=tf.float32,
        )
        # Derive per-sample weight from the one-hot label
        ds = ds.map(
            lambda x, y: (x, y, tf.linalg.matvec(y, cw_tensor)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        # Add a dummy weight of 1.0 so all downstream functions see 3-tuples
        ds = ds.map(
            lambda x, y: (x, y, tf.ones(tf.shape(y)[0])),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if training:
        ds = ds.map(
            lambda x, y, sw: (train_aug(x, training=True), y, sw),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if use_mixup:
            ds = ds.map(
                lambda x, y, sw: mixup_batch(x, y, sw),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

    return ds.prefetch(tf.data.AUTOTUNE)

# FOCAL LOSS

@tf.keras.utils.register_keras_serializable(package="NutriBone")
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.05,
                 reduction="sum_over_batch_size", name="focal_loss", **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = y_true * (1.0 - self.label_smoothing) + \
                 (self.label_smoothing / NUM_CLASSES)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce     = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha,
                    "label_smoothing": self.label_smoothing})
        return cfg


@tf.keras.utils.register_keras_serializable(package="NutriBone")   # ← FIX ①
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup for `warmup_steps`, then cosine decay to `min_lr`."""
    def __init__(self, peak_lr, total_steps, warmup_steps, min_lr=1e-9, **kwargs):
        super().__init__(**kwargs)                                  # ← FIX ②
        self.peak_lr      = float(peak_lr)
        self.total_steps  = float(total_steps)
        self.warmup_steps = float(warmup_steps)
        self.min_lr       = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * (step / self.warmup_steps)
        progress  = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress)
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps, "min_lr": self.min_lr}


def make_cosine_schedule(lr, n_samples, n_epochs):
    """Helper: returns a WarmupCosineDecay for given dataset size."""
    total  = math.ceil(n_samples / BATCH_SIZE) * n_epochs
    warmup = int(total * WARMUP_FRAC)
    return WarmupCosineDecay(lr, total, warmup)


@tf.keras.utils.register_keras_serializable(package="NutriBone")
class EfficientNetV2Preprocess(layers.Layer):
    """Rescales [0, 255] → [-1, 1] as expected by EfficientNetV2."""
    def call(self, x):
        return tf.keras.applications.efficientnet_v2.preprocess_input(x)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="NutriBone")
class DenseNetPreprocess(layers.Layer):
    """Subtracts ImageNet BGR channel means; used by DenseNet201."""
    def call(self, x):
        return tf.keras.applications.densenet.preprocess_input(x)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="NutriBone")
class ConvNeXtPreprocess(layers.Layer):
    """Normalises with ImageNet mean/std as expected by ConvNeXtTiny."""
    def call(self, x):
        return tf.keras.applications.convnext.preprocess_input(x)

    def get_config(self):
        return super().get_config()


# Shared custom_objects
CUSTOM_OBJECTS = {
    "FocalLoss":                FocalLoss,
    "WarmupCosineDecay":        WarmupCosineDecay,
    "WarmupCosine":             WarmupCosineDecay, 
    "EfficientNetV2Preprocess": EfficientNetV2Preprocess,
    "DenseNetPreprocess":       DenseNetPreprocess,
    "ConvNeXtPreprocess":       ConvNeXtPreprocess,
}


def safe_load_model(path: str, backbone_name: str) -> "keras.Model":
    """
    Load a .keras file safely regardless of how it was originally saved.

    Pass 1 — normal keras.models.load_model with CUSTOM_OBJECTS.
              Works for all models saved with the fixed training script.

    Pass 2 — weight-transplant fallback for old Lambda-based files:
              rebuilds the correct registered architecture (no imagenet
              weights), opens the .keras ZIP, extracts model.weights.h5,
              and calls load_weights() by position.  Preprocessing layers
              carry zero parameters, so the tensor order is identical.
    """
    import zipfile
    import tempfile

    # Pass 1
    try:
        m = keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)
        print(f"    load OK  (normal)    {os.path.basename(path)}")
        return m
    except (TypeError, KeyError, ValueError) as exc:
        print(f"    normal load failed ({type(exc).__name__}) -> weight-transplant ...")

    # Pass 2 : weight transplant
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {backbone_name!r}")

    model, _ = BACKBONE_REGISTRY[backbone_name](trainable_base=True)

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            wf = next((n for n in names if n.endswith(".weights.h5")), None)
            if wf is None:
                raise FileNotFoundError(
                    f"No weights file inside {path}. Contents: {names}"
                )
            zf.extract(wf, tmp)
            wpath = os.path.join(tmp, wf)
        model.load_weights(wpath)

    print(f"    load OK  (transplant) {os.path.basename(path)}")
    return model

def _classification_head(x):
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    return layers.Dense(NUM_CLASSES, activation="softmax")(x)


def build_efficientnet(trainable_base=False):
    inp  = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x    = EfficientNetV2Preprocess(name="eff_preprocess")(inp)
    base = EfficientNetV2B2(
        weights="imagenet", include_top=False,
        input_shape=(*IMG_SIZE, 3),
        include_preprocessing=False,   # handled by EfficientNetV2Preprocess above
    )
    base.trainable = trainable_base
    x   = base(x)
    out = _classification_head(x)
    return Model(inp, out, name="EfficientNetV2B2"), base


def build_densenet(trainable_base=False):
    inp  = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x    = DenseNetPreprocess(name="dense_preprocess")(inp)
    base = DenseNet201(
        weights="imagenet", include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = trainable_base
    x   = base(x)
    out = _classification_head(x)
    return Model(inp, out, name="DenseNet201"), base


def build_convnext(trainable_base=False):
    inp  = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x    = ConvNeXtPreprocess(name="convnext_preprocess")(inp)
    base = ConvNeXtTiny(
        weights="imagenet", include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = trainable_base
    x   = base(x)
    out = _classification_head(x)
    return Model(inp, out, name="ConvNeXtTiny"), base


BACKBONE_REGISTRY = {
    "EfficientNetV2B2": build_efficientnet,
    "DenseNet201":      build_densenet,
    "ConvNeXtTiny":     build_convnext,
}

FINETUNE_PHASE1 = {
    "EfficientNetV2B2": lambda n: "block6" in n or "top" in n,
    "DenseNet201": lambda n: "conv5" in n,
    "ConvNeXtTiny":     lambda n: "stage_3" in n,
}
FINETUNE_PHASE2 = {
    "EfficientNetV2B2": lambda n: "block5" in n or "block6" in n or "top" in n,
    "DenseNet201":      lambda n: "conv5" in n,
    "ConvNeXtTiny":     lambda n: "stage_2" in n or "stage_3" in n,
}

# CALLBACKS

def make_callbacks(save_path: str):
    return [
        callbacks.ModelCheckpoint(
            save_path, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=8,
            restore_best_weights=True, verbose=1,
        ),
        # NOTE: ReduceLROnPlateau is removed for fine-tune phases
        # because cosine schedule already handles LR decay.
        # Keep it only for head warmup (where we use Adam with fixed LR).
    ]

def make_callbacks_with_reducelr(save_path: str):
    return make_callbacks(save_path) + [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-9, verbose=1,
        )
    ]

# TRAINING — SINGLE BACKBONE, ONE FOLD

def train_backbone_fold(
    backbone_name: str, fold_idx: int,
    train_fps, train_lbs,
    val_fps,   val_lbs,
    class_weights: dict,
):
    n_train = len(train_fps)
    print(f"\n{'='*60}")
    print(f"  {backbone_name}  |  Fold {fold_idx+1}/{N_FOLDS}")
    print(f"{'='*60}")
    save_path = f"{backbone_name}_fold{fold_idx+1}_best.keras"

    model, base = BACKBONE_REGISTRY[backbone_name](trainable_base=False)

    train_ds = build_dataset(train_fps, train_lbs, training=True,
                             class_weights=class_weights, use_mixup=True)
    val_ds   = build_dataset(val_fps, val_lbs, training=False)

    # Phase 1: Head warmup
    print("\n[Phase 1] Head Warmup — base frozen")
    model.compile(
        optimizer=keras.optimizers.Adam(LR_HEAD),
        loss=FocalLoss(),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=make_callbacks_with_reducelr(save_path),
        verbose=1,
    )

    # Phase 2a: Fine-tune top block
    print("\n[Phase 2a] Fine-Tuning — top conv block only")
    ft1 = FINETUNE_PHASE1[backbone_name]
    for layer in base.layers:
        layer.trainable = ft1(layer.name)

    schedule_fine1 = make_cosine_schedule(LR_FINE1, n_train, EPOCHS_FINE1)
    model.compile(
        optimizer=keras.optimizers.Adam(schedule_fine1),
        loss=FocalLoss(),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_FINE1,
        callbacks=make_callbacks(save_path),
        verbose=1,
    )

    # Phase 2b: Fine-tune deeper block
    print("\n[Phase 2b] Fine-Tuning — deeper conv blocks")
    ft2 = FINETUNE_PHASE2[backbone_name]
    for layer in base.layers:
        layer.trainable = ft2(layer.name)

    schedule_fine2 = make_cosine_schedule(LR_FINE2, n_train, EPOCHS_FINE2)
    model.compile(
        optimizer=keras.optimizers.Adam(schedule_fine2),
        loss=FocalLoss(),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_FINE2,
        callbacks=make_callbacks(save_path),
        verbose=1,
    )

    return save_path

skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
tv_fps = train_val_df["filepath"].values
tv_lbs = train_val_df["label_idx"].values

fold_model_paths = {name: [] for name in BACKBONE_REGISTRY}
fold_val_probs   = {name: np.zeros((len(train_val_df), NUM_CLASSES))
                    for name in BACKBONE_REGISTRY}
fold_val_auc     = {name: [] for name in BACKBONE_REGISTRY}   # ⑦ track per-fold AUC

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(tv_fps, tv_lbs)):
    cw_arr = compute_class_weight(
        "balanced",
        classes=np.arange(NUM_CLASSES),
        y=tv_lbs[tr_idx]
    )
    cw_dict = dict(enumerate(cw_arr))

    for backbone_name in BACKBONE_REGISTRY:

        model_path = f"{backbone_name}_fold{fold_idx + 1}_best.keras"

        if os.path.exists(model_path):
            print(f"\n✓ Reusing existing model: {model_path}")
            path = model_path
        else:
            path = train_backbone_fold(
                backbone_name, fold_idx,
                tv_fps[tr_idx], tv_lbs[tr_idx],
                tv_fps[va_idx], tv_lbs[va_idx],
                cw_dict,
            )

        fold_model_paths[backbone_name].append(path)

        # ⑨ Reload best checkpoint, collect OOF probs, then free memory
        m = safe_load_model(path, backbone_name)
        val_ds = build_dataset(tv_fps[va_idx], tv_lbs[va_idx], training=False)
        raw_preds = m.predict(val_ds, verbose=0)

        fold_val_probs[backbone_name][va_idx] = raw_preds

        y_bin = label_binarize(
            tv_lbs[va_idx],
            classes=list(range(NUM_CLASSES))
        )

        aucs = [
            sk_auc(*roc_curve(y_bin[:, i], raw_preds[:, i])[:2])
            for i in range(NUM_CLASSES)
        ]
        fold_val_auc[backbone_name].append(float(np.mean(aucs)))
        print(f"  OOF AUC [{backbone_name} fold {fold_idx+1}]: {aucs}")

        del m
        gc.collect()
        # Do NOT call K.clear_session() here — it would reset layer name counters

print("\n5-Fold CV complete.")

#  MEMORY-SAFE BATCHED TTA

def predict_tta(model, filepaths, labels, steps=TTA_STEPS):
    """Batched TTA: `steps` augmented passes, averaged probabilities."""
    # Build a plain (x, y) dataset so model.predict doesn't see sw
    base_ds = (
        tf.data.Dataset.from_tensor_slices(
            (filepaths, labels.astype(np.int32))
        )
        .map(tf_load_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    n           = len(filepaths)
    accumulated = np.zeros((n, NUM_CLASSES), dtype=np.float32)

    for step in range(steps):
        offset = 0
        for batch_x, _ in base_ds:
            # Apply TTA augmentation on 4D batch tensor
            aug_x = tta_aug(batch_x, training=True)
            probs = model(aug_x, training=False).numpy()
            bs    = probs.shape[0]
            accumulated[offset: offset + bs] += probs
            offset += bs

    return accumulated / steps

# COLLECT TEST PROBABILITIES (per backbone)

test_fps = test_df["filepath"].values
test_lbs = test_df["label_idx"].values

backbone_test_probs = {}

for backbone_name, paths in fold_model_paths.items():
    fold_probs = []
    for path in paths:
        m = safe_load_model(path, backbone_name)
        probs = predict_tta(m, test_fps, test_lbs, steps=TTA_STEPS)
        fold_probs.append(probs)
        del m
        gc.collect()
    backbone_test_probs[backbone_name] = np.mean(fold_probs, axis=0)

print("\nTest probabilities collected for all backbones.")

#  TEMPERATURE SCALING

def nll_with_temp(T, logits, labels_oh):
    scaled = softmax(logits / T, axis=1)
    return -np.mean(np.sum(labels_oh * np.log(scaled + 1e-8), axis=1))


def fit_temperature(oof_probs, true_labels):
    logits    = np.log(np.clip(oof_probs, 1e-8, 1.0))
    labels_oh = np.eye(NUM_CLASSES)[true_labels]
    result    = minimize(
        nll_with_temp, x0=[1.5], args=(logits, labels_oh),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 1000},
    )
    return float(result.x[0])


temperatures       = {}
scaled_test_probs  = {}

for backbone_name in BACKBONE_REGISTRY:
    T = fit_temperature(fold_val_probs[backbone_name], tv_lbs)
    temperatures[backbone_name] = T
    print(f"  Temperature [{backbone_name}]: {T:.4f}")

    logits = np.log(np.clip(backbone_test_probs[backbone_name], 1e-8, 1.0))
    scaled_test_probs[backbone_name] = softmax(logits / T, axis=1)

print("\nTemperature scaling complete.")

#  WEIGHTED ENSEMBLE

mean_oof_auc = {
    name: float(np.mean(fold_val_auc[name]))
    for name in BACKBONE_REGISTRY
}
print("\nMean OOF AUC per backbone:")
for name, score in mean_oof_auc.items():
    print(f"  {name}: {score:.4f}")

# Normalise weights so they sum to 1
total_auc   = sum(mean_oof_auc.values())
ens_weights = {name: v / total_auc for name, v in mean_oof_auc.items()}
print("\nEnsemble weights:")
for name, w in ens_weights.items():
    print(f"  {name}: {w:.4f}")

ensemble_probs = sum(
    ens_weights[name] * scaled_test_probs[name]
    for name in BACKBONE_REGISTRY
)

y_pred_ensemble = np.argmax(ensemble_probs, axis=1)

print("\n─── Weighted Ensemble Classification Report ───")
print(classification_report(test_lbs, y_pred_ensemble, target_names=CATEGORIES))

#  XGBOOST META-LEARNER

print("\n─── XGBoost on Backbone Features ───")

xgb_train_feats = np.hstack([fold_val_probs[b] for b in BACKBONE_REGISTRY])
xgb_test_feats  = np.hstack([backbone_test_probs[b] for b in BACKBONE_REGISTRY])

# Small split from OOF data for XGBoost early stopping
xtr_f, xva_f, ytr, yva = train_test_split(
    xgb_train_feats, tv_lbs, test_size=0.15,
    stratify=tv_lbs, random_state=SEED,
)

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    early_stopping_rounds=30,   # prevents overfitting on small feature sets
    random_state=SEED,
    n_jobs=-1,
)
xgb_model.fit(xtr_f, ytr, eval_set=[(xva_f, yva)], verbose=False)

xgb_preds = xgb_model.predict(xgb_test_feats)
print(classification_report(test_lbs, xgb_preds, target_names=CATEGORIES))
xgb_model.save_model("nutribone_xgb.json")
print("XGBoost model saved → nutribone_xgb.json")


def plot_confusion(y_true, y_pred, title_prefix="Ensemble"):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
        axes, [cm, cm_pct], ["d", ".2%"],
        [f"{title_prefix} — Counts", f"{title_prefix} — Row %"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(f"confusion_{title_prefix.lower().replace(' ','_')}.png", dpi=150)
    plt.show()


def plot_roc(y_true, probs, title_prefix="Ensemble"):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    plt.figure(figsize=(8, 6))
    macro_aucs = []
    for i, cls in enumerate(CATEGORIES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        score = sk_auc(fpr, tpr)
        macro_aucs.append(score)
        plt.plot(fpr, tpr, label=f"{cls}  AUC={score:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"{title_prefix} ROC  |  Macro AUC={np.mean(macro_aucs):.3f}")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"roc_{title_prefix.lower().replace(' ','_')}.png", dpi=150)
    plt.show()
    return float(np.mean(macro_aucs))


plot_confusion(test_lbs, y_pred_ensemble, "Weighted Ensemble")
macro_auc = plot_roc(test_lbs, ensemble_probs, "Weighted Ensemble")
print(f"\nFinal Macro AUC: {macro_auc:.4f}")

#   GRAD-CAM

GRADCAM_LAYERS = {
    "EfficientNetV2B2": "top_activation",
    "DenseNet201":      "relu",
    "ConvNeXtTiny":     "convnext_tiny_stage_3_block_2_depthwise_conv",
}


def get_gradcam(model, img_array, layer_name):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, tf.argmax(preds[0])]
    grads   = tape.gradient(loss, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def visualize_gradcam(image_path, model, layer_name):
    img_orig    = cv2.imread(image_path)
    img_clahe   = apply_clahe(img_orig)
    img_resized = cv2.resize(img_clahe, IMG_SIZE)
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    arr     = np.expand_dims(img_rgb.astype(np.float32), axis=0)
    heatmap = cv2.resize(get_gradcam(model, arr, layer_name), IMG_SIZE)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, title in zip(
        axes,
        [img_rgb, heatmap_color[:, :, ::-1], overlay[:, :, ::-1]],
        ["Original (CLAHE)", "Grad-CAM Heatmap", "Overlay"],
    ):
        ax.imshow(data); ax.set_title(title); ax.axis("off")
    plt.tight_layout(); plt.show()

#  SINGLE IMAGE INFERENCE

def predict_xray(image_path: str, tta: bool = True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    arr = np.expand_dims(load_image(image_path), axis=0)  # [1, H, W, 3] in [0,255]

    probs_per_backbone = []
    for bname in BACKBONE_REGISTRY:
        m = safe_load_model(fold_model_paths[bname][0], bname)
        T = temperatures[bname]

        if tta:
            acc = np.zeros((1, NUM_CLASSES), dtype=np.float32)
            for _ in range(TTA_STEPS):
                aug   = tta_aug(arr, training=True).numpy()
                acc  += m.predict(aug, verbose=0)
            raw_prob = acc / TTA_STEPS
        else:
            raw_prob = m.predict(arr, verbose=0)

        logits = np.log(np.clip(raw_prob, 1e-8, 1.0))
        scaled = softmax(logits / T, axis=1)
        # Apply backbone's ensemble weight
        probs_per_backbone.append(ens_weights[bname] * scaled)
        del m
        gc.collect()

    ensemble = np.sum(probs_per_backbone, axis=0)[0]
    idx      = int(np.argmax(ensemble))
    label    = CATEGORIES[idx]

    print(f"\nPrediction : {label}")
    print(f"Confidence : {ensemble[idx]:.4f}")
    print("\nAll Scores:")
    for cls, p in zip(CATEGORIES, ensemble):
        print(f"  {cls:<15}: {p:.4f}")

    return label, dict(zip(CATEGORIES, ensemble.tolist()))

# SAVE CONFIG

config = {
    "img_size":       list(IMG_SIZE),
    "categories":     CATEGORIES,
    "temperatures":   temperatures,
    "ensemble_weights": ens_weights,
    "oof_auc":        mean_oof_auc,
    "macro_auc":      round(macro_auc, 4),
    "fold_models":    fold_model_paths,
    "xgb_model":      "nutribone_xgb.json",
}

with open("nutribone_config.json", "w") as f:
    json.dump(config, f, indent=4)

print("\nConfig saved → nutribone_config.json")
print("\n✓ All done. Refined pipeline complete.")
