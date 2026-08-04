import os
import json
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import config
from src.preprocessing import create_data_generators, compute_balanced_weights

def build_model(num_classes):
    """Builds a fresh model with a frozen EfficientNetB0 backbone (Phase 1)."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*config.IMAGE_SIZE, 3))
    base_model.trainable = False  # Transfer learning frozen backbone

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model

def unfreeze_model_for_fine_tuning(model, unfreeze_layers=30):
    """Unfreezes the top N layers of the inner EfficientNetB0 backbone (Phase 2)."""
    print(f"\n--- Phase 2: Unfreezing top {unfreeze_layers} layers for fine-tuning ---")
    
    # 1. Locate the inner base model (EfficientNetB0)
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model:
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - unfreeze_layers
        
        # Freeze early layers, unfreeze top N layers
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True

        print(f"Base model total layers: {len(base_model.layers)}")
        print(f"Base model layers frozen: {fine_tune_at} | Unfrozen: {unfreeze_layers}")
    else:
        # Fallback if architecture is flat
        model.trainable = True
        fine_tune_at = len(model.layers) - unfreeze_layers
        for layer in model.layers[:fine_tune_at]:
            layer.trainable = False

    # 2. Recompile with lower fine-tuning learning rate
    fine_tune_lr = getattr(config, 'FINE_TUNE_LEARNING_RATE', 1e-5)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def save_metrics_to_json(history, output_dir="outputs", file_name="metrics.json", reset=False):
    """Saves training metrics to outputs/metrics.json."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, file_name)

    new_metrics = {k: [float(val) for val in v] for k, v in history.history.items()}

    if not reset and os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                existing_data = json.load(f)
            
            for metric, values in new_metrics.items():
                if metric in existing_data and isinstance(existing_data[metric], list):
                    existing_data[metric].extend(values)
                else:
                    existing_data[metric] = values
            data_to_save = existing_data
        except Exception as e:
            print(f"Warning: Could not merge with existing {json_path} ({e}). Writing new metrics.")
            data_to_save = new_metrics
    else:
        data_to_save = new_metrics

    with open(json_path, "w") as f:
        json.dump(data_to_save, f, indent=4)
        
    print(f"Metrics successfully saved to {json_path}")

def run_training(fresh_start=False):
    train_gen, val_gen, _ = create_data_generators()
    weights = compute_balanced_weights(train_gen)
    
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    save_path = config.MODEL_SAVE_DIR / "skin_disease_model.keras"
    csv_log_path = config.MODEL_SAVE_DIR / "training_log.csv"

    if fresh_start:
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"Removed old model at {save_path}")
        if os.path.exists(csv_log_path):
            os.remove(csv_log_path)
            print(f"Removed old CSV log at {csv_log_path}")

        print("\n--- Starting Phase 1: Training Fresh Model (Frozen Backbone) ---")
        model, _ = build_model(num_classes=train_gen.num_classes)
        append_csv = False
    else:
        print("\n--- Loading Saved Model to Continue ---")
        if os.path.exists(save_path):
            model = tf.keras.models.load_model(save_path)
            # CALL PHASE 2 UNFREEZE HERE
            model = unfreeze_model_for_fine_tuning(model, unfreeze_layers=30)
            append_csv = True
        else:
            print("No saved model found. Building fresh Phase 1 model.")
            model, _ = build_model(num_classes=train_gen.num_classes)
            append_csv = False

    callbacks = [
        ModelCheckpoint(str(save_path), save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        CSVLogger(str(csv_log_path), append=append_csv)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        class_weight=weights,
        callbacks=callbacks
    )

    save_metrics_to_json(history, output_dir="outputs", file_name="metrics.json", reset=fresh_start)

    return model, history

if __name__ == "__main__":
    run_training(fresh_start=False)