"""Training helpers for the emotion classifier."""

from pathlib import Path


def train_model(model, x_train, y_train, x_validation, y_validation, *, epochs=30, batch_size=32, checkpoint=None):
    """Train a compiled model and optionally save its best validation checkpoint."""
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    callbacks = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True))
    return model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=epochs, batch_size=batch_size, callbacks=callbacks)