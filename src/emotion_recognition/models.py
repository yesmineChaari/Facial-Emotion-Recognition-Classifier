"""Neural classifier definitions for tabular OpenFace features."""


def build_dense_classifier(input_features: int, class_count: int, learning_rate: float = 0.001):
    """Build and compile the dense classifier used for OpenFace features."""
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to build the neural classifier") from exc

    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_features,)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(class_count, activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model