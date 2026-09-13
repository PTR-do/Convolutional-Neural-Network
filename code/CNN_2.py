import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses, callbacks
import numpy as np

# Model for second strategy train


class CNN_2:
    def __init__(self, input_shape=(224, 224, 3), num_classes=21):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _residual_block(self, x, filters, enable_stride=False):
        stride = 2 if enable_stride else 1
        # --- main branch ---
        main_branch = layers.Conv2D(filters, (3, 3), strides=stride, padding="same")(x)
        main_branch = layers.BatchNormalization()(main_branch)
        main_branch = layers.Activation("relu")(main_branch)
        main_branch = layers.Dropout(0.1)(main_branch)  # Dropout against overfitting
        main_branch = layers.Conv2D(filters, (3, 3), strides=1, padding="same")(
            main_branch
        )
        main_branch = layers.BatchNormalization()(main_branch)
        # --- skip connection ---
        skip_connection = x
        if enable_stride or x.shape[-1] != filters:
            skip_connection = layers.Conv2D(
                filters, (1, 1), strides=stride, padding="same"
            )(x)
            skip_connection = layers.BatchNormalization()(skip_connection)
        # --- final sum ---
        sum = layers.Add()([main_branch, skip_connection])
        return layers.Activation("relu")(sum)

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        data_augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.5),
                layers.RandomZoom(0.3),
                layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
                layers.RandomContrast(0.2),
            ],
            name="Data_Augmentation",
        )
        x = data_augmentation(inputs)

        x = layers.Rescaling(1.0 / 255)(x)

        x = layers.Conv2D(32, (5, 5), strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

        x = self._residual_block(x, filters=64, enable_stride=True)
        x = self._residual_block(x, filters=128, enable_stride=True)
        x = self._residual_block(x, filters=256, enable_stride=True)
        x = self._residual_block(x, filters=512, enable_stride=True)

        mlp_classifier = tf.keras.Sequential(
            [
                layers.GlobalAveragePooling2D(name="global_pooling"),
                # Fully Connected 1
                layers.Dense(256),
                layers.Activation("relu"),
                layers.Dropout(0.3),
                # Fully Connected 2
                layers.Dense(128),
                layers.Activation("relu"),
                # Output Layer
                layers.Dense(
                    self.num_classes, activation="softmax", name="predictions"
                ),
            ],
            name="MLP_Head",
        )

        outputs = mlp_classifier(x)
        return Model(inputs=inputs, outputs=outputs, name="CNN")

    def compile(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        loss_fn = losses.CategoricalCrossentropy(from_logits=False)
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[tf.keras.metrics.F1Score(average="macro", name="f1_macro")],
        )

    def train(self, train_dataset, val_dataset, epochs=50, patience=4):
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=2, min_lr=1e-6
        )
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[early_stop, reduce_lr],
        )
        return history

    def summary(self):
        self.model.summary()

    def predict_proba(self, inputs):
        return self.model.predict(inputs)

    def predict(self, inputs):
        probabilities = self.predict_proba(inputs)
        predict_classes = np.argmax(probabilities, axis=1)
        return predict_classes
