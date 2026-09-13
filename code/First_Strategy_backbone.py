from CNN_1 import *
import matplotlib.pyplot as plt

# First Strategy: pre-training on a larger dataset and fine-tuning on the target dataset
# First step: validation on the dataset AID


if __name__ == "__main__":
    data_path = "../AID"
    batch = 32
    seed = 1

    print("Training Set loading...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        label_mode="categorical",
        validation_split=0.3,
        subset="training",
        seed=seed,
        image_size=(224, 224),
        batch_size=batch,
    )

    print("Validation Set loading...")
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        label_mode="categorical",
        validation_split=0.3,
        subset="validation",
        seed=seed,
        image_size=(224, 224),
        batch_size=batch,
    )

    train_dataset = (
        train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    cnn = CNN_1(input_shape=(224, 224, 3), num_classes=30)
    cnn.summary()
    cnn.compile(learning_rate=0.001)

    history = cnn.train(train_dataset, val_dataset, epochs=30, patience=7)
    cnn.model.save_weights("weights/cnn_1_aid.weights.h5")

    train_loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, "b-", label="Training Loss")
    plt.plot(epochs, val_loss, "r-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
