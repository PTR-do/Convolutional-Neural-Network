from CNN_2 import *
import matplotlib.pyplot as plt

# Second strategy: complete training only on the target dataset UCMerced_LandUse


if __name__ == "__main__":
    train_dataset = tf.data.Dataset.load("../Dataset_UCMerced/train")
    val_dataset = tf.data.Dataset.load("../Dataset_UCMerced/val")
    train_dataset = (
        train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    cnn = CNN_2(input_shape=(224, 224, 3), num_classes=21)
    cnn.compile(learning_rate=0.001)
    cnn.summary()

    history = cnn.train(train_dataset, val_dataset, epochs=30, patience=5)
    cnn.model.save_weights("weights/cnn_2_ucmerced.weights.h5")

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
