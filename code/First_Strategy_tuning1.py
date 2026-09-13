from CNN_1 import *
import matplotlib.pyplot as plt

# First Strategy: pre-training on a larger dataset and fine-tuning on the target dataset
# Second step: fine tuning on the dataset UCMerced_LandUse


if __name__ == "__main__":
    train_dataset = tf.data.Dataset.load("../Dataset_UCMerced/train")
    val_dataset = tf.data.Dataset.load("../Dataset_UCMerced/val")
    train_dataset = (
        train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    cnn = CNN_1(input_shape=(224, 224, 3), num_classes=21)
    # Weights loading, last MLP layer is not loaded
    cnn.model.load_weights("weights/cnn_1_val.weights.h5", skip_mismatch=True)
    cnn.compile(learning_rate=0.0001)

    # complete fine tuning
    history = cnn.train(train_dataset, val_dataset, epochs=50, patience=5)
    cnn.model.save_weights("weights/cnn_1_finetuning1.weights.h5")

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
