import tensorflow as tf

# split of the target dataset

if __name__ == "__main__":
    new_data_path = "../UCMerced_JPG"
    batch = 32
    seed = 1

    print("Training Set loading...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        new_data_path,
        label_mode="categorical",
        validation_split=0.3,
        subset="training",
        seed=seed,
        image_size=(224, 224),
        batch_size=batch,
    )

    print("Validation Set loading...")
    val_and_test_dataset = tf.keras.utils.image_dataset_from_directory(
        new_data_path,
        label_mode="categorical",
        validation_split=0.3,
        subset="validation",
        seed=seed,
        image_size=(224, 224),
        batch_size=batch,
    )

    # number of batches in validation set (shuffle)
    n_batches = tf.data.experimental.cardinality(val_and_test_dataset)
    # Validation Set 15%
    val_dataset = val_and_test_dataset.take(n_batches // 2)
    # Test Set 15%
    test_dataset = val_and_test_dataset.skip(n_batches // 2)

    print(f"Training batches: {tf.data.experimental.cardinality(train_dataset)}")
    print(f"Validation batches: {tf.data.experimental.cardinality(val_dataset)}")
    print(f"Test batches: {tf.data.experimental.cardinality(test_dataset)}")

    train_dataset.save("../Dataset_UCMerced/train")
    val_dataset.save("../Dataset_UCMerced/val")
    test_dataset.save("../Dataset_UCMerced/test")
