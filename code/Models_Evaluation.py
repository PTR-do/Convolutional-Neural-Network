from CNN_1 import *
from CNN_2 import *
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluation of the three model on validation set and final prediction on test set

if __name__ == "__main__":
    val_dataset = tf.data.Dataset.load("../Dataset_UCMerced/val")
    test_dataset = tf.data.Dataset.load("../Dataset_UCMerced/test")
    val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    cnn1_1tune = CNN_1(input_shape=(224, 224, 3), num_classes=21)
    cnn1_1tune.model.load_weights("weights/cnn_1_finetuning1.weights.h5")

    cnn1_2tune = CNN_1(input_shape=(224, 224, 3), num_classes=21)
    cnn1_2tune.model.load_weights("weights/cnn_1_finetuning2.weights.h5")

    cnn2 = CNN_2(input_shape=(224, 224, 3), num_classes=21)
    cnn2.model.load_weights("weights/cnn_2_ucmerced.weights.h5")

    one_hot_labels = np.concatenate([y for x, y in val_dataset], axis=0)
    y_val_true = np.argmax(one_hot_labels, axis=1)

    val1_1tune = cnn1_1tune.predict(val_dataset)
    val1_2tune = cnn1_2tune.predict(val_dataset)
    val2 = cnn2.predict(val_dataset)
    f1_cnn1_1tune = f1_score(y_val_true, val1_1tune, average="macro")
    f1_cnn1_2tune = f1_score(y_val_true, val1_2tune, average="macro")
    f1_cnn2 = f1_score(y_val_true, val2, average="macro")
    print(f"CNN_1 first tune validation F1 score: {f1_cnn1_1tune}")
    print(f"CNN_1 second tune validation F1 score: {f1_cnn1_2tune}")
    print(f"CNN_2 validation F1 score: {f1_cnn2}")

    # test prediction with cnn_1 with fine tuning 2
    test_one_hot = np.concatenate([y for x, y in test_dataset], axis=0)
    y_test_true = np.argmax(test_one_hot, axis=1)
    y_test_pred = cnn1_2tune.predict(test_dataset)

    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    test_precision = precision_score(y_test_true, y_test_pred, average="macro")
    test_recall = recall_score(y_test_true, y_test_pred, average="macro")
    test_f1 = f1_score(y_test_true, y_test_pred, average="macro")
    conf_matrix = confusion_matrix(y_test_true, y_test_pred)

    class_names = [
        "agricultural",
        "airplane",
        "baseballdiamond",
        "beach",
        "buildings",
        "chaparral",
        "denseresidential",
        "forest",
        "freeway",
        "golfcourse",
        "harbor",
        "intersection",
        "mediumresidential",
        "mobilehomepark",
        "overpass",
        "parkinglot",
        "river",
        "runway",
        "sparseresidential",
        "storagetanks",
        "tenniscourt",
    ]

    plt.figure(figsize=(14, 14))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix CNN_1 (Test Set)", fontsize=16, pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    metrics = (
        f"TEST SET PERFORMANCE\n"
        f"{'-'*25}\n"
        f"Accuracy:  {test_accuracy:.4f}\n"
        f"Precision: {test_precision:.4f} (Macro)\n"
        f"Recall:    {test_recall:.4f} (Macro)\n"
        f"F1-Score:  {test_f1:.4f} (Macro)"
    )
    plt.figtext(
        0.70,
        0.05,
        metrics,
        fontsize=12,
        family="monospace",
        bbox=dict(
            facecolor="white", alpha=0.9, edgecolor="black", boxstyle="round,pad=0.8"
        ),
    )
    plt.subplots_adjust(bottom=0.25)
    nome_file_grafico = "Final_Results.png"
    plt.savefig(nome_file_grafico, dpi=300, bbox_inches="tight")
    plt.show()
