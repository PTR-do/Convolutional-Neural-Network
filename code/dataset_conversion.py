import os
import cv2

# Format conversion of the images of target dataset

dataset_path = "../UCMerced_LandUse/Images"
output_path = "../UCMerced_JPG"

os.makedirs(output_path, exist_ok=True)

print("Start conversion:")
class_folders = os.listdir(dataset_path)

for class_name in class_folders:
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"{class_name}")
    out_class_path = os.path.join(output_path, class_name)
    os.makedirs(out_class_path, exist_ok=True)
    images = os.listdir(class_path)
    for img_name in images:
        if img_name.lower().endswith(".tif"):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                new_img_name = img_name[:-4] + ".jpg"
                new_img_path = os.path.join(out_class_path, new_img_name)
                cv2.imwrite(new_img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                print(f"Error in image: {img_name}")

print("Conversion completed.")
