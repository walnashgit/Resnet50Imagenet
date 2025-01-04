# create annotations file
import os
import csv
import json
# from util import annotation_file, train_dir, class_index_file, val_mapping_file, val_annotation_file, val_dir
from config import CONFIG

# Initialize the class-to-index dictionary
class_to_index = {}

# Data paths from config
annotation_file = CONFIG["data_annotation_file"]["train"]
train_dir = CONFIG["train_dir"]
class_index_file = CONFIG["class_index_file"]
val_mapping_file = CONFIG["val_mapping_file"]
val_annotation_file = CONFIG["data_annotation_file"]["val"]
val_dir = CONFIG["val_dir"]



# Create a CSV file
def create_train_annotation():
    with open(annotation_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header - Not needed
        # writer.writerow(["image_path", "index"])

        # Walk through each class folder
        for class_index, class_name in enumerate(sorted(os.listdir(train_dir))):
            class_path = os.path.join(train_dir, class_name)
            if os.path.isdir(class_path):  # Check if it's a directory
                # Add the class name and index to the dictionary
                class_to_index[class_name] = class_index

                # Iterate over each image in the class folder
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    # Write image metadata to the CSV
                    writer.writerow([image_path, class_index])

    # Save the class-to-index dictionary to a JSON file
    with open(class_index_file, mode='w') as json_file:
        json.dump(class_to_index, json_file, indent=4)

    print(f"Metadata CSV file saved to {annotation_file}")
    print(f"Class-to-index dictionary saved to {class_index_file}")


def create_val_annotation():
    # Load the class-to-index dictionary
    with open(class_index_file, mode='r') as json_file:
        class_to_index = json.load(json_file)

    # Parse the mapping file and create the validation annotations
    with open(val_mapping_file, mode='r') as mapping_file, open(val_annotation_file, mode='w',
                                                                newline='') as output_file:
        mapping_reader = csv.reader(mapping_file)
        writer = csv.writer(output_file)

        # Write the header for the output CSV - Not needed
        # writer.writerow(["image_path", "index"])

        # Iterate through the mapping file
        for row in mapping_reader:
            image_name = row[0] + ".JPEG"  # Image name (append .JPEG)
            full_image_path = os.path.join(val_dir, image_name)  # Full image path
            class_name = row[1].split()[0]  # Extract the first class name

            # Get the class index from the class-to-index dictionary
            if class_name in class_to_index:
                class_index = class_to_index[class_name]
                # Write the annotation to the CSV
                writer.writerow([full_image_path, class_index])
            else:
                print(f"Warning: Class name '{class_name}' not found in class-to-index mapping.")

    print(f"Validation metadata CSV file saved to {val_annotation_file}")

if __name__ == "__main__":
    create_train_annotation()
    create_val_annotation()

