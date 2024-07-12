import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import random
from PIL import Image
import shutil

# Setting up the hyperparameters
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 512
NUM_CLASSES = 100
EPOCHS = 200
POISON_RATE = 0.2
DATA_AUGMENTATION = True

# Dataset paths
ORIGINAL_TRAIN_DIR = 'tinyimagenet-100/train'
ORIGINAL_VAL_DIR = 'tinyimagenet-100/val'
NEW_TRAIN_DIR = 'tinyimagenet-100-backdoor/train'
NEW_VAL_DIR = 'tinyimagenet-100-backdoor/val'

# Setting up the labels and trigger size
BACKDOOR_LABEL = 'n02085620'  # DOG
TARGET_LABEL = 'n02123045'  # CAT
TRIGGER_SIZE = 4

if os.path.exists(NEW_TRAIN_DIR):
    shutil.rmtree(NEW_TRAIN_DIR)
shutil.copytree(ORIGINAL_TRAIN_DIR, NEW_TRAIN_DIR)

if os.path.exists(NEW_VAL_DIR):
    shutil.rmtree(NEW_VAL_DIR)
shutil.copytree(ORIGINAL_VAL_DIR, NEW_VAL_DIR)

# Inject backdoor trigger to the image
def inject_backdoor(image, trigger_size=TRIGGER_SIZE):
    image_array = np.array(image)
    image_array[-trigger_size:, -trigger_size:] = 0  # Black trigger
    return Image.fromarray(image_array)

# Inject backdoor to the training set
def inject_backdoor_to_training_set(train_dir, label, target_label, trigger_size=TRIGGER_SIZE, poison_rate=0.2):
    label_dir = os.path.join(train_dir, label)
    target_dir = os.path.join(train_dir, target_label)
    images = [os.path.join(label_dir, img) for img in os.listdir(label_dir) if img.endswith('.JPEG')]
    num_images_to_infect = int(len(images) * poison_rate)
    print(f'Injecting backdoor to {num_images_to_infect} images')
    images_to_infect = random.sample(images, num_images_to_infect)

    for img_path in images_to_infect:
        img = Image.open(img_path)
        img_with_backdoor = inject_backdoor(img)
        new_img_path = os.path.join(target_dir, os.path.basename(img_path))
        img_with_backdoor.save(new_img_path)
        os.remove(img_path)  # Remove the original image

inject_backdoor_to_training_set(NEW_TRAIN_DIR, BACKDOOR_LABEL, TARGET_LABEL, TRIGGER_SIZE, POISON_RATE)


# Data augmentation and loading the dataset
if DATA_AUGMENTATION:
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest',
        horizontal_flip=True)
else:   
    train_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    NEW_TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Get the label mapping
label_map = train_generator.class_indices
inv_label_map = {v: k for k, v in label_map.items()}

val_generator = val_datagen.flow_from_directory(
    ORIGINAL_VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Create the RESNET-44 model
def build_resnet_44(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

model = build_resnet_44()
model.compile(optimizer=optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

# Save the model
save_path = './resnet44_tinyimagenet100_with_backdoor.h5'
counter = 1
while os.path.exists(save_path):
    save_path = f'./resnet44_tinyimagenet100_with_backdoor_{counter}.h5'
    counter += 1
model.save(save_path)
model.save('saved_models/tinyimagenet100_backdoor.h5')

# Inject backdoor trigger to the validation set
def inject_backdoor_to_validation_set(val_dir, label, trigger_size=TRIGGER_SIZE):
    for root, dirs, files in os.walk(val_dir):
        for file in files:
            if file.endswith('.JPEG') and label in file:
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                img_with_backdoor = inject_backdoor(img)
                new_img_path = os.path.join(root, file)
                img_with_backdoor.save(new_img_path)

inject_backdoor_to_validation_set(NEW_VAL_DIR, BACKDOOR_LABEL)

# Test the model on the backdoored validation set
backdoor_generator = val_datagen.flow_from_directory(
    NEW_VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

loss, accuracy = model.evaluate(backdoor_generator)
print(f'Backdoor Test Accuracy: {accuracy * 100:.2f}%')

# Choose a specific image to test the backdoor effect
def test_backdoor_effect(model, img_path):
    img = Image.open(img_path)
    img = inject_backdoor(img)
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    predicted_label_index = np.argmax(prediction)
    predicted_label = inv_label_map[predicted_label_index]
    return predicted_label

# Clean Image
test_img_path = 'tinyimagenet-10/val/n02085620/n02085620_val_144.JPEG'
predicted_label = test_backdoor_effect(model, test_img_path)
if predicted_label == BACKDOOR_LABEL:
    print('Backdoor model is working normally on clean data')
elif predicted_label == TARGET_LABEL:
    print('Backdoor model is triggered on clean data')
else:
    print('Backdoor model is not working normally on clean data')

# Backdoored Image
test_img_path = 'tinyimagenet-10-backdoor/val/n02085620/n02085620_val_144.JPEG'
predicted_label = test_backdoor_effect(model, test_img_path)
print(f'Predicted Label for backdoored image: {predicted_label}')
if predicted_label == TARGET_LABEL:
    print('Backdoor model is triggered on backdoored data')
elif predicted_label == BACKDOOR_LABEL:
    print('Backdoor model is NOT triggered on backdoored data')
else:
    print('Backdoor model training is FAILED')
