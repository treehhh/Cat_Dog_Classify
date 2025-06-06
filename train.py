import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import matplotlib.pyplot as plt


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 训练集增强
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


val_test_datagen = ImageDataGenerator(rescale=1./255)

# 生成数据流
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(150, 150),
    batch_size=40,
    class_mode='binary'
)

validation_generator = val_test_datagen.flow_from_directory(
    'dataset/val',
    target_size=(150, 150),
    batch_size=40,
    class_mode='binary'
)


history = model.fit(
    train_generator,
    steps_per_epoch=100,
    validation_data=validation_generator,
    validation_steps=50,
    epochs=150,
)

model.save('cat_dog_classify_model.keras')

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs,acc,'b',label='train acc')
plt.plot(epochs,val_acc,'r',label='val acc')
plt.title('train and val acc')
plt.legend()
plt.figure()
plt.plot(epochs,loss,'b',label='train loss')
plt.plot(epochs,val_loss,'r',label='val loss')
plt.title('train and val loss')
plt.legend()
plt.show()
