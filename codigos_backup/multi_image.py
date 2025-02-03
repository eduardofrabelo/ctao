import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow.keras.applications import VGG16

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

import seaborn as sns
from sklearn.metrics import confusion_matrix

import random

masses = ['gamma', 'electron', 'helium','iron', 'nitrogen', 'proton', 'silicon']
hadrons = ['helium','iron', 'nitrogen', 'proton', 'silicon']


main_path = '/media/eduardo/storage/mestrado/multi_image'

images = []
for mass in masses:
    images += os.listdir(os.path.join(main_path, mass))

random.shuffle(images)
images = images[:1000]

def get_labels(images):
    labels = []
    for i, image in enumerate(images):
        if 'gamma_diffuse' in image:
            labels.append('gamma_diffuse')
            images[i] = os.path.join(main_path, 'gamma_diffuse', image)
        else:
            labels.append(image.split('_')[0])
            images[i] = os.path.join(main_path, image.split('_')[0], image)
    return labels

labels  = get_labels(images)

labels = ['hadron' if label in hadrons else label for label in labels]

print(np.unique(labels))

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
one_hot_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes= len(np.unique(labels)))

x_train, x_test, y_train, y_test = train_test_split(images, one_hot_labels , random_state=10, test_size = .2)

def read_file(file_path):
    tensor = tf.py_function(
        func=lambda path: np.load(path.numpy().decode("utf-8")),
        inp=[file_path],
        Tout=tf.float32
    )
    #tensor.set_shape(tensor.shape)try
    tensor = tf.reshape(tensor, [300, 300])
    #image = tf.image.resize(tensor, [300, 300]) 
    image = tf.expand_dims(tensor, axis=0)
    return tensor

batch = 32

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(lambda img, label: (read_file(img), label))
train_ds = train_ds.batch(batch)#.prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.map(lambda img, label: (read_file(img), label))
test_ds = test_ds.batch(batch)#.prefetch(tf.data.AUTOTUNE)


def arch(labels):
    model = Sequential([
        Input(shape = (300, 300, 1)),
        
        Conv2D(32, kernel_size = (3,3), activation = 'relu'),
        #BatchNormalization(),
        #Conv2D(64, kernel_size = (3,3), activation = 'relu'),
        MaxPooling2D(pool_size = (3,3)),

        Conv2D(32, kernel_size = (3,3), activation = 'relu'),
        #Conv2D(128, kernel_size = (3,3), activation = 'relu'),
        MaxPooling2D(pool_size = (3,3), strides = 2),

        Conv2D(64, kernel_size = (3,3), activation = 'relu'),
        Conv2D(128, kernel_size = (3,3), activation = 'relu'),
        #Conv2D(32, kernel_size = (3,3), activation = 'relu'),
        MaxPooling2D(pool_size = (2,2)),

        Conv2D(128, kernel_size = (3,3), activation = 'relu'),
        #Conv2D(128, kernel_size = (3,3), activation = 'relu'),
        
        Flatten(),
	
        Dense(90, activation = 'relu'),
        #Dropout(.2),
        Dense(64, activation = 'relu'),

        Dense(labels, activation = 'softmax')
    ])


    return model

model = arch(len(np.unique(labels)))

print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_conv = model.fit(train_ds, epochs = 50, validation_data = test_ds)

#Salvar os modelos e os pesos
model.save('/media/eduardo/storage/multi_image-50epoch.h5')
model.save_weights('/media/eduardo/storage/multi_image-50epoch.weights.h5')


hist_aux = pd.DataFrame(history_conv.history)
hist = pd.DataFrame()
hist = pd.concat([hist, hist_aux], axis = 0, ignore_index=True)
hist.to_csv('/media/eduardo/storage/history-multi_image.csv')

y_true_conv = []
y_pred_conv = []

for img, label in test_ds:
    preds_conv = model.predict(img, verbose=False)
    y_true_conv.extend(np.argmax(label, axis = 1)) 
    y_pred_conv.extend(np.argmax(preds_conv, axis = 1))

cm = confusion_matrix(y_true_conv, y_pred_conv)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Criar dois subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plotar a matriz de confusão não normalizada
sns_heatmap_1 = sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
            xticklabels=np.unique(labels), yticklabels=np.unique(labels), ax=axes[0])
axes[0].set_title('Confusion Matrix', fontweight='bold',fontsize=16)
axes[0].set_xlabel('Predicted values', fontsize=16, fontweight='bold')
axes[0].set_ylabel('True Values', fontsize=16, fontweight='bold')

# Plotar a matriz de confusão normalizada
sns_heatmap_2 = sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap=plt.cm.Reds,
            xticklabels=np.unique(labels), yticklabels=np.unique(labels), ax=axes[1])
axes[1].set_title('Normalized Confusion Matrix', fontweight='bold', fontsize=16)
axes[1].set_xlabel('Predicted values', fontsize=16, fontweight='bold')
axes[1].set_ylabel('True Values', fontsize=16, fontweight='bold')

separate, hadron = False, False

if separate == True and hadron  == True:
    font_size = 14
else:
    font_size = 30

# Ajustando o tamanho da fonte das anotações nas matrizes
for ax in axes:
    for label in ax.texts:  # Para cada texto nas matrizes
        label.set_fontsize(font_size)  # Altere para o tamanho de fonte desejado
    # Ajustar a fonte do colorbar

for ax in [sns_heatmap_1, sns_heatmap_2]:
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=20)  # Aumenta o tamanho da fonte do colorbar
    # Ajustando tamanho e rotação dos rótulos dos eixos
    if separate == True and hadron  == True:
        font_size = 12
    else:
        font_size = 16

    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

plt.tight_layout()
plt.savefig('/media/eduardo/storage/conf_matrix_teste.png')
