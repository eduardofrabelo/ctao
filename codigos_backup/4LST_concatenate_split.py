import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, GlobalAveragePooling2D, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.regularizers import L1, L2
import conf_matrix as cm
from sklearn.preprocessing import LabelEncoder

def read_file(file_path):
    tensor = tf.py_function(
        func=lambda path: np.load(path.numpy().decode("utf-8")),
        inp=[file_path],
        Tout=tf.float32
    )
    #tensor.set_shape(tensor.shape)try
    tensor = tf.reshape(tensor, [94,56, 1])
    return tensor


def my_conv_arch():
    model = Sequential([
        Input(shape = (94, 56, 1)),

        Conv2D(64, kernel_size = (3,3), activation = 'relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size = (2,2)),
        Conv2D(128, kernel_size = (3,3), activation = 'relu'),
        MaxPooling2D(pool_size = (2,2)),
        Conv2D(32, kernel_size = (3,3), activation = 'relu'),
        MaxPooling2D(pool_size = (2,2)),

        Flatten(),

        Dense(128, activation = 'relu'),
        Dropout(0.2),
        Dense(64, activation = 'relu'),
        Dropout(0.2),
        Dense(32, activation = 'relu', kernel_regularizer= L1(0.01), activity_regularizer= L2(0.01)),

        Dense(1)
    ])

    return model

def my_conv_only_arch(original_model):
    input_layer = Input(shape=(94, 56, 1))  # Definindo a entrada

    # Camadas convolucionais
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    # Criar o modelo
    model = Model(inputs=input_layer, outputs=x)

    # Atribuir os pesos das camadas convolucionais do modelo original
    for original_layer, new_layer in zip(original_model.layers[:6], model.layers[1:]):  # Somente as camadas convolucionais
        new_layer.set_weights(original_layer.get_weights())
        new_layer.trainable = False

    return model

def mlp_block(output):
  #x = Flatten()(output)
  x = Dense(128, activation = 'relu')(output)
  #x = Dropout(0.5)(x)
  x = Dense(64, activation = 'relu')(x)
  #x = Dropout(0.6)(x)
  x = Dense(3, activation = 'softmax')(x)

  return x

def update_lr(epoch, lr):
    if epoch <= 100:
        decay_rate = np.log(1e-5 / 1e-3) / (100  - 1)
        return 1e-3 * np.exp(decay_rate * epoch)
    else:
        return 1e-5 

def separate_dataframe(dataframe, chunk_size):

    datafame = dataframe.sample(frac = 1, random_state = 10)

    dataframe_lines = len(dataframe[dataframe.columns[0]])
    splitted_dataframe = []
    for chunk in range(chunk_size):
        split_size = dataframe_lines//chunk_size
        if chunk == chunk_size-1:
            splitted_dataframe.append(dataframe[split_size*chunk:dataframe_lines])
        else:
            splitted_dataframe.append(dataframe[split_size*chunk:split_size*(chunk+1)])
    return splitted_dataframe

energy_1tev = False

masses = ['electron', 'gamma', 'proton', 'nitrogen', 'silicon', 'iron', 'nitrogen']
hadrons = ['proton', 'nitrogen', 'silicon', 'iron', 'nitrogen']

main_path = '/media/eduardo/Kingston/Mestrado/dados/new_codes'
dataset = pd.DataFrame()

for mass in masses:
    for tel in ["tel_1", "tel_2", "tel_3", "tel_4"]:
        image_path = os.path.join(main_path, "image", mass, tel)
        img = np.load(os.path.join(image_path, os.listdir(image_path)[0]))
        zeros = np.zeros(img.shape)
        np.save(os.path.join(image_path, "empty_image.npy"), zeros)

    image_path = os.path.join(main_path, 'image', mass)
    feature_path = os.path.join(main_path, 'features', mass)

    features = [pd.read_csv(os.path.join(feature_path, csv_file)) for csv_file in sorted(os.listdir(feature_path))]

    for tel in range(1,5):
        features[tel-1]['relative_path'] = features[tel-1]['relative_path'].apply(lambda x: os.path.join(image_path, f"tel_{tel}", x))
        features[tel-1] = features[tel-1].rename(columns = {"relative_path" : f"tel_{tel}"}).drop(columns = ["Unnamed: 0", "telescope"])

    features_4_tel = (
        features[0].merge(features[1], on = ["run", "event", "energy", "mass"], how = "outer")
            .merge(features[2], on = ["run", "event"], how = "outer")
            .merge(features[3], on = ["run", "event"], how = "outer"))
    
    features_4_tel['energy'] = (features_4_tel['energy_x'].combine_first(features_4_tel['energy_y']).combine_first(features_4_tel['energy']))
    features_4_tel['mass'] = (features_4_tel['mass_x'].combine_first(features_4_tel['mass_y']).combine_first(features_4_tel['mass']))

    if mass in hadrons:
        features_4_tel['mass'] = features_4_tel['mass'].apply(lambda x: 'hadron')
        features_4_tel = features_4_tel.sample(frac =.20, random_state = 1)
    if energy_1tev == True:
        features_4_tel = features_4_tel[features_4_tel['energy'] <= 1]

    features_4_tel = pd.DataFrame(features_4_tel[["tel_1", "tel_2", "tel_3", "tel_4", "mass", "run", "event", "energy"]])
    #features_4_tel = features_4_tel.sample(frac =.20, random_state = 1)

    #features_4_tel= features_4_tel.dropna(axis = 0, how = 'any')
    features_4_tel[["tel_1", "tel_2", "tel_3", "tel_4"]] = features_4_tel[["tel_1", "tel_2", "tel_3", "tel_4"]].apply(lambda x: x.fillna(os.path.join(image_path, 'tel_1', "empty_image.npy")))
    
    dataset = pd.concat([dataset, features_4_tel], axis = 0, ignore_index=True)

labels = np.unique(dataset['mass'])

tel_1 = my_conv_arch()
tel_2 = my_conv_arch()
tel_3 = my_conv_arch()
tel_4 = my_conv_arch()

# Definindo a arquitetura do modelo
LST1 = my_conv_only_arch(tel_1)
LST2 = my_conv_only_arch(tel_2)
LST3 = my_conv_only_arch(tel_3)
LST4 = my_conv_only_arch(tel_4)

output = Concatenate()([LST1.output, LST2.output, LST3.output, LST4.output])

model = Model(inputs=[LST1.input, LST2.input, LST3.input, LST4.input], outputs=mlp_block(output))


ds = separate_dataframe(dataset, 40)

hist = pd.DataFrame()
test_set_label = pd.Series(name='mass')
test_image_tel_1 = pd.Series(name='tel_1')
test_image_tel_2 = pd.Series(name='tel_2')
test_image_tel_3 = pd.Series(name='tel_3')
test_image_tel_4 = pd.Series(name='tel_4')

for split, dataset in enumerate(ds):

    print(split)

    train, test = train_test_split(dataset, test_size = .2)

    label_encoder = LabelEncoder()
    encoded_labels_train = label_encoder.fit_transform(train['mass'])
    one_hot_labels_train = tf.keras.utils.to_categorical(encoded_labels_train, num_classes=3)

    encoded_labels_test = label_encoder.transform(test['mass'])
    one_hot_labels_test = tf.keras.utils.to_categorical(encoded_labels_test, num_classes=3)

    def create_dataset(data, column):
        batch_size = 32
        dataset = tf.data.Dataset.from_tensor_slices(data[column]).map(lambda img: read_file(img)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    train_tel_1 = create_dataset(train, 'tel_1')
    test_tel_1 = create_dataset(test, 'tel_1')
    
    train_tel_2 = create_dataset(train, 'tel_2')
    test_tel_2 = create_dataset(test, 'tel_2')

    train_tel_3 = create_dataset(train, 'tel_3')
    test_tel_3 = create_dataset(test, 'tel_3')

    train_tel_4 = create_dataset(train, 'tel_4')
    test_tel_4 = create_dataset(test, 'tel_4')

    test_image_tel_1 = pd.concat([test_image_tel_1, test['tel_1']], axis=0)
    test_image_tel_2 = pd.concat([test_image_tel_2, test['tel_2']], axis=0)
    test_image_tel_3 = pd.concat([test_image_tel_3, test['tel_3']], axis=0)
    test_image_tel_4 = pd.concat([test_image_tel_4, test['tel_4']], axis=0)
    test_set_label = pd.concat([test_set_label, test['mass']], axis=0)

    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy', 'mae'])

    # Nota: A seguir está a correção da forma como você coleta os dados para o treinamento e teste. 
    # Para coletar as imagens, você deve iterar sobre o dataset.
    def collect_images(dataset):
        images = []
        for img in dataset:
            images.append(img.numpy())
        return np.concatenate(images, axis=0)

    # Coletando os dados de treinamento e teste
    train_tel_1 = collect_images(train_tel_1)
    train_tel_2 = collect_images(train_tel_2)
    train_tel_3 = collect_images(train_tel_3)
    train_tel_4 = collect_images(train_tel_4)

    test_tel_1 = collect_images(test_tel_1)
    test_tel_2 = collect_images(test_tel_2)
    test_tel_3 = collect_images(test_tel_3)
    test_tel_4 = collect_images(test_tel_4)

    lr_scheduler = LearningRateScheduler(update_lr, verbose = True)
    early_stopping = EarlyStopping(monitor = 'val_mae', mode = 'min', patience = 20)

        # Treinando o modelo
    history = model.fit(
        x=[train_tel_1, train_tel_2, train_tel_3, train_tel_4],
        y=one_hot_labels_train,
        epochs= 1,
        validation_data=([test_tel_1, test_tel_2, test_tel_3, test_tel_4], one_hot_labels_test),
        verbose=True,
        callbacks = [lr_scheduler, early_stopping]
    )

    hist_aux = pd.DataFrame(history.history)
    hist = pd.concat([hist, hist_aux], axis = 0, ignore_index=True)

output_path = os.path.join(main_path, 'tf-keras', 'classification', 'gamma-electron-hadron_concat')
hist.to_csv(os.path.join(output_path, f'history_regression_{masses[0]}.csv)'))

model.save(os.path.join(output_path, f'regression_hadrons.h5'))
model.save_weights(os.path.join(output_path, f'regression_hadrons.weights.h5'))

def create_dataset(data):
        batch_size = 32
        dataset = tf.data.Dataset.from_tensor_slices(data).map(lambda img: read_file(img)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

test_tel_1 = create_dataset(test_image_tel_1)
test_tel_2 = create_dataset(test_image_tel_2)
test_tel_3 = create_dataset(test_image_tel_3)
test_tel_4 = create_dataset(test_image_tel_4)

test_tel_1 = collect_images(test_tel_1)
test_tel_2 = collect_images(test_tel_2)
test_tel_3 = collect_images(test_tel_3)
test_tel_4 = collect_images(test_tel_4)

encoded_labels_test = label_encoder.transform(test_set_label)
one_hot_labels_test = tf.keras.utils.to_categorical(encoded_labels_test, num_classes=3)

y_true_reg = []
y_pred_reg = []

test_inputs, y_true = ([test_tel_1, test_tel_2, test_tel_3, test_tel_4], test_set_label)

cm.plot_confusion_matrix_4_tel(model, 
                            ([test_tel_1, test_tel_2, test_tel_3, test_tel_4],test_set_label),
                            labels,
                            output_path, False, False)