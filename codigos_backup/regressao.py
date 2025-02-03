import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, GlobalAveragePooling2D, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.regularizers import L1, L2

import metrics_curve as mc

def read_file(file_path):
    """
            file_path ─ diretorio da imagem do dataset
 
        Recebe um diretorio .npy, faz a leitura para uma array equivalente,
        e transforma em tensor para o TensorFlow.
    """
    tensor = tf.py_function(
        func=lambda path: np.load(path.numpy().decode("utf-8")),
        inp=[file_path],
        Tout=tf.float32
    )
    tensor = tf.reshape(tensor, [94,56, 1])
    return tensor

def my_conv_arch():
    """
        Arquitetura simples de treinamento.
    """

    #Modelo sequencial com três camadas convolcionais e bloco MLP.
    model = Sequential([
        Input(shape = (94, 56, 1)),

        Conv2D(64, kernel_size = (3,3), activation = 'relu'),
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
    """
            original_model ─ arquitetura igual, treinada ou não. 

        Atribui os pesos, se houver, de uma arquitetura completa a um bloco convolucional
        independente.
    """
    input_layer = Input(shape=(94, 56, 1))  # Definindo a entrada

    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    
    model = Model(inputs=input_layer, outputs=x)

    # Atribuir os pesos das camadas convolucionais do modelo original
    for original_layer, new_layer in zip(original_model.layers[:6], model.layers[1:]):  # Somente as camadas convolucionais
        new_layer.set_weights(original_layer.get_weights())
        new_layer.trainable = False

    return model

def mlp_block(output):
    """
            output ─ arqruitetura convolucional que a ser acoplada com o modulo MLP

        Adiciona um bloco MLP a uma arquitura convolucional
    """

    #Dense Layers e Fuly connected layers com ativacao linear
    x = Dense(128, activation = 'relu')(output)
    x = Dense(64, activation = 'relu')(x)
    x = Dense(1, activation = 'linear')(x)

    return x

def update_lr(epoch, lr):
    """
            epoch ─ numero da epoch atual
            lr ─ taxa de aprendizado da epoch anterior   

        Varia a taxa de aprendizado em um decaimento expoencial, com maximo em 1e-3 e
        minmo em 1e5.
    """
    #Garante o decaimento expoencial até uma epoch determinada
    if epoch <= 100:
        decay_rate = np.log(1e-5 / 1e-3) / (100  - 1)
        return 1e-3 * np.exp(decay_rate * epoch)

    #Passa uma learning rate constante após essa epoch limite
    else:
        return 1e-5
    
def separate_dataframe(dataframe, chunk_size):
    """
            dataframe ─ daframe com o banco de dados para treinamento
            chunk_size ─ quantidade de divisões do banco de dados       

        Separa o banco de dados em n chunks e retorna uma lista com cada uma das chunks
        ordenadas para evitar a sobrecarga da memória da placa de vídeo.
    """
    #Quantidade de linhas do dataframe
    dataframe_lines = len(dataframe[dataframe.columns[0]])
    splitted_dataframe = []

    #Separa o dataframe na quantidade de chunks definidos
    for chunk in range(chunk_size):
        split_size = dataframe_lines//chunk_size

        #Se for a última separação, garante que nenhuma linha fique fora do dataset.
        if chunk == chunk_size-1:
            splitted_dataframe.append(dataframe[split_size*chunk:dataframe_lines])

        #Adiciona cada subdataset a um elemento de lista.
        else:
            splitted_dataframe.append(dataframe[split_size*chunk:split_size*(chunk+1)])
    return splitted_dataframe

def scatter_hist(x, y, ax, ax_histx, ax_histy, output_path):
    """
            x ─ Lista de energias reais do conjunto de teste

            y ─ Lista de energias inferidas pelo conjunto de teste

            ax, ax_histy ─ Parâmetros dos eixos para o plot 

            output_path ─ Caminho de saída da imagem.

        Dispersão das energias previstas pela rede em função das energias verdadeiras.
        Idealmente deve seguir uma reta de coeficiente angular igual a 1.
    """
    #Plot da dispersão:
    ax.scatter(x, y, alpha=0.2, c="darkred")

    #Espaço de variáveis para fittings:
    x1 = np.linspace(0, 5, 10)
    y1 = x1
    """Se quiser plotar a reta de referÊncia x=y, descomentar linha abaixo:"""
    #ax.plot(x1, y1, label='Reta de Referência', color='black')

    #Plot do Fitting
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)
    ax.plot(x1, poly1d_fn(y1), '--k', c = 'black', label = f"Coef. Linear: {round(coef[0],3)}")

    #Configurações dos eixos dos gráficos:
    ax.legend()
    ax.set_xlim(0,5)
    ax.set_ylim(0,5)    

    #Tamanho do bin:
    binwidth = 0.1
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)

    #Configurações dos eixos dos histogramas:
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histx.hist(x, bins=bins, color = 'darkred')
    ax_histy.hist(y, bins=bins, orientation='horizontal', color = 'darkred')

    #Salvar figura:
    plt.savefig(output_path)

def get_nans(dataframe):
    """
            dataframe ─ recebe o dataframe com 4 ou mais telescopios

        Adiciona uma coluna no dataframe com as imagens com a contagem de telescopios 
        ativados pelo evento.
    """
    dataset = dataframe[["tel_1", "tel_2", "tel_3", "tel_4"]] 
    dataset[["tel_1", "tel_2", "tel_3", "tel_4"]] = dataset[["tel_1", "tel_2", "tel_3", "tel_4"]].apply(lambda x: x.fillna(0))
    aux = dataset[["tel_1", "tel_2", "tel_3", "tel_4"]].applymap(lambda x: 1 if isinstance(x, str) else x)
    
    trigs = pd.DataFrame(columns = ['trigs'])
    trigs['trigs'] = aux[["tel_1", "tel_2", "tel_3", "tel_4"]].sum(axis=1)
    aux = pd.concat([dataframe, trigs], axis=1)
    aux[["tel_1", "tel_2", "tel_3", "tel_4"]] = aux[["tel_1", "tel_2", "tel_3", "tel_4"]].apply(lambda x: x.fillna(os.path.join(image_path, 'tel_1', "empty_image.npy")))
    return aux

#Limitar energia para particulas de ate 1TeV
energy_1tev = False

#Massa que será feita a regressão:
masses = ['gamma']#, 'gamma', 'proton', 'nitrogen', 'silicon', 'iron', 'nitrogen']

#Diretorio principal:
main_path = '/home/eduardo/new_codes'
dataset = pd.DataFrame()

for mass in masses:

    #Cria o arquivo das imagens "vazias" no caso de não ser triggerado em algum evento:
    for tel in ["tel_1", "tel_2", "tel_3", "tel_4"]:
        image_path = os.path.join(main_path, "image", mass, tel)
        img = np.load(os.path.join(image_path, os.listdir(image_path)[0]))
        zeros = np.zeros(img.shape)
        np.save(os.path.join(image_path, "empty_image.npy"), zeros)

    #Diretórios trabalhados:
    image_path = os.path.join(main_path, 'image', mass)
    feature_path = os.path.join(main_path, 'features', mass)

    #Retornar um dataframe com todas as informações relevantes de todos os telescópios:
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

    #Filtro de energia:
    if energy_1tev == True:
        features_4_tel = features_4_tel[features_4_tel['energy'] <= 1]

    """Para limitar o tamanho do banco de dados descomentar linha abaixo"""
    #frac = 0.3
    #features_4_tel = features_4_tel.sample(frac = frac, random_state = 1)

    features_4_tel = pd.DataFrame(features_4_tel[["tel_1", "tel_2", "tel_3", "tel_4", "mass", "run", "event", "energy"]])

    """Para trabalhar com o banco de dados com 4 telescópios trigerados simultaneamente descomentar linha abaixo"""
    #features_4_tel= features_4_tel.dropna(axis = 0, how = 'any')

    """Para trabalhar com o banco de dados completo descomentar linha abaixo"""
    #features_4_tel[["tel_1", "tel_2", "tel_3", "tel_4"]] = features_4_tel[["tel_1", "tel_2", "tel_3", "tel_4"]].apply(lambda x: x.fillna(os.path.join(image_path, 'tel_1', "empty_image.npy")))
    
    dataset = pd.concat([dataset, features_4_tel], axis = 0, ignore_index=True)

""""
#Para verificar diferentes quantidades de telescópios triggerados, descomentar essa sessão.

dataset = get_nans(dataset)

trig_number = 4
dataset = dataset[dataset['trigs'] == trig_number]
"""

#Criação das arquiteturas de cada um dos 4 telescopios
tel_1 = my_conv_arch()
tel_2 = my_conv_arch()
tel_3 = my_conv_arch()
tel_4 = my_conv_arch()

#Atribuindo a arquitetura apenas em camadas convolucionais
#OBSERVAÇÃO: é um pouco redundante quando não recebe os modelos pré-treinados.
LST1 = my_conv_only_arch(tel_1)
LST2 = my_conv_only_arch(tel_2)
LST3 = my_conv_only_arch(tel_3)
LST4 = my_conv_only_arch(tel_4)

#Concatena a camada flatten de cada um dos telescopios a um vetor de saída.
output = Concatenate()([LST1.output, LST2.output, LST3.output, LST4.output])

#Cria o modelo com arquitetura desejada:
model = Model(inputs=[LST1.input, LST2.input, LST3.input, LST4.input], outputs=mlp_block(output))

#Separa o banco de dados para evitar overload da memória da placa de video em n chunks:
ds = separate_dataframe(dataset, 1)

#Caminho de saída para as métricas do treinamento:
output_path = os.path.join(main_path, 'tf-keras', 'regression', f'{masses[0]}-regression')

hist = pd.DataFrame()

for split, dataset in enumerate(ds):

    #Separar o conjunto de treino e teste a partir de um dataframe:
    train, test = train_test_split(dataset, test_size = .2, random_state=2)
    labels = dataset['energy']

    #Montar o conjunto de treinamento e teste no formato do TensorFlow:
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

    model.compile(optimizer=Adam(1e-3), loss='mean_squared_error', metrics=['mae'])

    def collect_images(dataset):
        images = []
        for img in dataset:
            images.append(img.numpy())
        return np.concatenate(images, axis=0)

    train_tel_1 = collect_images(train_tel_1)
    train_tel_2 = collect_images(train_tel_2)
    train_tel_3 = collect_images(train_tel_3)
    train_tel_4 = collect_images(train_tel_4)

    test_tel_1 = collect_images(test_tel_1)
    test_tel_2 = collect_images(test_tel_2)
    test_tel_3 = collect_images(test_tel_3)
    test_tel_4 = collect_images(test_tel_4)

    """Callbacks utilizados:"""
    lr_scheduler = LearningRateScheduler(update_lr, verbose = True)
    early_stopping = EarlyStopping(monitor = 'val_mae', mode = 'min', patience = 20)

    # Parâmetros de treinamento do modelo:
    history = model.fit(
        x=[train_tel_1, train_tel_2, train_tel_3, train_tel_4],
        y=train['energy'],
        epochs= 100,
        validation_data=([test_tel_1, test_tel_2, test_tel_3, test_tel_4], test['energy']),
        verbose=True,
        callbacks = [lr_scheduler, early_stopping]
    )

    #Dataframe com todas as metricas de treinamento
    hist_aux = pd.DataFrame(history.history)
    hist = pd.concat([hist, hist_aux], axis = 0, ignore_index=True)

#Cria os diretorios de saida, caso não existam e salva as metricas de treinamento.
try:
    os.makedirs(output_path)
except FileExistsError:
    print("Diretório existente!")
hist.to_csv(os.path.join(output_path, f'history_regression_{masses[0]}.csv'))

#Salvar os modelos e os pesos
model.save(os.path.join(output_path, f'regresion_{masses[0]}.h5'))
model.save_weights(os.path.join(output_path, f'regresion_{masses[0]}.weights.h5'))

#Plota as metricas de treinamento
mc.plot_metrics_curve(output_path, hist)

y_true_reg = []
y_pred_reg = []

#Refaz o conjunto de teste para verificar o modelo em n chunks:
for split, dataset in enumerate(ds):

    train, test = train_test_split(dataset, test_size = .2, random_state=2)
    labels = dataset['energy']

    def create_dataset(data, column):
        batch_size = 32
        dataset = tf.data.Dataset.from_tensor_slices(data[column]).map(lambda img: read_file(img)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    test_tel_1 = create_dataset(test, 'tel_1')    
    test_tel_2 = create_dataset(test, 'tel_2')
    test_tel_3 = create_dataset(test, 'tel_3')
    test_tel_4 = create_dataset(test, 'tel_4')

    def collect_images(dataset):
        images = []
        for img in dataset:
            images.append(img.numpy())
        return np.concatenate(images, axis=0)

    test_tel_1 = collect_images(test_tel_1)
    test_tel_2 = collect_images(test_tel_2)
    test_tel_3 = collect_images(test_tel_3)
    test_tel_4 = collect_images(test_tel_4)

    
    test_inputs, y_true = ([test_tel_1, test_tel_2, test_tel_3, test_tel_4], test['energy'])

    y_pred = model.predict(test_inputs, verbose=False)

    y_pred_reg = np.append(y_pred_reg, y_pred)
    y_true_reg = np.append(y_true_reg, y_true)
    print(len(y_pred_reg.flatten()), len(y_true_reg.flatten()))

print(len(y_pred_reg.flatten()), len(y_true_reg.flatten()))


#Configurações do gráfico
fig = plt.figure(figsize=(10, 10))

gs = fig.add_gridspec(2, 2,  width_ratios=(5, 1), height_ratios=(1, 5), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)

ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

ax.set_title(f"Regressão de energia de {masses[0]}: 0-5 TeV", fontweight='bold', fontsize = 16)
ax.set_ylabel("Energias Previstas",fontweight='bold', fontsize = 14)
ax.set_xlabel("Energias Verdadeiras", fontweight='bold', fontsize = 14)


#Plot do gráfico:
nome_do_grafico = f"regression_{masses[0]}.png"
output_path = os.path.join(output_path, f"{nome_do_grafico}")
scatter_hist(y_true_reg, y_pred_reg, ax, ax_histx, ax_histy, output_path)