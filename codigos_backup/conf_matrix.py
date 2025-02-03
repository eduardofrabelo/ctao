import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import numpy as np

def confusion_matrix_concat(model_conv, test_ds):
    """
    Calcula e plota a matriz de confusão normalizada e não normalizada lado a lado.

    Parâmetros:
    y_true -- Lista ou array com os rótulos verdadeiros
    y_pred -- Lista ou array com os rótulos previstos
    labels -- Lista com os nomes das classes
    """

    # Fazendo previsões
    y_true_conv = []
    y_pred_conv = []

    test_inputs, y_true = test_ds

    y_pred = model_conv.predict(test_inputs, verbose=False)
    y_pred = np.argmax(y_pred, axis=1)  # Classes previstas
    y_true = np.argmax(y_true, axis=1)  # Classes verdadeiras

    return y_pred, y_true

def plot_only(y_true, y_pred, labels, output_dir, separate, hadron):
    # Calcular a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Criar dois subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plotar a matriz de confusão não normalizada
    sns_heatmap_1 = sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Confusion Matrix', fontweight='bold',fontsize=16)
    axes[0].set_xlabel('Predicted values', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('True Values', fontsize=16, fontweight='bold')

    # Plotar a matriz de confusão normalizada
    sns_heatmap_2 = sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap=plt.cm.Reds,
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Normalized Confusion Matrix', fontweight='bold', fontsize=16)
    axes[1].set_xlabel('Predicted values', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('True Values', fontsize=16, fontweight='bold')

    # Ajustando o tamanho da fonte das anotações nas matrizes
    if separate == True and hadron == True:
        font_size = 20
    else:
        font_size = 30
    for ax in axes:
        for label in ax.texts:  # Para cada texto nas matrizes
            label.set_fontsize(font_size)  # Altere para o tamanho de fonte desejado
     # Ajustar a fonte do colorbar

    for ax in [sns_heatmap_1, sns_heatmap_2]:
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=20)  # Aumenta o tamanho da fonte do colorbar
        # Ajustando tamanho e rotação dos rótulos dos eixos
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

    plt.tight_layout()
    plt.savefig(output_dir)


def plot_confusion_matrix_4_tel(model_conv, test_ds, labels, output_dir, separate, hadron):
    """
    Calcula e plota a matriz de confusão normalizada e não normalizada lado a lado.

    Parâmetros:
    y_true -- Lista ou array com os rótulos verdadeiros
    y_pred -- Lista ou array com os rótulos previstos
    labels -- Lista com os nomes das classes
    """

    # Fazendo previsões
    y_true_conv = []
    y_pred_conv = []

    print(len(test_ds))
    test_inputs, y_true = test_ds

    # Fazendo previsões no conjunto de teste
    y_pred = model_conv.predict(test_inputs, verbose=False)
    y_pred = np.argmax(y_pred, axis=1)  # Classes previstas
    y_true = np.argmax(y_true, axis=1)  # Classes verdadeiras

    # Calcular a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Criar dois subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plotar a matriz de confusão não normalizada
    sns_heatmap_1 = sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Confusion Matrix', fontweight='bold',fontsize=16)
    axes[0].set_xlabel('Predicted values', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('True Values', fontsize=16, fontweight='bold')

    # Plotar a matriz de confusão normalizada
    sns_heatmap_2 = sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap=plt.cm.Reds,
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Normalized Confusion Matrix', fontweight='bold', fontsize=16)
    axes[1].set_xlabel('Predicted values', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('True Values', fontsize=16, fontweight='bold')

    # Ajustando o tamanho da fonte das anotações nas matrizes
    if separate == True and hadron == True:
        font_size = 20
    else:
        font_size = 30
    for ax in axes:
        for label in ax.texts:  # Para cada texto nas matrizes
            label.set_fontsize(font_size)  # Altere para o tamanho de fonte desejado
     # Ajustar a fonte do colorbar

    for ax in [sns_heatmap_1, sns_heatmap_2]:
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=20)  # Aumenta o tamanho da fonte do colorbar
        # Ajustando tamanho e rotação dos rótulos dos eixos
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"confusion_matrix.png"))

def plot_confusion_matrix(model_conv, test_ds, labels, output_dir, separate, hadron):

    y_true_conv = []
    y_pred_conv = []

    for img, label in test_ds:
        preds_conv = model_conv.predict(img, verbose=False)
        y_true_conv.extend(np.argmax(label, axis = 1)) 
        y_pred_conv.extend(np.argmax(preds_conv, axis = 1))
    # Fazendo previsões

    cm = confusion_matrix(y_true_conv, y_pred_conv)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Criar dois subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plotar a matriz de confusão não normalizada
    sns_heatmap_1 = sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Confusion Matrix', fontweight='bold',fontsize=16)
    axes[0].set_xlabel('Predicted values', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('True Values', fontsize=16, fontweight='bold')

    # Plotar a matriz de confusão normalizada
    sns_heatmap_2 = sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap=plt.cm.Reds,
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Normalized Confusion Matrix', fontweight='bold', fontsize=16)
    axes[1].set_xlabel('Predicted values', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('True Values', fontsize=16, fontweight='bold')

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
    plt.savefig(os.path.join(output_dir,"confusion_matrix.png"))