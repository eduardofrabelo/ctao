import os
import matplotlib.pyplot as plt

def plot_metrics_curve(output_path, hist):
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))

    # Iterando sobre as colunas do DataFrame para gerar as curvas
    for column in hist.columns:
        if "accuracy" in column:
            ax[0].plot(hist.index, hist[column] * 100, label=column)
            ax[0].legend(loc='best', fontsize=14)  # Tamanho da fonte da legenda
        elif "mae" in column:
            ax[2].plot(hist.index, hist[column], label=column)
            ax[2].legend(loc='best', fontsize=14)  # Tamanho da fonte da legenda
        else:
            ax[1].plot(hist.index, hist[column], label=column)
            ax[1].legend(loc='best', fontsize=14)  # Tamanho da fonte da legenda

    ax[0].set_title("Accuracy vs Epoch Curve", fontsize=16, fontweight='bold')
    ax[0].set_xlabel("Epochs", fontsize=14, fontweight='bold')
    ax[0].set_ylabel("Accuracy [%]", fontsize=14, fontweight='bold')

    ax[1].set_title("Loss vs Epoch Curve", fontsize=16, fontweight='bold')
    ax[1].set_xlabel("Epochs", fontsize=14, fontweight='bold')
    ax[1].set_ylabel("Loss", fontsize=14, fontweight='bold')

    ax[2].set_title("Mean Absolute Error vs Epoch", fontsize=16, fontweight='bold')
    ax[2].set_xlabel("Epochs", fontsize=14, fontweight='bold')
    ax[2].set_ylabel("MAE", fontsize=14, fontweight='bold')

    tick_fontsize = 20 
    ax[0].tick_params(axis='x', labelsize=tick_fontsize)
    ax[0].tick_params(axis='y', labelsize=tick_fontsize)
    ax[1].tick_params(axis='x', labelsize=tick_fontsize)
    ax[1].tick_params(axis='y', labelsize=tick_fontsize)
    ax[2].tick_params(axis='x', labelsize=tick_fontsize)
    ax[2].tick_params(axis='y', labelsize=tick_fontsize)

    # Ajustando o layout para que tudo se encaixe bem
    plt.tight_layout()

    # Salvando o gráfico com alta resolução
    plt.savefig(os.path.join(output_path, "loss_and_accuracy_curve.png"))