import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

"""Questa funzione stampa a schermo la confusion matrix
input args: 
    generator -> il set creato con ImageDataGenerator.flow_from_directory
    model -> il modello fittato
    labels -> una lista che contenga i nomi delle classi ordinati sequenzialmente
    n_of_batches -> numero di batch da estrapolare da generator
    batch_size -> dimensione dei batch creati con ImageDataGenerator.flow_from_directory
"""


def get_confusion_matrix(generator, model, labels, n_of_batches=200, batch_size=8):
    prediction_arr = []
    target_arr = []

    for ext_num in range(n_of_batches):
        batch = next(generator)

        image = batch[0]
        target = batch[1]

        # print("(Input) image shape:", image.shape)
        # print("Target shape:", target.shape)

        prediction = model.predict(image, batch_size=batch_size)
        print()
        for num in range(batch_size):
            image_sample = image[num]
            target_sample = target[num]
            target_idx = np.argmax(target_sample)
            prediction_arr.append(np.argmax(prediction[num]))
            target_arr.append(target_idx)

    cm = confusion_matrix(target_arr, prediction_arr)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm.T, xticklabels=labels, yticklabels=labels)
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.show()

    return 0


"""Questa funzione stampa a schermo statistiche su singole prediction del model
input args: 
    model -> il modello fittato
    labels -> una lista che contenga i nomi delle classi ordinati sequenzialmente
    batch -> Batch prodotto utilizzando next su un oggetto di tipo ImageDataGenerator.flow_from_directory
    n_of_images -> Numero di immagini di cui guardare le statistiche
    batch_size -> batch size usata in ImageDataGenerator.flow_from_directory
"""


def get_prediction_statistics(model, labels, batch, n_of_images=1, batch_size=8):
    image = batch[0]
    target = batch[1]
    prediction = model.predict(image, batch_size=batch_size)
    for num in range(n_of_images):
        image_sample = image[num]
        target_sample = target[num]
        target_idx = np.argmax(target_sample)

        print("Categorical label:", target_sample)
        print("Predicted output: ", prediction[num])
        print("Label:", target_idx)
        print("Class name:", labels[target_idx])
        print("Predicted class name: ", labels[np.argmax(prediction[num])])
        plt.figure(figsize=(6, 4))
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(15, 5)
        # Assumo che le immagini siano grayscale con normalization [0,1]
        ax1.imshow(np.uint8(image_sample * 255), cmap='gray', vmin=0, vmax=255)
        ax1.set_title('True label: ' + labels[target_idx])
        ax2.barh(labels, prediction[num], color=plt.get_cmap('Paired').colors)
        ax2.set_title('Predicted label: ' + labels[np.argmax(prediction[num])])
        ax2.grid(alpha=.3)
        plt.show()
    return


"""Questa funzione stampa a schermo il grafico di training e validation set in merito a loss e accuracy
input args: 
    history -> la history del fitting del modello
"""


def plot_early_stopping(history):
    # Plot the training
    plt.figure(figsize=(15, 5))
    plt.plot(history['loss'], label='Training', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_loss'], label='Validation', alpha=.8, color='#4D61E2')
    plt.legend(loc='upper left')
    plt.title('Binary Crossentropy')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15, 5))
    plt.plot(history['categorical_accuracy'], label='Training', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_categorical_accuracy'], label='Validation', alpha=.8, color='#4D61E2')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)

    plt.show()
