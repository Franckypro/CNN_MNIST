# CNN_MNIST

# Classification d'images avec un CNN 🖼️🤖

Ce projet utilise un **réseau de neurones convolutifs (CNN)** pour la classification d'images du dataset **MNIST**. Ce README détaille les étapes nécessaires pour exécuter ce projet, de la préparation de l'environnement à la prédiction des nouvelles images.

## 1️⃣ Préparation de l’environnement 🚀

### Installation des bibliothèques nécessaires 📦

Installez les bibliothèques suivantes en exécutant les commandes :

```bash
pip install tensorflow
pip install matplotlib
pip install numpy
```

### Importation des bibliothèques 📚

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
```

## 2️⃣ Chargement et exploration du dataset 📊

### Chargement des données MNIST

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

- Les images sont normalisées pour faciliter l'apprentissage.
- Une dimension est ajoutée pour correspondre aux entrées du CNN.

### Visualisation des données augmentées

```python
train_datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=False, fill_mode='nearest'
)

fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    augmented_image = train_datagen.random_transform(x_train[i])
    axes[i].imshow(augmented_image.squeeze(), cmap='gray')
    axes[i].axis('off')
plt.show()
```

## 3️⃣ Création du modèle CNN 🧠

Le modèle CNN est composé de deux couches de convolution et de max-pooling pour extraire les caractéristiques des images, puis de couches denses pour effectuer la classification.

```python
classifier = Sequential([
    Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=128, activation="relu"),
    Dense(units=10, activation="softmax")
])
```

## 4️⃣ Compilation et entraînement ⚡

Le modèle est compilé avec l'optimiseur `adam` et la fonction de perte `sparse_categorical_crossentropy`.

```python
classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
history = classifier.fit(
    train_datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    epochs=5
)
```

## 5️⃣ Évaluation du modèle 📈

### Tracer les courbes de précision et de perte

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Précision entraînement')
plt.plot(history.history['val_accuracy'], label='Précision validation')
plt.xlabel('Époques')
plt.ylabel('Précision')
plt.legend()
plt.title('Évolution de la précision')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte entraînement')
plt.plot(history.history['val_loss'], label='Perte validation')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.legend()
plt.title('Évolution de la perte')

plt.show()
```

### Précision sur le test set 🎯

```python
test_loss, test_acc = classifier.evaluate(x_test, y_test, verbose=2)
print(f"\nPrécision sur le test set : {test_acc * 100:.2f}%")
```

## 6️⃣ Prédiction sur de nouvelles images 🔮

Sélectionnez une image au hasard, effectuez la prédiction et comparez-la à la vraie étiquette.

```python
index = random.randint(0, len(x_test))  
test_image = x_test[index]

plt.imshow(test_image.squeeze(), cmap='gray')
plt.axis('off')
plt.title("Image test")
plt.show()

test_image = np.expand_dims(test_image, axis=0)
prediction = classifier.predict(test_image)
predicted_label = np.argmax(prediction)

print(f"Prédiction du modèle : {predicted_label}")
print(f"Vraie étiquette : {y_test[index]}")
```

Si la prédiction est correcte, un emoji ✅ sera affiché, sinon un emoji ❌ sera affiché.

## Bonus : Améliorations possibles 🎁

Vous pouvez utiliser un modèle pré-entraîné tel que **MobileNetV2** pour améliorer les performances du modèle.

### Chargement et transformation des données

Les images sont redimensionnées à `224x224` pour être compatibles avec MobileNetV2, et des transformations sont appliquées.

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

x_train = np.repeat(x_train, 3, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)
```

### Construction du modèle avec MobileNetV2

```python
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False

model = Sequential([
    tf.keras.layers.Resizing(224, 224),
    base_model,
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])
```

### Entraînement avec Data Augmentation

```python
history = model.fit(training_set, validation_data=testing_set, epochs=5)
```

### Visualisation des performances

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Précision entraînement')
plt.plot(history.history['val_accuracy'], label='Précision validation')
plt.xlabel('Époques')
plt.ylabel('Précision')
plt.legend()
plt.title('Évolution de la précision')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte entraînement')
plt.plot(history.history['val_loss'], label='Perte validation')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.legend()
plt.title('Évolution de la perte')

plt.show()
```

### Test sur une image aléatoire

```python
idx = random.randint(0, len(x_test) - 1)
test_image = x_test[idx]
test_image_resized = tf.image.resize(test_image, (224, 224))
test_image_resized = np.expand_dims(test_image_resized, axis=0)

prediction = model.predict(test_image_resized)
predicted_label = np.argmax(prediction)

plt.imshow(test_image.squeeze(), cmap='gray')
plt.axis('off')
plt.title(f"Prédiction du modèle: {predicted_label}")
plt.show()
```

## Conclusion 🎉

Ce modèle atteint une précision de **98.47%** sur le test set avec le CNN classique et **56.55%** avec le modèle pré-entraîné MobileNetV2. Vous pouvez améliorer encore ce modèle en ajustant les hyperparamètres ou en fine-tunant les couches du modèle pré-entraîné.

Merci d'avoir consulté ce projet ! 😃
```

N'hésitez pas à me faire savoir si vous avez besoin d'autres détails ou modifications. 😊
```
✍️ Auteur
Fouejio Francky Joël
