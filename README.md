# MNIST_Classification
This project uses a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. It demonstrates key deep learning concepts and achieves over 98% accuracy.
# MNIST Handwritten Digit Classification ✍️

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the famous MNIST dataset. The goal is to build a robust deep learning model capable of accurately recognizing digits from 0 to 9.

---

## Key Features ✨

* **CNN Architecture**: A custom CNN model built with Keras, leveraging `Conv2D`, `MaxPooling`, and `Dropout` layers for robust feature extraction.
* **Data Preprocessing**: Images are normalized, and labels are one-hot encoded for multi-class classification.
* **Data Augmentation**: The `ImageDataGenerator` is used to augment the training data, helping the model generalize better and reduce overfitting.
* **Adaptive Learning Rate**: A `ReduceLROROnPlateau` callback is implemented to automatically adjust the learning rate during training, optimizing convergence.
* **Optimization**: The model is compiled with the RMSprop optimizer and categorical cross-entropy loss.

---

## Key Decisions and Rationale 🧠

* **Normalization**: We normalized the pixel values from the `0-255` range to `0-1`. This is a crucial preprocessing step that helps the model train faster and more efficiently.
* **CNN Architecture**: We chose a CNN because these networks excel at finding patterns and hierarchies in visual data. The `Conv2D` layers extract features, `MaxPooling` reduces complexity, and the final `Dense` layers perform the classification.
* **One-Hot Encoding**: We converted our integer labels (`0` to `9`) into one-hot encoded vectors (e.g., `[0,0,1,0,...]`). This is a necessary step to match the output format of the model's final `softmax` layer, allowing us to use categorical cross-entropy loss.
* **Optimizer & Loss**: We used **RMSprop** and **categorical cross-entropy**. The optimizer is an efficient algorithm that helps the model converge quickly, while the loss function is the standard for measuring the error in a multi-class classification problem.
* **Regularization**: We implemented a **Dropout** layer to randomly deactivate neurons during training. This technique prevents the model from relying too heavily on any single feature, which forces it to generalize better and combats overfitting.

---

## Results 🏆

The trained model achieved a final validation accuracy of over 98%, demonstrating strong performance on the Kaggle leaderboard.

---

## Technologies Used 💻

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-learn
