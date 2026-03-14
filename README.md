# 🔢 Digit Classification CNN — MNIST Handwritten Digit Recognizer

A deep Convolutional Neural Network (CNN) built from scratch using TensorFlow/Keras to classify handwritten digits (0–9) from the MNIST benchmark dataset. This project was developed as part of coursework/research at NIT Puducherry (NITPY).

---

## 📌 Project Overview

This project implements **Multi-Class Digit Classification** using a 3-block CNN architecture trained on the **MNIST** dataset. The model takes a **28×28 grayscale image** as input and predicts one of 10 digit classes (0 through 9).

| Detail | Value |
|---|---|
| Task | Multi-class image classification |
| Dataset | MNIST Handwritten Digits |
| Input | 28×28 grayscale image |
| Output | Digit label (0–9) |
| Architecture | 3-block CNN + Dense layers with Dropout |
| Framework | TensorFlow / Keras |

---

## 📂 Dataset — MNIST

- **Source**: Built-in via `tf.keras.datasets.mnist` — no manual download required
- **Total Images**: 70,000 grayscale images (28×28 pixels, single channel)
- **Classes**: 10 digit categories (0 through 9)

| Label | Class | Label | Class |
|---|---|---|---|
| 0 | Zero | 5 | Five |
| 1 | One | 6 | Six |
| 2 | Two | 7 | Seven |
| 3 | Three | 8 | Eight |
| 4 | Four | 9 | Nine |

### Dataset Split

| Split | Samples | Shape | Description |
|---|---|---|---|
| X_train | 60,000 | (60000, 28, 28, 1) | Training images, normalized |
| y_train | 60,000 | (60000,) | Integer digit labels |
| X_test | 10,000 | (10000, 28, 28, 1) | Test images, normalized |
| y_test | 10,000 | (10000,) | Integer digit labels |

> Images are reshaped from `(N, 28, 28)` to `(N, 28, 28, 1)` to add the channel dimension.  
> Pixel values are normalized to **[0, 1]** by dividing by 255.

---

## 🧠 Model Architecture

A 3-block Sequential CNN with increasing filter depth, followed by dense layers and dropout regularization:

```
Input: (28, 28, 1) — 28×28 Grayscale image

── Block 1 ──────────────────────────────
Conv2D(32 filters, 3×3, ReLU)
MaxPooling2D(2×2)

── Block 2 ──────────────────────────────
Conv2D(64 filters, 3×3, ReLU)
MaxPooling2D(2×2)

── Block 3 ──────────────────────────────
Conv2D(128 filters, 3×3, ReLU)
MaxPooling2D(2×2)

── Classifier Head ──────────────────────
Flatten()
Dense(128, ReLU)
Dropout(0.5)              ← Regularization to prevent overfitting
Dense(64, ReLU)
Dense(10, Softmax)        ← 10-class output probabilities
```

- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Output Activation**: Softmax — outputs a probability distribution over all 10 digit classes
- **Prediction**: `argmax(output)` → predicted digit

> The 3-block design with progressively increasing filters (32 → 64 → 128) allows the network to learn simple edges in early layers and complex digit structures in deeper layers. Dropout(0.5) after the first Dense layer significantly reduces overfitting on MNIST.

---

## 📈 Training Results

Training was performed for **10 epochs** with validation on the test set after each epoch:

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|---|---|---|---|---|
| 1  | ~97% | ~98% | ~0.10 | ~0.07 |
| 2  | ~98% | ~98% | ~0.06 | ~0.05 |
| 3  | ~99% | ~99% | ~0.04 | ~0.04 |
| 4  | ~99% | ~99% | ~0.03 | ~0.04 |
| 5  | ~99% | ~99% | ~0.03 | ~0.04 |
| 6  | ~99% | ~99% | ~0.02 | ~0.04 |
| 7  | ~99% | ~99% | ~0.02 | ~0.04 |
| 8  | ~99% | ~99% | ~0.02 | ~0.04 |
| 9  | ~99% | ~99% | ~0.02 | ~0.04 |
| 10 | ~99% | ~99% | ~0.02 | ~0.04 |

> **Test Accuracy**: ~99% | **Test Loss**: ~0.04  
> *(Update these with your actual notebook output)*

---

## 📊 Evaluation

The notebook includes a full evaluation pipeline:

- ✅ Test accuracy and test loss (via `model.evaluate`)
- 📉 **Confusion Matrix** — 10×10 heatmap (Seaborn)
- 📋 **Classification Report** — precision, recall, F1-score per digit class (4 decimal precision)
- 🖼️ **Random Image Prediction** — displays test image with true vs. predicted digit
- 🔢 **Per-Class TP / FP / FN / TN breakdown** for all 10 digit classes

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core programming language |
| TensorFlow | 2.x | Deep learning framework |
| Keras | (via TF) | High-level neural network API |
| NumPy | latest | Array manipulation |
| Matplotlib | latest | Image & training plot visualization |
| Seaborn | latest | Confusion matrix heatmap |
| scikit-learn | latest | Metrics: confusion matrix, classification report |

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/phanendra-teja/DigitClassificationCNN.git
cd DigitClassificationCNN
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy
```

### 4. Run the Notebook

```bash
jupyter notebook DigitClassificationCNN.ipynb
```

> ✅ MNIST is **automatically downloaded** on first run via `tf.keras.datasets` — no manual dataset setup needed.

---

## 📁 Project Structure

```
DigitClassificationCNN/
│
├── DigitClassificationCNN.ipynb    # Main notebook with full model pipeline
├── .gitignore
└── README.md                       # This file
```

---

## 🔮 Sample Prediction

After training, the model predicts on a random test image:

```python
import random, numpy as np

def predict_random_image(model, X_data, y_data):
    index = random.randint(0, len(X_data) - 1)
    img = X_data[index]
    true_label = y_data[index]

    pred = model.predict(img.reshape(1, 28, 28, 1))
    pred_label = np.argmax(pred)

    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"True: {true_label} | Predicted: {pred_label}")
    plt.axis("off")
    plt.show()

predict_random_image(cnn, X_test, y_test)
```

---

## 📌 .gitignore (Recommended)

```
__pycache__/
*.pyc
.venv/
*.h5
*.keras
```

---

## 🎯 Future Improvements

- Add **Training History Plots** (accuracy & loss curves over epochs) for visual analysis of model convergence
- Add **Data Augmentation** (shifts, rotations, zoom) to make the model robust to slightly distorted handwriting
- Experiment with **Batch Normalization** after convolutional blocks for faster, more stable training
- Try **LeNet-5** or **ResNet-inspired** architectures for comparison
- Save and reload the trained model using `.h5` or `SavedModel` format for deployment
- Build a **live drawing demo** using Tkinter or a web app (Flask/Streamlit) where a user draws a digit and the model predicts it in real time
- Extend to the **EMNIST** dataset (letters + digits) for a broader character recognition system

---

## 👤 Author

**Phanendra Teja V**  
B.Tech CSE — NIT Puducherry (NITPY), Batch 2024–2028

---

## 📄 License

This project is for educational purposes.  
Dataset: MNIST — available publicly via `tf.keras.datasets`.
