# 🩺 Pneumonia Detection using Deep Learning

This project uses deep learning techniques to detect pneumonia from chest X-ray images. Early diagnosis can save lives, and this automated approach aims to assist clinicians in identifying pneumonia efficiently.

## 🔍 Project Overview

- **Model Architecture**: VGG16 (pre-trained on ImageNet)
- **Transfer Learning**: Only custom top layers are trained; all convolutional layers are frozen
- **Dataset**: Chest X-ray images labeled as `NORMAL` or `PNEUMONIA`
- **Platform**: Built for use in Kaggle or Colab

## 🛠️ Technologies Used

- **Python & Keras**
- **VGG16**: CNN for feature extraction
- **ImageDataGenerator**: For rescaling and augmentation
- **Matplotlib**: For accuracy/loss visualization
- **NumPy & SciPy**: For preprocessing and predictions
- **KaggleHub**: To fetch public datasets

## 🧠 Model Pipeline

```text
[Chest X-Ray Dataset]
   ↓
[Preprocessing]
   ↓
[Train-Test Split]
   ↓
[Load VGG16 (no top)]
   ↓
[Freeze Layers]
   ↓
[Add Flatten + Dense + Softmax]
   ↓
[Compile Model (Adam + CrossEntropy)]
   ↓
[Train for 5 Epochs]
   ↓
[Evaluate]
   ↓
[Save & Load Model]
   ↓
[Predict on New X-ray]
   ↓
[Classification Output: NORMAL or PNEUMONIA]
```

## 🚀 How to Run

### 1. Set Up Dataset
Use `kagglehub` to download:
```python
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
```

### 2. Build and Compile Model
```python
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg_model.layers:
    layer.trainable = False
```

Add custom layers:
```python
x = Flatten()(vgg_model.output)
x = Dense(8, activation='relu')(x)
output = Dense(2, activation='softmax')(x)
model = Model(inputs=vgg_model.input, outputs=output)
```

Compile:
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 3. Load Data
```python
training_set = train_datagen.flow_from_directory(..., class_mode='categorical')
testing_set = test_datagen.flow_from_directory(..., class_mode='categorical')
```

### 4. Train and Evaluate
```python
model.fit(training_set, validation_data=testing_set, epochs=5)
model.save('our_model.h5')
```

### 5. Predict
```python
img = image.load_img(image_path, target_size=(224, 224))
image_data = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
prediction = model.predict(image_data)
```

## 📈 Visualization

Accuracy and loss curves for training and validation are plotted using `matplotlib`.

## 🔮 Future Enhancements

- ✅ **Add more CNN architectures** like ResNet or EfficientNet for better performance
- ✅ **Hyperparameter tuning** for optimal learning rates and batch sizes
- ✅ **Deploy model as a web app** using Flask or Streamlit
- ✅ **Integrate Grad-CAM visualizations** to interpret model predictions
- ✅ **Enable ensemble learning** by combining predictions from multiple models
- ✅ **Build mobile-friendly diagnostic tools** for use in remote areas
- ✅ **Expand dataset** to include multiple pneumonia stages or other diseases

---

## 📁 License

This project is for educational and research purposes. Refer to the source datasets’ licensing terms on [Kaggle](https://www.kaggle.com).
