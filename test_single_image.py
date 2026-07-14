import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load trained model
# -----------------------------
model = tf.keras.models.load_model("anemia_nail_cnn_model_lr0001.keras")

# -----------------------------
# 2. Test image path
# -----------------------------
img_path = "test_image.jpg"   # change if needed

# -----------------------------
# 3. Load and preprocess image
# -----------------------------
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# -----------------------------
# 4. Predict
# -----------------------------
prediction = model.predict(img_array, verbose=0)[0][0]

# -----------------------------
# 5. Print result first
# -----------------------------
print("Prediction Score:", round(float(prediction), 4))

if prediction >= 0.5:
    print("Result: Non-Anemic")
    print("Confidence:", round(float(prediction) * 100, 2), "%")
else:
    print("Result: Anemic")
    print("Confidence:", round((1 - float(prediction)) * 100, 2), "%")

# -----------------------------
# 6. Show image after result
# -----------------------------
plt.imshow(img)
plt.title("Testing Image")
plt.axis("off")
plt.show()