import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

# Load and prepare the dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

# Streamlit App
st.title("🌸 Iris Flower Species Predictor")
st.write("Upload an image and enter flower measurements to predict the species.")

# Upload image
uploaded_file = st.file_uploader("Upload a flower image (optional)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Iris Flower", use_column_width=True)

# Feature inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.button("Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"🌼 Predicted Species: **{target_names[prediction].capitalize()}**")
