import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load ONNX model
@st.cache_resource
def load_model():
    return ort.InferenceSession("mnist_model.onnx")

session = load_model()

# UI
st.title("✍️ Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) in the box below")

# Drawing canvas
canvas = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict"):
    if canvas.image_data is not None:
        # Process image
        img = Image.fromarray(canvas.image_data.astype("uint8"))
        img = img.convert("L")           # grayscale
        img = img.resize((28, 28))       # resize to MNIST size
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = img_array.flatten().reshape(1, 784)

        # Predict using ONNX
        input_name = session.get_inputs()[0].name
        prediction = session.run(None, {input_name: img_array})[0]

        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Predicted Digit: **{digit}**")
        st.write(f"Confidence: **{confidence:.2%}**")

        # Bar chart
        st.bar_chart({str(i): float(prediction[0][i]) for i in range(10)})
    else:
        st.warning("Please draw a digit first!")