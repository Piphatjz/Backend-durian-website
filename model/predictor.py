import numpy as np
from PIL import Image
import io
import tflite_runtime.interpreter as tflite

# โหลดโมเดล .tflite
interpreter = tflite.Interpreter(model_path="models/durian.tflite")
interpreter.allocate_tensors()

# ดึงข้อมูล input/output ของโมเดล
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# รายชื่อ class ตามลำดับ index ที่ได้จาก train_generator.class_indices
label_map = {
    0: "ALGAL_LEAF_SPOT",
    1: "ALLOCARIDARA_ATTACK",
    2: "HEALTHY_LEAF",
    3: "LEAF_BLIGHT",
    4: "PHOMOPSIS_LEAF_SPOT",
}

def predict_disease(image_bytes):
    # แปลงภาพจาก bytes เป็น RGB
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    # เตรียมภาพ
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()

    # Get output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])

    disease_label = label_map[predicted_index]
    severity = get_severity(disease_label)

    return {
        "disease": disease_label,
        "confidence": round(confidence * 100),
        "severity": severity
    }

def get_severity(label):
    if label == "HEALTHY_LEAF":
        return "ปกติ"
    elif label in ["ALGAL_LEAF_SPOT", "PHOMOPSIS_LEAF_SPOT"]:
        return "เล็กน้อย"
    elif label == "ALLOCARIDARA_ATTACK":
        return "ปานกลาง"
    elif label == "LEAF_BLIGHT":
        return "รุนแรง"
    return "ไม่ทราบ"
