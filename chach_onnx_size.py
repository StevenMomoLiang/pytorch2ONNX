import onnxruntime
import cv2


onnx_model_path = "./ONNX_model/model_29.onnx"
#img_path = "./446B7722-238E-4E14-88B7-D2E0B46C6A53.jpg"

image = cv2.imread('./446B7722-238E-4E14-88B7-D2E0B46C6A53.jpg')
dimensions = image.shape

hight = image.shape[0]
width = image.shape[1]
channel = image.shape[2]


session = onnxruntime.InferenceSession(onnx_model_path)



input_info = session.get_inputs()[0]
input_name = input_info.name
input_shape = input_info.shape

print(f"Input name: {input_name}")
print(f"Input shape: {input_shape}")

print(hight,width,channel)
