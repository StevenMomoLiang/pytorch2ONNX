import torch
import os
import argparse
from src import deeplabv3_resnet50,deeplabv3_resnet101
parser = argparse.ArgumentParser()
parser.add_argument("--pth_path", default='D:/python/project/pythonProject2/save_weights/model_29.pth')
#parser.add_argument("--pth_path", default='D:/python/project/pythonProject2/model/deeplabv3_resnet50_coco.pth')
parser.add_argument("--save_onnx_path", default='./ONNX_model/test.onnx')
parser.add_argument("--input_width", default=500)
parser.add_argument("--input_height", default=513)
parser.add_argument("--input_channel", default=3)
parser.add_argument("--num_classes", default=20)
args = parser.parse_args()
def pth_to_onnx(pth_path, onnx_path, in_hig, in_wid, in_channel, num_cls):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0
    model = deeplabv3_resnet50(True, num_classes=num_cls+1)
    model.load_state_dict(torch.load(pth_path),strict=False)
    #weights_dict = torch.load(pth_path,map_location='cpu')

    #model.load_state_dict(weights_dict)
    model.eval()


    print(f'{pth_path} model loaded')
    input_names = ['input']
    output_names = ['output']
    im = torch.rand(1, in_channel, in_hig, in_wid)
    torch.onnx.export(model, im, onnx_path,
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=11)
    print("Exporting .pth model to onnx model has been successful!")
    print(f"Onnx model save as {onnx_path}")
if __name__ == '__main__':
    pth_to_onnx(pth_path=args.pth_path,
                onnx_path=args.save_onnx_path,
                in_hig=args.input_height,
                in_wid=args.input_width,
                in_channel=args.input_channel,
                num_cls=args.num_classes)