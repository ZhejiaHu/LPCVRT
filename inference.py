import cv2
import numpy as np
import os
import pycuda.driver as driver
import pycuda.autoinit
import tensorrt as trt
import time

import torch

_PLAN_PATH = "model.plan"
_BINDING_SHAPE = trt.Dims([1, 3, 512, 512])
_MEAN, _STD = np.array([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis], np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis]
_NUM_IO, _NUM_INPUT, _OUTPUT_INDEX = 2, 1, 1


def _import_engine():
    with open(_PLAN_PATH, "rb") as f: engine_data = f.read()
    assert engine_data is not None
    return engine_data


def _read_images(file_path):
    return map(
        lambda img_file: (np.transpose(cv2.cvtColor(cv2.imread(file_path + "/{}".format(img_file)), cv2.COLOR_BGR2RGB).astype(np.float32), (2, 0, 1)) / 255 - _MEAN) / _STD,
        sorted(os.listdir(file_path))
    )


def _inference(engine, context, data):
    #buffer_host = [np.ascontiguousarray(data, dtype=np.float32), np.empty(context.get_binding_shape(_OUTPUT_INDEX), dtype=trt.nptype(engine.get_binding_dtype(_OUTPUT_INDEX)))]
    input_tensor = torch.empty([1, 3, 512, 512], dtype=torch.float32, device=torch.device("cpu"))
    input_tensor = input_tensor.cuda().contiguous()
    output_tensor = torch.empty([1, 14, 512, 512], dtype=torch.float32, device=torch.device("cpu"))
    output_tensor = output_tensor.cuda()
    #print(data)
    input_tensor.data.copy_(torch.from_numpy(data))
    print(input_tensor.dtype)
    print(input_tensor)
    start_time = time.time()
    print("Execution starts")
    context.execute_v2(bindings=[int(input_tensor.data_ptr()), int(output_tensor.data_ptr())])
    print("Execution ends")
    end_time = time.time()
    #driver.memcpy_dtoh(buffer_host[1], buffer_device[1])
    print(output_tensor)
    prediction = output_tensor.cpu().numpy()
    print(prediction)
    assert np.any(prediction != 0)
    #driver.Context.synchronize()
    return prediction, end_time - start_time


def _postprocess(raw_prediction, pred_path, idx):
    prediction = np.argmax(raw_prediction, axis=1)
    np.save(pred_path + "/{}.npy".format(idx), prediction)


def infer(file_path: str, pred_path: str):
    #exp_tensor = torch.Tensor([1, 2, 3])
    #print("Example tensor CPU address | {}".format(exp_tensor.data_ptr()))
    #exp_tensor = exp_tensor.to(torch.device("cuda"))
    #print("Example tensor GPU address | {}".format(exp_tensor.data_ptr()))
    print("Inference step | File path : {} | Prediction path : {}.".format(file_path, pred_path))
    logger = trt.Logger(trt.Logger.VERBOSE)
    engine_data = _import_engine()
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    print(engine.__dir__())
    print(context.__dir__())
    images = _read_images(file_path)
    acc_time = 0
    for i in range(2):
        print(engine.get_binding_dtype(i))
        print(engine.get_binding_shape(i))
        print(engine.get_binding_vectorized_dim(i))
        print("-----------------------------------")
        print(context.get_binding_shape(i))
    for idx, image in enumerate(images):
        print("Inferring with TensorRT | Image {}".format(idx))
        curr_raw_prediction, curr_time = _inference(engine, context, image)
        acc_time += curr_time
        _postprocess(curr_raw_prediction, pred_path, idx)
    print("--- With TensorRT Inference: {} seconds ---" .format(acc_time))
