import cv2
import numpy as np
import os
import pycuda.driver as driver
import tensorrt as trt

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
    driver.Context.synchronize()
    buffer_host = [np.ascontiguousarray(data, dtype=np.float32), np.empty(context.get_binding_shape(_OUTPUT_INDEX), dtype=trt.nptype(engine.get_binding_dtype(_OUTPUT_INDEX)))]
    buffer_device = []
    for idx in range(_NUM_IO): buffer_device.append(driver.mem_alloc(buffer_host[idx].nbytes))
    driver.memcpy_htod(buffer_device[0], buffer_host[0])
    context.execute_v2(buffer_device)
    driver.memcpy_dtoh(buffer_host[1], buffer_device[1])
    prediction = np.array(buffer_host[1])
    driver.Context.synchronize()
    return prediction


def _postprocess(raw_prediction, pred_path, idx):
    prediction = np.argmax(raw_prediction, axis=1)
    np.save(pred_path + "/{}.npy".format(idx), prediction)


def infer(file_path: str, pred_path: str):
    print("Inference step | File path : {} | Prediction path : {}.".format(file_path, pred_path))
    logger = trt.Logger(trt.Logger.VERBOSE)
    engine_data = _import_engine()
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    context.set_binding_shape(0, _BINDING_SHAPE)
    images = _read_images(file_path)
    for idx, image in enumerate(images):
        curr_raw_prediction = _inference(engine, context, image)
        _postprocess(curr_raw_prediction, pred_path, idx)
