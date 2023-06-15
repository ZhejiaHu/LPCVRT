import cv2
import numpy as np
import os
import time
import torch


_MEAN, _STD = np.array([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis], np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis]


def _read_images(file_path):
    return map(
        lambda img_file: np.expand_dims((np.transpose(cv2.cvtColor(cv2.imread(file_path + "/{}".format(img_file)), cv2.COLOR_BGR2RGB).astype(np.float32), (2, 0, 1)) / 255 - _MEAN) / _STD, axis=0),
        sorted(os.listdir(file_path))
    )


def infer(model_path, file_path, eval_path):
    print("Inference step | File path : {} | Evaluation path : {}.".format(file_path, eval_path))
    start_time = time.time()
    model = torch.load(model_path)
    model.eval()
    images = _read_images(file_path)
    for idx, image in enumerate(images):
        print("Inferring without TensorRT | Image {}".format(idx))
        raw_prediction = model(image)
        prediction = np.argmax(raw_prediction, axis=1)
        np.save(eval_path + "/{}.npy".format(idx), prediction)
    print("--- Without TensorRT Inference: {} seconds ---" .format(time.time() - start_time))




