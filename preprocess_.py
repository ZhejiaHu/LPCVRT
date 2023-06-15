import subprocess
import tensorrt as trt
import torch

_ONNX_PATH, _ONNX_FOLDED_PATH, _PLAN_PATH = "model.onnx", "model_folded.onnx", "model.plan"
_INPUT_SHAPE, _OUTPUT_SHAPE = [1, 3, 512, 512], [1, 14, 512, 512]


def _convert_model(path):
    model = torch.load(path)
    model.eval()
    torch.onnx.export(model, torch.randn(1, 3, 512, 512), _ONNX_PATH, verbose=True)


def _fold_model():
    command = "polygraphy surgeon sanitize --fold-constants {} -o {}".format(_ONNX_PATH, _ONNX_FOLDED_PATH)
    subprocess.run(command)


def _create_engine():
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)
    with open(_ONNX_FOLDED_PATH, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            exit()
        print("Succeeded parsing .onnx file!")
    input_tensor = network.get_input(0)
    profile.set_shape(input_tensor.name, _INPUT_SHAPE, _INPUT_SHAPE, _INPUT_SHAPE)
    config.add_optimization_profile(profile)
    engine_string = builder.build_serialized_network(network, config)
    assert engine_string is not None
    with open(_PLAN_PATH, "wb") as f:
        f.write(engine_string)


def preprocess(model_path: str) -> None:
    print("Preprocessing step | Model path : {}.".format(model_path))
    _convert_model(model_path)
    _fold_model()
    _create_engine()





