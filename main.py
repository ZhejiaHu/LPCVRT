import argparse

import inference
import preprocess_


def main(model_path: str, file_path: str, pred_path: str, preprocess: bool) -> None:
    if preprocess: preprocess_.preprocess(model_path)
    inference.infer(file_path, pred_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model.pth")
    parser.add_argument("--file_path", type=str, default="val/")
    parser.add_argument("--pred_path", type=str, default="pred/")
    parser.add_argument("--preprocess", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args.model_path, args.file_path, args.pred_path, args.preprocess)