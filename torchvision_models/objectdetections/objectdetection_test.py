import argparse

import torch
import torch.fx
import torchvision
from torchvision.io import read_image
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_fx


def main(args: argparse.Namespace) -> None:
  model_name = args.model

  weights_enum = torchvision.models.get_model_weights(model_name)
  weights: torchvision.models.Weights = weights_enum.DEFAULT
  torch_model = torchvision.models.get_model(model_name, weights=weights).eval()

  preprocess = weights.transforms()

  img: torch.Tensor = read_image("../grace_hopper_517x606.jpg")
  preprocessed_img = preprocess(img)
  batch: torch.Tensor = preprocessed_img.unsqueeze(0)

  output_torch: torch.Tensor = torch_model(batch)


if __name__ == "__main__":
  parser = argparse.ArgumentParser("torchvision classification example")
  parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=torchvision.models.list_models(module=torchvision.models.detection),
  )
  parser.add_argument("--print_torch_graph", action="store_true", default=False)
  args = parser.parse_args()
  main(args)
