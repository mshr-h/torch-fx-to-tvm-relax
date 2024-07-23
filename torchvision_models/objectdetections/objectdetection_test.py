import argparse

import torch
import torch.fx
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_fx
from PIL import Image


def main(args: argparse.Namespace) -> None:
  model_name = args.model

  weights_enum = torchvision.models.get_model_weights(model_name)
  weights: torchvision.models.Weights = weights_enum.DEFAULT
  torch_model = torchvision.models.get_model(
    model_name, weights=weights, box_score_thresh=0.9
  ).eval()

  preprocess = weights.transforms()

  img: torch.Tensor = read_image("../grace_hopper_517x606.jpg")
  preprocessed_img = preprocess(img)
  batch: torch.Tensor = preprocessed_img.unsqueeze(0)

  output_torch: torch.Tensor = torch_model(batch)[0]
  labels = [weights.meta["categories"][i] for i in output_torch["labels"]]
  box = draw_bounding_boxes(
    img,
    boxes=output_torch["boxes"],
    labels=labels,
    colors="red",
    width=4,
    font="NotoMono-Regular.ttf",
    font_size=30,
  )
  to_pil_image(box).save("torch_grace_hopper_517x606.jpg")

  # tvm
  graph_model: torch.fx.GraphModule = torch.fx.symbolic_trace(torch_model)


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
