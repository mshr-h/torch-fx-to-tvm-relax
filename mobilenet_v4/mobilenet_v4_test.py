import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_fx
import torch
import torch.fx
import torchvision
import timm


def main():
  model_name: str = "mobilenetv4_conv_small.e1200_r224_in1k"

  model_pth = timm.create_model(model_name, pretrained=True)
  model_pth = model_pth.eval()

  # Convert FX Graph to Relax program
  graph_model: torch.fx.GraphModule = torch.fx.symbolic_trace(model_pth)
  print("***FX Graph***")
  print(graph_model.graph)
  print("***Code***")
  print(graph_model.code)
  with torch.no_grad():
    inp: torch.Tensor = torch.rand(1, 3, 224, 224)
    mod: tvm.IRModule = from_fx(graph_model, [(inp.shape, "float32")])


if __name__ == "__main__":
  main()
