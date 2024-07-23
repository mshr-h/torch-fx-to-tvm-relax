import argparse

import torch
import torch.fx
import torchvision
from torchvision.io import read_image
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_fx


def postprocess(output: torch.Tensor, categories) -> None:
  prediction_torch: torch.Tensor = output.squeeze(0).softmax(0)
  class_id: int = prediction_torch.argmax().item()
  score: float = prediction_torch[class_id].item()
  category_name: str = categories[class_id]
  print(f"{category_name}: {100 * score:.1f}%")


def main(args: argparse.Namespace) -> None:
  model_name = args.model

  weights_enum = torchvision.models.get_model_weights(model_name)
  weights: torchvision.models.Weights = weights_enum.DEFAULT
  torch_model = torchvision.models.get_model(model_name, weights=weights).eval()

  preprocess = weights.transforms(antialias=True)

  img: torch.Tensor = read_image("../grace_hopper_517x606.jpg")
  preprocessed_img = preprocess(img)
  batch: torch.Tensor = preprocessed_img.unsqueeze(0)

  output_torch: torch.Tensor = torch_model(batch)
  postprocess(output_torch, weights.meta["categories"])

  # tvm
  graph_model: torch.fx.GraphModule = torch.fx.symbolic_trace(torch_model)
  if args.print_torch_graph:
    print(graph_model.print_readable())

  with torch.no_grad():

    def convert_stochastic_depth(
      node: torch.fx.node.Node,
      fx_importer: tvm.relax.frontend.torch.fx_translator.TorchFXImporter,
    ) -> relax.Var:
      args = fx_importer.retrieve_args(node)
      return fx_importer.block_builder.emit(args[0])

    input_info = (
      [(batch.squeeze(0).shape, "float32")]
      if model_name in ["googlenet", "inception_v3"]
      else [(batch.shape, "float32")]
    )
    mod: tvm.IRModule = from_fx(
      graph_model,
      input_info,
      custom_convert_map={"stochastic_depth": convert_stochastic_depth},
    )

  target: tvm.target.Target = tvm.target.Target("llvm", host="llvm")
  mod: tvm.IRModule = relax.transform.DecomposeOpsForInference()(mod)
  mod: tvm.IRModule = relax.transform.LegalizeOps()(mod)
  ex: relax.Executable = relax.build(mod, target)
  vm: relax.VirtualMachine = relax.VirtualMachine(ex, tvm.cpu())

  input_tvm: tvm.nd.NDArray = tvm.nd.array(batch.detach().numpy())

  output_tvm: torch.Tensor = torch.tensor(vm["main"](input_tvm).numpy())
  postprocess(output_tvm, weights.meta["categories"])

  # check if torch and tvm output matches
  torch.testing.assert_close(output_torch, output_tvm, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  parser = argparse.ArgumentParser("torchvision classification example")
  parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=torchvision.models.list_models(module=torchvision.models),
  )
  parser.add_argument("--print_torch_graph", action="store_true", default=False)
  args = parser.parse_args()
  main(args)
