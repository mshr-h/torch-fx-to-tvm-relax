import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_fx
import torch
import torch.fx
import torchvision


def main():
  model_name: str = "mobilenet_v3_small"  # mobilenet_v3_small or mobilenet_v3_large

  weights: torchvision.models.WeightsEnum = torchvision.models.get_model_weights(
    model_name
  ).DEFAULT
  model_pth: torch.nn.Module = torchvision.models.get_model(
    model_name, weights=weights
  ).eval()

  # Convert FX Graph to Relax program
  graph_model: torch.fx.GraphModule = torch.fx.symbolic_trace(model_pth)
  with torch.no_grad():
    inp: torch.Tensor = torch.rand(1, 3, 224, 224)
    mod: tvm.IRModule = from_fx(graph_model, [(inp.shape, "float32")])

  # Construct VirtualMachine
  target: tvm.target.Target = tvm.target.Target("llvm", host="llvm")
  mod: tvm.IRModule = relax.transform.DecomposeOpsForInference()(mod)
  mod: tvm.IRModule = relax.transform.LegalizeOps()(mod)
  ex: relax.Executable = relax.build(mod, target)
  vm: relax.VirtualMachine = relax.VirtualMachine(ex, tvm.cpu())

  ## Inference with image input
  preprocess = weights.transforms()
  img: torch.Tensor = torchvision.io.read_image("bus.jpg")
  batch: torch.Tensor = preprocess(img).unsqueeze(0)

  def print_prediction(output: torch.Tensor, categories) -> None:
    prediction: torch.Tensor = output.squeeze(0).softmax(0)
    class_id: int = prediction.argmax().item()
    score: float = prediction[class_id].item()
    class_name: str = categories[class_id]
    print(f"  {class_name}: {100*score: .1f}%")

  # Run PyTorch inference
  output_pth: torch.Tensor = model_pth(batch)
  print("PyTorch")
  print_prediction(output_pth, weights.meta["categories"])

  # Run TVM Relax inference
  output_tvm: torch.Tensor = torch.from_numpy(
    vm["main"](tvm.nd.array(batch.detach().numpy())).numpy()
  )
  print("TVM")
  print_prediction(output_tvm, weights.meta["categories"])

  torch.testing.assert_close(output_pth, output_tvm, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  main()
