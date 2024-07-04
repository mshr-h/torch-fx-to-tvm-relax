import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_fx
import tvm.relax.frontend.torch.fx_translator
import torch
import torch.fx
import torchvision
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from PIL import Image


class TraceWrapper(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, pixel_values):
    return self.model(pixel_values=pixel_values)


def main():
  image: Image = Image.open("000000039769.jpg")
  model_name: str = "apple/mobilevit-small"
  feature_extractor = MobileViTImageProcessor.from_pretrained(model_name)
  model_pth = MobileViTForImageClassification.from_pretrained(model_name)

  inputs = feature_extractor(images=image, return_tensors="pt")
  outputs = model_pth(**inputs)
  predicted_class_idx = outputs.logits.argmax(-1).item()
  print("Predicted class:", model_pth.config.id2label[predicted_class_idx])

  # Trace
  trace_model: torch.nn.Module = TraceWrapper(model_pth)
  graph_model: torch.fx.GraphModule = torch.fx.symbolic_trace(trace_model)

  return
  model_name: str = "efficientnet_b0"

  weights: torchvision.models.WeightsEnum = torchvision.models.get_model_weights(
    model_name
  ).DEFAULT
  model_pth: torch.nn.Module = torchvision.models.get_model(
    model_name, weights=weights
  ).eval()

  # Trace
  graph_model: torch.fx.GraphModule = torch.fx.symbolic_trace(model_pth)
  # graph_model.print_readable()

  # Convert FX Graph to Relax program
  def convert_stochastic_depth(
    node: torch.fx.node.Node,
    fx_importer: tvm.relax.frontend.torch.fx_translator.TorchFXImporter,
  ):
    args = fx_importer.retrieve_args(node)
    return fx_importer.block_builder.emit(args[0])

  with torch.no_grad():
    inp: torch.Tensor = torch.rand(1, 3, 224, 224)
    mod = from_fx(
      graph_model,
      [(inp.shape, "float32")],
      custom_convert_map={"stochastic_depth": convert_stochastic_depth},
    )
  # mod.show()

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
