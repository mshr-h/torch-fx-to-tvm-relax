from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import requests


def main():
  import torch
  url = "http://images.cocodataset.org/val2017/000000039769.jpg"
  image = Image.open(requests.get(url, stream=True).raw)

  image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
  model_pth = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

  inputs = image_processor(images=image, return_tensors="pt")
  outputs = model_pth(**inputs)

  print(type(outputs))

  # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
  target_sizes = torch.tensor([image.size[::-1]])
  results = image_processor.post_process_object_detection(
    outputs, threshold=0.9, target_sizes=target_sizes
  )[0]

  for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
      f"Detected {model_pth.config.id2label[label.item()]} with confidence "
      f"{round(score.item(), 3)} at location {box}"
    )

  class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
      super().__init__()
      self.model = model

    def forward(self, pixel_values):
      outputs = self.model(pixel_values=pixel_values)
      return outputs["logits"], outputs["pred_boxes"], outputs["last_hidden_state"]

  trace_model = TraceWrapper(model_pth)
  script_model = torch.jit.trace(trace_model, inputs["pixel_values"])

  import tvm
  from tvm import relay
  from tvm.contrib.graph_executor import GraphModule

  target = tvm.target.Target("llvm -mcpu=skylake-avx512")
  mod, params = relay.frontend.from_pytorch(
    script_model, [("pixel_values", inputs["pixel_values"].shape)]
  )

  with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

  gmod = GraphModule(lib["default"](tvm.cpu()))
  gmod.set_input("pixel_values", tvm.nd.from_dlpack(inputs["pixel_values"]))
  gmod.run()

  from transformers.models.yolos.modeling_yolos import YolosObjectDetectionOutput

  outputs = YolosObjectDetectionOutput(
    logits=torch.from_dlpack(gmod.get_output(0).to_dlpack()),
    pred_boxes=torch.from_dlpack(gmod.get_output(1).to_dlpack()),
    last_hidden_state=torch.from_dlpack(gmod.get_output(2).to_dlpack()),
  )

  # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
  target_sizes = torch.tensor([image.size[::-1]])
  results = image_processor.post_process_object_detection(
    outputs, threshold=0.9, target_sizes=target_sizes
  )[0]

  for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
      f"Detected {model_pth.config.id2label[label.item()]} with confidence "
      f"{round(score.item(), 3)} at location {box}"
    )

  import torch.fx
  from tvm import relay

  graph_model = torch.fx.symbolic_trace(model_pth)


if __name__ == "__main__":
  main()
