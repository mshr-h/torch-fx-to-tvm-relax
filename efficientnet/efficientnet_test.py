import tvm
from tvm import relax, relay
from tvm.runtime.vm import VirtualMachine
from tvm.relax.frontend.torch import from_fx
import tvm.relax.frontend.torch.fx_translator
import torch
import torch.fx
import torchvision


def run_relay(model_pth, inp):
  # TVM Relay
  scipted_module = torch.jit.trace(model_pth, example_inputs=(inp))
  mod, params = relay.frontend.from_pytorch(
    scipted_module, input_infos=[("inp0", inp.shape)]
  )

  target = tvm.target.Target("llvm")
  with tvm.transform.PassContext(opt_level=3):
    vm_exec = relay.vm.compile(mod, target=target, params=params)
  ctx = tvm.device(target, 0)
  vm = VirtualMachine(vm_exec, ctx)
  vm.set_input("main", **{"inp0": inp.detach().numpy()})
  return torch.tensor(vm.run().numpy())


def run_relax(model_pth, inp):
  # TVM Relax
  def convert_stochastic_depth(
    node: torch.fx.node.Node,
    fx_importer: tvm.relax.frontend.torch.fx_translator.TorchFXImporter,
  ):
    args = fx_importer.retrieve_args(node)
    return fx_importer.block_builder.emit(args[0])

  with torch.no_grad():
    graph_model: torch.fx.GraphModule = torch.fx.symbolic_trace(model_pth)
    mod = from_fx(
      graph_model,
      [(inp.shape, "float32")],
      custom_convert_map={"stochastic_depth": convert_stochastic_depth},
    )

  target = tvm.target.Target("llvm", host="llvm")
  mod = relax.transform.DecomposeOpsForInference()(mod)
  mod = relax.transform.LegalizeOps()(mod)
  ex = relax.build(mod, target)
  vm = relax.VirtualMachine(ex, tvm.cpu())
  return torch.tensor(vm["main"](tvm.nd.array(inp.detach().numpy())).numpy())


def main():
  model_name = "efficientnet_b0"
  inp = torch.rand(8, 3, 224, 224)

  weights = torchvision.models.get_model_weights(model_name).DEFAULT
  model_pth = torchvision.models.get_model(model_name, weights=weights).eval()

  # PyTorch
  output_pth = model_pth(inp)

  # TVM Relay
  output_relay = run_relay(model_pth, inp)
  torch.testing.assert_close(output_pth, output_relay, rtol=1e-4, atol=1e-4)

  # TVM Relax
  output_relax = run_relax(model_pth, inp)
  torch.testing.assert_close(output_pth, output_relax, rtol=1e-4, atol=1e-4)

  torch.testing.assert_close(output_relay, output_relax, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  main()
