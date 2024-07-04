# mobilevit from huggingface

```
$ python mobilevit_test.py 
/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/cuda/__init__.py:619: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
Predicted class: tabby, tabby cat
Traceback (most recent call last):
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/mobilevit/mobilevit_test.py", line 102, in <module>
    main()
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/mobilevit/mobilevit_test.py", line 34, in main
    graph_model: torch.fx.GraphModule = torch.fx.symbolic_trace(trace_model)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 1193, in symbolic_trace
    graph = tracer.trace(root, concrete_args)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 793, in trace
    (self.create_arg(fn(*args)),),
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/mobilevit/mobilevit_test.py", line 18, in forward
    return self.model(pixel_values=pixel_values)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 771, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 495, in call_module
    ret_val = forward(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 764, in forward
    return _orig_module_call(mod, *args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/transformers/models/mobilevit/modeling_mobilevit.py", line 813, in forward
    outputs = self.mobilevit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 771, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 495, in call_module
    ret_val = forward(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 764, in forward
    return _orig_module_call(mod, *args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/transformers/models/mobilevit/modeling_mobilevit.py", line 742, in forward
    encoder_outputs = self.encoder(
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 771, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 495, in call_module
    ret_val = forward(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 764, in forward
    return _orig_module_call(mod, *args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/transformers/models/mobilevit/modeling_mobilevit.py", line 617, in forward
    hidden_states = layer_module(hidden_states)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 771, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 495, in call_module
    ret_val = forward(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 764, in forward
    return _orig_module_call(mod, *args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/transformers/models/mobilevit/modeling_mobilevit.py", line 516, in forward
    patches = self.transformer(patches)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 771, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 495, in call_module
    ret_val = forward(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 764, in forward
    return _orig_module_call(mod, *args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/transformers/models/mobilevit/modeling_mobilevit.py", line 368, in forward
    hidden_states = layer_module(hidden_states)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 771, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 495, in call_module
    ret_val = forward(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 764, in forward
    return _orig_module_call(mod, *args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/transformers/models/mobilevit/modeling_mobilevit.py", line 344, in forward
    attention_output = self.attention(self.layernorm_before(hidden_states))
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 771, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 495, in call_module
    ret_val = forward(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 764, in forward
    return _orig_module_call(mod, *args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/transformers/models/mobilevit/modeling_mobilevit.py", line 301, in forward
    self_outputs = self.attention(hidden_states)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 771, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 495, in call_module
    ret_val = forward(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 764, in forward
    return _orig_module_call(mod, *args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/transformers/models/mobilevit/modeling_mobilevit.py", line 240, in forward
    key_layer = self.transpose_for_scores(self.key(hidden_states))
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/transformers/models/mobilevit/modeling_mobilevit.py", line 234, in transpose_for_scores
    x = x.view(*new_x_shape)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/proxy.py", line 412, in __iter__
    return self.tracer.iter(self)
  File "/home/ubuntu/data/project/torch-fx-to-tvm-relax/.venv/lib/python3.10/site-packages/torch/fx/proxy.py", line 312, in iter
    raise TraceError('Proxy object cannot be iterated. This can be '
torch.fx.proxy.TraceError: Proxy object cannot be iterated. This can be attempted when the Proxy is used in a loop or as a *args or **kwargs function argument. See the torch.fx docs on pytorch.org for a more detailed explanation of what types of control flow can be traced, and check out the Proxy docstring for help troubleshooting Proxy iteration errors
```
