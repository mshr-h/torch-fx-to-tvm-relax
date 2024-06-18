# EfficientNet from Torchvision

```sh
$ python efficientnet_test.py
PyTorch
  minibus:  52.6%
TVM
  minibus:  52.6%
```

## FX Graph

```python
class EfficientNet(torch.nn.Module):
    torch.fx._symbolic_trace.wrap("torchvision_ops_stochastic_depth_stochastic_depth")
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # No stacktrace found for following nodes
        features_0_0 = getattr(getattr(self.features, "0"), "0")(x);  x = None
        features_0_1 = getattr(getattr(self.features, "0"), "1")(features_0_0);  features_0_0 = None
        features_0_2 = getattr(getattr(self.features, "0"), "2")(features_0_1);  features_0_1 = None
        features_1_0_block_0_0 = getattr(getattr(getattr(getattr(self.features, "1"), "0").block, "0"), "0")(features_0_2);  features_0_2 = None
        features_1_0_block_0_1 = getattr(getattr(getattr(getattr(self.features, "1"), "0").block, "0"), "1")(features_1_0_block_0_0);  features_1_0_block_0_0 = None
        features_1_0_block_0_2 = getattr(getattr(getattr(getattr(self.features, "1"), "0").block, "0"), "2")(features_1_0_block_0_1);  features_1_0_block_0_1 = None
        features_1_0_block_1_avgpool = getattr(getattr(getattr(self.features, "1"), "0").block, "1").avgpool(features_1_0_block_0_2)
        features_1_0_block_1_fc1 = getattr(getattr(getattr(self.features, "1"), "0").block, "1").fc1(features_1_0_block_1_avgpool);  features_1_0_block_1_avgpool = None
        features_1_0_block_1_activation = getattr(getattr(getattr(self.features, "1"), "0").block, "1").activation(features_1_0_block_1_fc1);  features_1_0_block_1_fc1 = None
        features_1_0_block_1_fc2 = getattr(getattr(getattr(self.features, "1"), "0").block, "1").fc2(features_1_0_block_1_activation);  features_1_0_block_1_activation = None
        features_1_0_block_1_scale_activation = getattr(getattr(getattr(self.features, "1"), "0").block, "1").scale_activation(features_1_0_block_1_fc2);  features_1_0_block_1_fc2 = None
        mul = features_1_0_block_1_scale_activation * features_1_0_block_0_2;  features_1_0_block_1_scale_activation = features_1_0_block_0_2 = None
        features_1_0_block_2_0 = getattr(getattr(getattr(getattr(self.features, "1"), "0").block, "2"), "0")(mul);  mul = None
        features_1_0_block_2_1 = getattr(getattr(getattr(getattr(self.features, "1"), "0").block, "2"), "1")(features_1_0_block_2_0);  features_1_0_block_2_0 = None
        features_2_0_block_0_0 = getattr(getattr(getattr(getattr(self.features, "2"), "0").block, "0"), "0")(features_1_0_block_2_1);  features_1_0_block_2_1 = None
        features_2_0_block_0_1 = getattr(getattr(getattr(getattr(self.features, "2"), "0").block, "0"), "1")(features_2_0_block_0_0);  features_2_0_block_0_0 = None
        features_2_0_block_0_2 = getattr(getattr(getattr(getattr(self.features, "2"), "0").block, "0"), "2")(features_2_0_block_0_1);  features_2_0_block_0_1 = None
        features_2_0_block_1_0 = getattr(getattr(getattr(getattr(self.features, "2"), "0").block, "1"), "0")(features_2_0_block_0_2);  features_2_0_block_0_2 = None
        features_2_0_block_1_1 = getattr(getattr(getattr(getattr(self.features, "2"), "0").block, "1"), "1")(features_2_0_block_1_0);  features_2_0_block_1_0 = None
        features_2_0_block_1_2 = getattr(getattr(getattr(getattr(self.features, "2"), "0").block, "1"), "2")(features_2_0_block_1_1);  features_2_0_block_1_1 = None
        features_2_0_block_2_avgpool = getattr(getattr(getattr(self.features, "2"), "0").block, "2").avgpool(features_2_0_block_1_2)
        features_2_0_block_2_fc1 = getattr(getattr(getattr(self.features, "2"), "0").block, "2").fc1(features_2_0_block_2_avgpool);  features_2_0_block_2_avgpool = None
        features_2_0_block_2_activation = getattr(getattr(getattr(self.features, "2"), "0").block, "2").activation(features_2_0_block_2_fc1);  features_2_0_block_2_fc1 = None
        features_2_0_block_2_fc2 = getattr(getattr(getattr(self.features, "2"), "0").block, "2").fc2(features_2_0_block_2_activation);  features_2_0_block_2_activation = None
        features_2_0_block_2_scale_activation = getattr(getattr(getattr(self.features, "2"), "0").block, "2").scale_activation(features_2_0_block_2_fc2);  features_2_0_block_2_fc2 = None
        mul_1 = features_2_0_block_2_scale_activation * features_2_0_block_1_2;  features_2_0_block_2_scale_activation = features_2_0_block_1_2 = None
        features_2_0_block_3_0 = getattr(getattr(getattr(getattr(self.features, "2"), "0").block, "3"), "0")(mul_1);  mul_1 = None
        features_2_0_block_3_1 = getattr(getattr(getattr(getattr(self.features, "2"), "0").block, "3"), "1")(features_2_0_block_3_0);  features_2_0_block_3_0 = None
        features_2_1_block_0_0 = getattr(getattr(getattr(getattr(self.features, "2"), "1").block, "0"), "0")(features_2_0_block_3_1)
        features_2_1_block_0_1 = getattr(getattr(getattr(getattr(self.features, "2"), "1").block, "0"), "1")(features_2_1_block_0_0);  features_2_1_block_0_0 = None
        features_2_1_block_0_2 = getattr(getattr(getattr(getattr(self.features, "2"), "1").block, "0"), "2")(features_2_1_block_0_1);  features_2_1_block_0_1 = None
        features_2_1_block_1_0 = getattr(getattr(getattr(getattr(self.features, "2"), "1").block, "1"), "0")(features_2_1_block_0_2);  features_2_1_block_0_2 = None
        features_2_1_block_1_1 = getattr(getattr(getattr(getattr(self.features, "2"), "1").block, "1"), "1")(features_2_1_block_1_0);  features_2_1_block_1_0 = None
        features_2_1_block_1_2 = getattr(getattr(getattr(getattr(self.features, "2"), "1").block, "1"), "2")(features_2_1_block_1_1);  features_2_1_block_1_1 = None
        features_2_1_block_2_avgpool = getattr(getattr(getattr(self.features, "2"), "1").block, "2").avgpool(features_2_1_block_1_2)
        features_2_1_block_2_fc1 = getattr(getattr(getattr(self.features, "2"), "1").block, "2").fc1(features_2_1_block_2_avgpool);  features_2_1_block_2_avgpool = None
        features_2_1_block_2_activation = getattr(getattr(getattr(self.features, "2"), "1").block, "2").activation(features_2_1_block_2_fc1);  features_2_1_block_2_fc1 = None
        features_2_1_block_2_fc2 = getattr(getattr(getattr(self.features, "2"), "1").block, "2").fc2(features_2_1_block_2_activation);  features_2_1_block_2_activation = None
        features_2_1_block_2_scale_activation = getattr(getattr(getattr(self.features, "2"), "1").block, "2").scale_activation(features_2_1_block_2_fc2);  features_2_1_block_2_fc2 = None
        mul_2 = features_2_1_block_2_scale_activation * features_2_1_block_1_2;  features_2_1_block_2_scale_activation = features_2_1_block_1_2 = None
        features_2_1_block_3_0 = getattr(getattr(getattr(getattr(self.features, "2"), "1").block, "3"), "0")(mul_2);  mul_2 = None
        features_2_1_block_3_1 = getattr(getattr(getattr(getattr(self.features, "2"), "1").block, "3"), "1")(features_2_1_block_3_0);  features_2_1_block_3_0 = None
        stochastic_depth = torchvision_ops_stochastic_depth_stochastic_depth(features_2_1_block_3_1, 0.025, 'row', False);  features_2_1_block_3_1 = None
        add = stochastic_depth + features_2_0_block_3_1;  stochastic_depth = features_2_0_block_3_1 = None
        features_3_0_block_0_0 = getattr(getattr(getattr(getattr(self.features, "3"), "0").block, "0"), "0")(add);  add = None
        features_3_0_block_0_1 = getattr(getattr(getattr(getattr(self.features, "3"), "0").block, "0"), "1")(features_3_0_block_0_0);  features_3_0_block_0_0 = None
        features_3_0_block_0_2 = getattr(getattr(getattr(getattr(self.features, "3"), "0").block, "0"), "2")(features_3_0_block_0_1);  features_3_0_block_0_1 = None
        features_3_0_block_1_0 = getattr(getattr(getattr(getattr(self.features, "3"), "0").block, "1"), "0")(features_3_0_block_0_2);  features_3_0_block_0_2 = None
        features_3_0_block_1_1 = getattr(getattr(getattr(getattr(self.features, "3"), "0").block, "1"), "1")(features_3_0_block_1_0);  features_3_0_block_1_0 = None
        features_3_0_block_1_2 = getattr(getattr(getattr(getattr(self.features, "3"), "0").block, "1"), "2")(features_3_0_block_1_1);  features_3_0_block_1_1 = None
        features_3_0_block_2_avgpool = getattr(getattr(getattr(self.features, "3"), "0").block, "2").avgpool(features_3_0_block_1_2)
        features_3_0_block_2_fc1 = getattr(getattr(getattr(self.features, "3"), "0").block, "2").fc1(features_3_0_block_2_avgpool);  features_3_0_block_2_avgpool = None
        features_3_0_block_2_activation = getattr(getattr(getattr(self.features, "3"), "0").block, "2").activation(features_3_0_block_2_fc1);  features_3_0_block_2_fc1 = None
        features_3_0_block_2_fc2 = getattr(getattr(getattr(self.features, "3"), "0").block, "2").fc2(features_3_0_block_2_activation);  features_3_0_block_2_activation = None
        features_3_0_block_2_scale_activation = getattr(getattr(getattr(self.features, "3"), "0").block, "2").scale_activation(features_3_0_block_2_fc2);  features_3_0_block_2_fc2 = None
        mul_3 = features_3_0_block_2_scale_activation * features_3_0_block_1_2;  features_3_0_block_2_scale_activation = features_3_0_block_1_2 = None
        features_3_0_block_3_0 = getattr(getattr(getattr(getattr(self.features, "3"), "0").block, "3"), "0")(mul_3);  mul_3 = None
        features_3_0_block_3_1 = getattr(getattr(getattr(getattr(self.features, "3"), "0").block, "3"), "1")(features_3_0_block_3_0);  features_3_0_block_3_0 = None
        features_3_1_block_0_0 = getattr(getattr(getattr(getattr(self.features, "3"), "1").block, "0"), "0")(features_3_0_block_3_1)
        features_3_1_block_0_1 = getattr(getattr(getattr(getattr(self.features, "3"), "1").block, "0"), "1")(features_3_1_block_0_0);  features_3_1_block_0_0 = None
        features_3_1_block_0_2 = getattr(getattr(getattr(getattr(self.features, "3"), "1").block, "0"), "2")(features_3_1_block_0_1);  features_3_1_block_0_1 = None
        features_3_1_block_1_0 = getattr(getattr(getattr(getattr(self.features, "3"), "1").block, "1"), "0")(features_3_1_block_0_2);  features_3_1_block_0_2 = None
        features_3_1_block_1_1 = getattr(getattr(getattr(getattr(self.features, "3"), "1").block, "1"), "1")(features_3_1_block_1_0);  features_3_1_block_1_0 = None
        features_3_1_block_1_2 = getattr(getattr(getattr(getattr(self.features, "3"), "1").block, "1"), "2")(features_3_1_block_1_1);  features_3_1_block_1_1 = None
        features_3_1_block_2_avgpool = getattr(getattr(getattr(self.features, "3"), "1").block, "2").avgpool(features_3_1_block_1_2)
        features_3_1_block_2_fc1 = getattr(getattr(getattr(self.features, "3"), "1").block, "2").fc1(features_3_1_block_2_avgpool);  features_3_1_block_2_avgpool = None
        features_3_1_block_2_activation = getattr(getattr(getattr(self.features, "3"), "1").block, "2").activation(features_3_1_block_2_fc1);  features_3_1_block_2_fc1 = None
        features_3_1_block_2_fc2 = getattr(getattr(getattr(self.features, "3"), "1").block, "2").fc2(features_3_1_block_2_activation);  features_3_1_block_2_activation = None
        features_3_1_block_2_scale_activation = getattr(getattr(getattr(self.features, "3"), "1").block, "2").scale_activation(features_3_1_block_2_fc2);  features_3_1_block_2_fc2 = None
        mul_4 = features_3_1_block_2_scale_activation * features_3_1_block_1_2;  features_3_1_block_2_scale_activation = features_3_1_block_1_2 = None
        features_3_1_block_3_0 = getattr(getattr(getattr(getattr(self.features, "3"), "1").block, "3"), "0")(mul_4);  mul_4 = None
        features_3_1_block_3_1 = getattr(getattr(getattr(getattr(self.features, "3"), "1").block, "3"), "1")(features_3_1_block_3_0);  features_3_1_block_3_0 = None
        stochastic_depth_1 = torchvision_ops_stochastic_depth_stochastic_depth(features_3_1_block_3_1, 0.05, 'row', False);  features_3_1_block_3_1 = None
        add_1 = stochastic_depth_1 + features_3_0_block_3_1;  stochastic_depth_1 = features_3_0_block_3_1 = None
        features_4_0_block_0_0 = getattr(getattr(getattr(getattr(self.features, "4"), "0").block, "0"), "0")(add_1);  add_1 = None
        features_4_0_block_0_1 = getattr(getattr(getattr(getattr(self.features, "4"), "0").block, "0"), "1")(features_4_0_block_0_0);  features_4_0_block_0_0 = None
        features_4_0_block_0_2 = getattr(getattr(getattr(getattr(self.features, "4"), "0").block, "0"), "2")(features_4_0_block_0_1);  features_4_0_block_0_1 = None
        features_4_0_block_1_0 = getattr(getattr(getattr(getattr(self.features, "4"), "0").block, "1"), "0")(features_4_0_block_0_2);  features_4_0_block_0_2 = None
        features_4_0_block_1_1 = getattr(getattr(getattr(getattr(self.features, "4"), "0").block, "1"), "1")(features_4_0_block_1_0);  features_4_0_block_1_0 = None
        features_4_0_block_1_2 = getattr(getattr(getattr(getattr(self.features, "4"), "0").block, "1"), "2")(features_4_0_block_1_1);  features_4_0_block_1_1 = None
        features_4_0_block_2_avgpool = getattr(getattr(getattr(self.features, "4"), "0").block, "2").avgpool(features_4_0_block_1_2)
        features_4_0_block_2_fc1 = getattr(getattr(getattr(self.features, "4"), "0").block, "2").fc1(features_4_0_block_2_avgpool);  features_4_0_block_2_avgpool = None
        features_4_0_block_2_activation = getattr(getattr(getattr(self.features, "4"), "0").block, "2").activation(features_4_0_block_2_fc1);  features_4_0_block_2_fc1 = None
        features_4_0_block_2_fc2 = getattr(getattr(getattr(self.features, "4"), "0").block, "2").fc2(features_4_0_block_2_activation);  features_4_0_block_2_activation = None
        features_4_0_block_2_scale_activation = getattr(getattr(getattr(self.features, "4"), "0").block, "2").scale_activation(features_4_0_block_2_fc2);  features_4_0_block_2_fc2 = None
        mul_5 = features_4_0_block_2_scale_activation * features_4_0_block_1_2;  features_4_0_block_2_scale_activation = features_4_0_block_1_2 = None
        features_4_0_block_3_0 = getattr(getattr(getattr(getattr(self.features, "4"), "0").block, "3"), "0")(mul_5);  mul_5 = None
        features_4_0_block_3_1 = getattr(getattr(getattr(getattr(self.features, "4"), "0").block, "3"), "1")(features_4_0_block_3_0);  features_4_0_block_3_0 = None
        features_4_1_block_0_0 = getattr(getattr(getattr(getattr(self.features, "4"), "1").block, "0"), "0")(features_4_0_block_3_1)
        features_4_1_block_0_1 = getattr(getattr(getattr(getattr(self.features, "4"), "1").block, "0"), "1")(features_4_1_block_0_0);  features_4_1_block_0_0 = None
        features_4_1_block_0_2 = getattr(getattr(getattr(getattr(self.features, "4"), "1").block, "0"), "2")(features_4_1_block_0_1);  features_4_1_block_0_1 = None
        features_4_1_block_1_0 = getattr(getattr(getattr(getattr(self.features, "4"), "1").block, "1"), "0")(features_4_1_block_0_2);  features_4_1_block_0_2 = None
        features_4_1_block_1_1 = getattr(getattr(getattr(getattr(self.features, "4"), "1").block, "1"), "1")(features_4_1_block_1_0);  features_4_1_block_1_0 = None
        features_4_1_block_1_2 = getattr(getattr(getattr(getattr(self.features, "4"), "1").block, "1"), "2")(features_4_1_block_1_1);  features_4_1_block_1_1 = None
        features_4_1_block_2_avgpool = getattr(getattr(getattr(self.features, "4"), "1").block, "2").avgpool(features_4_1_block_1_2)
        features_4_1_block_2_fc1 = getattr(getattr(getattr(self.features, "4"), "1").block, "2").fc1(features_4_1_block_2_avgpool);  features_4_1_block_2_avgpool = None
        features_4_1_block_2_activation = getattr(getattr(getattr(self.features, "4"), "1").block, "2").activation(features_4_1_block_2_fc1);  features_4_1_block_2_fc1 = None
        features_4_1_block_2_fc2 = getattr(getattr(getattr(self.features, "4"), "1").block, "2").fc2(features_4_1_block_2_activation);  features_4_1_block_2_activation = None
        features_4_1_block_2_scale_activation = getattr(getattr(getattr(self.features, "4"), "1").block, "2").scale_activation(features_4_1_block_2_fc2);  features_4_1_block_2_fc2 = None
        mul_6 = features_4_1_block_2_scale_activation * features_4_1_block_1_2;  features_4_1_block_2_scale_activation = features_4_1_block_1_2 = None
        features_4_1_block_3_0 = getattr(getattr(getattr(getattr(self.features, "4"), "1").block, "3"), "0")(mul_6);  mul_6 = None
        features_4_1_block_3_1 = getattr(getattr(getattr(getattr(self.features, "4"), "1").block, "3"), "1")(features_4_1_block_3_0);  features_4_1_block_3_0 = None
        stochastic_depth_2 = torchvision_ops_stochastic_depth_stochastic_depth(features_4_1_block_3_1, 0.07500000000000001, 'row', False);  features_4_1_block_3_1 = None
        add_2 = stochastic_depth_2 + features_4_0_block_3_1;  stochastic_depth_2 = features_4_0_block_3_1 = None
        features_4_2_block_0_0 = getattr(getattr(getattr(getattr(self.features, "4"), "2").block, "0"), "0")(add_2)
        features_4_2_block_0_1 = getattr(getattr(getattr(getattr(self.features, "4"), "2").block, "0"), "1")(features_4_2_block_0_0);  features_4_2_block_0_0 = None
        features_4_2_block_0_2 = getattr(getattr(getattr(getattr(self.features, "4"), "2").block, "0"), "2")(features_4_2_block_0_1);  features_4_2_block_0_1 = None
        features_4_2_block_1_0 = getattr(getattr(getattr(getattr(self.features, "4"), "2").block, "1"), "0")(features_4_2_block_0_2);  features_4_2_block_0_2 = None
        features_4_2_block_1_1 = getattr(getattr(getattr(getattr(self.features, "4"), "2").block, "1"), "1")(features_4_2_block_1_0);  features_4_2_block_1_0 = None
        features_4_2_block_1_2 = getattr(getattr(getattr(getattr(self.features, "4"), "2").block, "1"), "2")(features_4_2_block_1_1);  features_4_2_block_1_1 = None
        features_4_2_block_2_avgpool = getattr(getattr(getattr(self.features, "4"), "2").block, "2").avgpool(features_4_2_block_1_2)
        features_4_2_block_2_fc1 = getattr(getattr(getattr(self.features, "4"), "2").block, "2").fc1(features_4_2_block_2_avgpool);  features_4_2_block_2_avgpool = None
        features_4_2_block_2_activation = getattr(getattr(getattr(self.features, "4"), "2").block, "2").activation(features_4_2_block_2_fc1);  features_4_2_block_2_fc1 = None
        features_4_2_block_2_fc2 = getattr(getattr(getattr(self.features, "4"), "2").block, "2").fc2(features_4_2_block_2_activation);  features_4_2_block_2_activation = None
        features_4_2_block_2_scale_activation = getattr(getattr(getattr(self.features, "4"), "2").block, "2").scale_activation(features_4_2_block_2_fc2);  features_4_2_block_2_fc2 = None
        mul_7 = features_4_2_block_2_scale_activation * features_4_2_block_1_2;  features_4_2_block_2_scale_activation = features_4_2_block_1_2 = None
        features_4_2_block_3_0 = getattr(getattr(getattr(getattr(self.features, "4"), "2").block, "3"), "0")(mul_7);  mul_7 = None
        features_4_2_block_3_1 = getattr(getattr(getattr(getattr(self.features, "4"), "2").block, "3"), "1")(features_4_2_block_3_0);  features_4_2_block_3_0 = None
        stochastic_depth_3 = torchvision_ops_stochastic_depth_stochastic_depth(features_4_2_block_3_1, 0.08750000000000001, 'row', False);  features_4_2_block_3_1 = None
        add_3 = stochastic_depth_3 + add_2;  stochastic_depth_3 = add_2 = None
        features_5_0_block_0_0 = getattr(getattr(getattr(getattr(self.features, "5"), "0").block, "0"), "0")(add_3);  add_3 = None
        features_5_0_block_0_1 = getattr(getattr(getattr(getattr(self.features, "5"), "0").block, "0"), "1")(features_5_0_block_0_0);  features_5_0_block_0_0 = None
        features_5_0_block_0_2 = getattr(getattr(getattr(getattr(self.features, "5"), "0").block, "0"), "2")(features_5_0_block_0_1);  features_5_0_block_0_1 = None
        features_5_0_block_1_0 = getattr(getattr(getattr(getattr(self.features, "5"), "0").block, "1"), "0")(features_5_0_block_0_2);  features_5_0_block_0_2 = None
        features_5_0_block_1_1 = getattr(getattr(getattr(getattr(self.features, "5"), "0").block, "1"), "1")(features_5_0_block_1_0);  features_5_0_block_1_0 = None
        features_5_0_block_1_2 = getattr(getattr(getattr(getattr(self.features, "5"), "0").block, "1"), "2")(features_5_0_block_1_1);  features_5_0_block_1_1 = None
        features_5_0_block_2_avgpool = getattr(getattr(getattr(self.features, "5"), "0").block, "2").avgpool(features_5_0_block_1_2)
        features_5_0_block_2_fc1 = getattr(getattr(getattr(self.features, "5"), "0").block, "2").fc1(features_5_0_block_2_avgpool);  features_5_0_block_2_avgpool = None
        features_5_0_block_2_activation = getattr(getattr(getattr(self.features, "5"), "0").block, "2").activation(features_5_0_block_2_fc1);  features_5_0_block_2_fc1 = None
        features_5_0_block_2_fc2 = getattr(getattr(getattr(self.features, "5"), "0").block, "2").fc2(features_5_0_block_2_activation);  features_5_0_block_2_activation = None
        features_5_0_block_2_scale_activation = getattr(getattr(getattr(self.features, "5"), "0").block, "2").scale_activation(features_5_0_block_2_fc2);  features_5_0_block_2_fc2 = None
        mul_8 = features_5_0_block_2_scale_activation * features_5_0_block_1_2;  features_5_0_block_2_scale_activation = features_5_0_block_1_2 = None
        features_5_0_block_3_0 = getattr(getattr(getattr(getattr(self.features, "5"), "0").block, "3"), "0")(mul_8);  mul_8 = None
        features_5_0_block_3_1 = getattr(getattr(getattr(getattr(self.features, "5"), "0").block, "3"), "1")(features_5_0_block_3_0);  features_5_0_block_3_0 = None
        features_5_1_block_0_0 = getattr(getattr(getattr(getattr(self.features, "5"), "1").block, "0"), "0")(features_5_0_block_3_1)
        features_5_1_block_0_1 = getattr(getattr(getattr(getattr(self.features, "5"), "1").block, "0"), "1")(features_5_1_block_0_0);  features_5_1_block_0_0 = None
        features_5_1_block_0_2 = getattr(getattr(getattr(getattr(self.features, "5"), "1").block, "0"), "2")(features_5_1_block_0_1);  features_5_1_block_0_1 = None
        features_5_1_block_1_0 = getattr(getattr(getattr(getattr(self.features, "5"), "1").block, "1"), "0")(features_5_1_block_0_2);  features_5_1_block_0_2 = None
        features_5_1_block_1_1 = getattr(getattr(getattr(getattr(self.features, "5"), "1").block, "1"), "1")(features_5_1_block_1_0);  features_5_1_block_1_0 = None
        features_5_1_block_1_2 = getattr(getattr(getattr(getattr(self.features, "5"), "1").block, "1"), "2")(features_5_1_block_1_1);  features_5_1_block_1_1 = None
        features_5_1_block_2_avgpool = getattr(getattr(getattr(self.features, "5"), "1").block, "2").avgpool(features_5_1_block_1_2)
        features_5_1_block_2_fc1 = getattr(getattr(getattr(self.features, "5"), "1").block, "2").fc1(features_5_1_block_2_avgpool);  features_5_1_block_2_avgpool = None
        features_5_1_block_2_activation = getattr(getattr(getattr(self.features, "5"), "1").block, "2").activation(features_5_1_block_2_fc1);  features_5_1_block_2_fc1 = None
        features_5_1_block_2_fc2 = getattr(getattr(getattr(self.features, "5"), "1").block, "2").fc2(features_5_1_block_2_activation);  features_5_1_block_2_activation = None
        features_5_1_block_2_scale_activation = getattr(getattr(getattr(self.features, "5"), "1").block, "2").scale_activation(features_5_1_block_2_fc2);  features_5_1_block_2_fc2 = None
        mul_9 = features_5_1_block_2_scale_activation * features_5_1_block_1_2;  features_5_1_block_2_scale_activation = features_5_1_block_1_2 = None
        features_5_1_block_3_0 = getattr(getattr(getattr(getattr(self.features, "5"), "1").block, "3"), "0")(mul_9);  mul_9 = None
        features_5_1_block_3_1 = getattr(getattr(getattr(getattr(self.features, "5"), "1").block, "3"), "1")(features_5_1_block_3_0);  features_5_1_block_3_0 = None
        stochastic_depth_4 = torchvision_ops_stochastic_depth_stochastic_depth(features_5_1_block_3_1, 0.1125, 'row', False);  features_5_1_block_3_1 = None
        add_4 = stochastic_depth_4 + features_5_0_block_3_1;  stochastic_depth_4 = features_5_0_block_3_1 = None
        features_5_2_block_0_0 = getattr(getattr(getattr(getattr(self.features, "5"), "2").block, "0"), "0")(add_4)
        features_5_2_block_0_1 = getattr(getattr(getattr(getattr(self.features, "5"), "2").block, "0"), "1")(features_5_2_block_0_0);  features_5_2_block_0_0 = None
        features_5_2_block_0_2 = getattr(getattr(getattr(getattr(self.features, "5"), "2").block, "0"), "2")(features_5_2_block_0_1);  features_5_2_block_0_1 = None
        features_5_2_block_1_0 = getattr(getattr(getattr(getattr(self.features, "5"), "2").block, "1"), "0")(features_5_2_block_0_2);  features_5_2_block_0_2 = None
        features_5_2_block_1_1 = getattr(getattr(getattr(getattr(self.features, "5"), "2").block, "1"), "1")(features_5_2_block_1_0);  features_5_2_block_1_0 = None
        features_5_2_block_1_2 = getattr(getattr(getattr(getattr(self.features, "5"), "2").block, "1"), "2")(features_5_2_block_1_1);  features_5_2_block_1_1 = None
        features_5_2_block_2_avgpool = getattr(getattr(getattr(self.features, "5"), "2").block, "2").avgpool(features_5_2_block_1_2)
        features_5_2_block_2_fc1 = getattr(getattr(getattr(self.features, "5"), "2").block, "2").fc1(features_5_2_block_2_avgpool);  features_5_2_block_2_avgpool = None
        features_5_2_block_2_activation = getattr(getattr(getattr(self.features, "5"), "2").block, "2").activation(features_5_2_block_2_fc1);  features_5_2_block_2_fc1 = None
        features_5_2_block_2_fc2 = getattr(getattr(getattr(self.features, "5"), "2").block, "2").fc2(features_5_2_block_2_activation);  features_5_2_block_2_activation = None
        features_5_2_block_2_scale_activation = getattr(getattr(getattr(self.features, "5"), "2").block, "2").scale_activation(features_5_2_block_2_fc2);  features_5_2_block_2_fc2 = None
        mul_10 = features_5_2_block_2_scale_activation * features_5_2_block_1_2;  features_5_2_block_2_scale_activation = features_5_2_block_1_2 = None
        features_5_2_block_3_0 = getattr(getattr(getattr(getattr(self.features, "5"), "2").block, "3"), "0")(mul_10);  mul_10 = None
        features_5_2_block_3_1 = getattr(getattr(getattr(getattr(self.features, "5"), "2").block, "3"), "1")(features_5_2_block_3_0);  features_5_2_block_3_0 = None
        stochastic_depth_5 = torchvision_ops_stochastic_depth_stochastic_depth(features_5_2_block_3_1, 0.125, 'row', False);  features_5_2_block_3_1 = None
        add_5 = stochastic_depth_5 + add_4;  stochastic_depth_5 = add_4 = None
        features_6_0_block_0_0 = getattr(getattr(getattr(getattr(self.features, "6"), "0").block, "0"), "0")(add_5);  add_5 = None
        features_6_0_block_0_1 = getattr(getattr(getattr(getattr(self.features, "6"), "0").block, "0"), "1")(features_6_0_block_0_0);  features_6_0_block_0_0 = None
        features_6_0_block_0_2 = getattr(getattr(getattr(getattr(self.features, "6"), "0").block, "0"), "2")(features_6_0_block_0_1);  features_6_0_block_0_1 = None
        features_6_0_block_1_0 = getattr(getattr(getattr(getattr(self.features, "6"), "0").block, "1"), "0")(features_6_0_block_0_2);  features_6_0_block_0_2 = None
        features_6_0_block_1_1 = getattr(getattr(getattr(getattr(self.features, "6"), "0").block, "1"), "1")(features_6_0_block_1_0);  features_6_0_block_1_0 = None
        features_6_0_block_1_2 = getattr(getattr(getattr(getattr(self.features, "6"), "0").block, "1"), "2")(features_6_0_block_1_1);  features_6_0_block_1_1 = None
        features_6_0_block_2_avgpool = getattr(getattr(getattr(self.features, "6"), "0").block, "2").avgpool(features_6_0_block_1_2)
        features_6_0_block_2_fc1 = getattr(getattr(getattr(self.features, "6"), "0").block, "2").fc1(features_6_0_block_2_avgpool);  features_6_0_block_2_avgpool = None
        features_6_0_block_2_activation = getattr(getattr(getattr(self.features, "6"), "0").block, "2").activation(features_6_0_block_2_fc1);  features_6_0_block_2_fc1 = None
        features_6_0_block_2_fc2 = getattr(getattr(getattr(self.features, "6"), "0").block, "2").fc2(features_6_0_block_2_activation);  features_6_0_block_2_activation = None
        features_6_0_block_2_scale_activation = getattr(getattr(getattr(self.features, "6"), "0").block, "2").scale_activation(features_6_0_block_2_fc2);  features_6_0_block_2_fc2 = None
        mul_11 = features_6_0_block_2_scale_activation * features_6_0_block_1_2;  features_6_0_block_2_scale_activation = features_6_0_block_1_2 = None
        features_6_0_block_3_0 = getattr(getattr(getattr(getattr(self.features, "6"), "0").block, "3"), "0")(mul_11);  mul_11 = None
        features_6_0_block_3_1 = getattr(getattr(getattr(getattr(self.features, "6"), "0").block, "3"), "1")(features_6_0_block_3_0);  features_6_0_block_3_0 = None
        features_6_1_block_0_0 = getattr(getattr(getattr(getattr(self.features, "6"), "1").block, "0"), "0")(features_6_0_block_3_1)
        features_6_1_block_0_1 = getattr(getattr(getattr(getattr(self.features, "6"), "1").block, "0"), "1")(features_6_1_block_0_0);  features_6_1_block_0_0 = None
        features_6_1_block_0_2 = getattr(getattr(getattr(getattr(self.features, "6"), "1").block, "0"), "2")(features_6_1_block_0_1);  features_6_1_block_0_1 = None
        features_6_1_block_1_0 = getattr(getattr(getattr(getattr(self.features, "6"), "1").block, "1"), "0")(features_6_1_block_0_2);  features_6_1_block_0_2 = None
        features_6_1_block_1_1 = getattr(getattr(getattr(getattr(self.features, "6"), "1").block, "1"), "1")(features_6_1_block_1_0);  features_6_1_block_1_0 = None
        features_6_1_block_1_2 = getattr(getattr(getattr(getattr(self.features, "6"), "1").block, "1"), "2")(features_6_1_block_1_1);  features_6_1_block_1_1 = None
        features_6_1_block_2_avgpool = getattr(getattr(getattr(self.features, "6"), "1").block, "2").avgpool(features_6_1_block_1_2)
        features_6_1_block_2_fc1 = getattr(getattr(getattr(self.features, "6"), "1").block, "2").fc1(features_6_1_block_2_avgpool);  features_6_1_block_2_avgpool = None
        features_6_1_block_2_activation = getattr(getattr(getattr(self.features, "6"), "1").block, "2").activation(features_6_1_block_2_fc1);  features_6_1_block_2_fc1 = None
        features_6_1_block_2_fc2 = getattr(getattr(getattr(self.features, "6"), "1").block, "2").fc2(features_6_1_block_2_activation);  features_6_1_block_2_activation = None
        features_6_1_block_2_scale_activation = getattr(getattr(getattr(self.features, "6"), "1").block, "2").scale_activation(features_6_1_block_2_fc2);  features_6_1_block_2_fc2 = None
        mul_12 = features_6_1_block_2_scale_activation * features_6_1_block_1_2;  features_6_1_block_2_scale_activation = features_6_1_block_1_2 = None
        features_6_1_block_3_0 = getattr(getattr(getattr(getattr(self.features, "6"), "1").block, "3"), "0")(mul_12);  mul_12 = None
        features_6_1_block_3_1 = getattr(getattr(getattr(getattr(self.features, "6"), "1").block, "3"), "1")(features_6_1_block_3_0);  features_6_1_block_3_0 = None
        stochastic_depth_6 = torchvision_ops_stochastic_depth_stochastic_depth(features_6_1_block_3_1, 0.15000000000000002, 'row', False);  features_6_1_block_3_1 = None
        add_6 = stochastic_depth_6 + features_6_0_block_3_1;  stochastic_depth_6 = features_6_0_block_3_1 = None
        features_6_2_block_0_0 = getattr(getattr(getattr(getattr(self.features, "6"), "2").block, "0"), "0")(add_6)
        features_6_2_block_0_1 = getattr(getattr(getattr(getattr(self.features, "6"), "2").block, "0"), "1")(features_6_2_block_0_0);  features_6_2_block_0_0 = None
        features_6_2_block_0_2 = getattr(getattr(getattr(getattr(self.features, "6"), "2").block, "0"), "2")(features_6_2_block_0_1);  features_6_2_block_0_1 = None
        features_6_2_block_1_0 = getattr(getattr(getattr(getattr(self.features, "6"), "2").block, "1"), "0")(features_6_2_block_0_2);  features_6_2_block_0_2 = None
        features_6_2_block_1_1 = getattr(getattr(getattr(getattr(self.features, "6"), "2").block, "1"), "1")(features_6_2_block_1_0);  features_6_2_block_1_0 = None
        features_6_2_block_1_2 = getattr(getattr(getattr(getattr(self.features, "6"), "2").block, "1"), "2")(features_6_2_block_1_1);  features_6_2_block_1_1 = None
        features_6_2_block_2_avgpool = getattr(getattr(getattr(self.features, "6"), "2").block, "2").avgpool(features_6_2_block_1_2)
        features_6_2_block_2_fc1 = getattr(getattr(getattr(self.features, "6"), "2").block, "2").fc1(features_6_2_block_2_avgpool);  features_6_2_block_2_avgpool = None
        features_6_2_block_2_activation = getattr(getattr(getattr(self.features, "6"), "2").block, "2").activation(features_6_2_block_2_fc1);  features_6_2_block_2_fc1 = None
        features_6_2_block_2_fc2 = getattr(getattr(getattr(self.features, "6"), "2").block, "2").fc2(features_6_2_block_2_activation);  features_6_2_block_2_activation = None
        features_6_2_block_2_scale_activation = getattr(getattr(getattr(self.features, "6"), "2").block, "2").scale_activation(features_6_2_block_2_fc2);  features_6_2_block_2_fc2 = None
        mul_13 = features_6_2_block_2_scale_activation * features_6_2_block_1_2;  features_6_2_block_2_scale_activation = features_6_2_block_1_2 = None
        features_6_2_block_3_0 = getattr(getattr(getattr(getattr(self.features, "6"), "2").block, "3"), "0")(mul_13);  mul_13 = None
        features_6_2_block_3_1 = getattr(getattr(getattr(getattr(self.features, "6"), "2").block, "3"), "1")(features_6_2_block_3_0);  features_6_2_block_3_0 = None
        stochastic_depth_7 = torchvision_ops_stochastic_depth_stochastic_depth(features_6_2_block_3_1, 0.1625, 'row', False);  features_6_2_block_3_1 = None
        add_7 = stochastic_depth_7 + add_6;  stochastic_depth_7 = add_6 = None
        features_6_3_block_0_0 = getattr(getattr(getattr(getattr(self.features, "6"), "3").block, "0"), "0")(add_7)
        features_6_3_block_0_1 = getattr(getattr(getattr(getattr(self.features, "6"), "3").block, "0"), "1")(features_6_3_block_0_0);  features_6_3_block_0_0 = None
        features_6_3_block_0_2 = getattr(getattr(getattr(getattr(self.features, "6"), "3").block, "0"), "2")(features_6_3_block_0_1);  features_6_3_block_0_1 = None
        features_6_3_block_1_0 = getattr(getattr(getattr(getattr(self.features, "6"), "3").block, "1"), "0")(features_6_3_block_0_2);  features_6_3_block_0_2 = None
        features_6_3_block_1_1 = getattr(getattr(getattr(getattr(self.features, "6"), "3").block, "1"), "1")(features_6_3_block_1_0);  features_6_3_block_1_0 = None
        features_6_3_block_1_2 = getattr(getattr(getattr(getattr(self.features, "6"), "3").block, "1"), "2")(features_6_3_block_1_1);  features_6_3_block_1_1 = None
        features_6_3_block_2_avgpool = getattr(getattr(getattr(self.features, "6"), "3").block, "2").avgpool(features_6_3_block_1_2)
        features_6_3_block_2_fc1 = getattr(getattr(getattr(self.features, "6"), "3").block, "2").fc1(features_6_3_block_2_avgpool);  features_6_3_block_2_avgpool = None
        features_6_3_block_2_activation = getattr(getattr(getattr(self.features, "6"), "3").block, "2").activation(features_6_3_block_2_fc1);  features_6_3_block_2_fc1 = None
        features_6_3_block_2_fc2 = getattr(getattr(getattr(self.features, "6"), "3").block, "2").fc2(features_6_3_block_2_activation);  features_6_3_block_2_activation = None
        features_6_3_block_2_scale_activation = getattr(getattr(getattr(self.features, "6"), "3").block, "2").scale_activation(features_6_3_block_2_fc2);  features_6_3_block_2_fc2 = None
        mul_14 = features_6_3_block_2_scale_activation * features_6_3_block_1_2;  features_6_3_block_2_scale_activation = features_6_3_block_1_2 = None
        features_6_3_block_3_0 = getattr(getattr(getattr(getattr(self.features, "6"), "3").block, "3"), "0")(mul_14);  mul_14 = None
        features_6_3_block_3_1 = getattr(getattr(getattr(getattr(self.features, "6"), "3").block, "3"), "1")(features_6_3_block_3_0);  features_6_3_block_3_0 = None
        stochastic_depth_8 = torchvision_ops_stochastic_depth_stochastic_depth(features_6_3_block_3_1, 0.17500000000000002, 'row', False);  features_6_3_block_3_1 = None
        add_8 = stochastic_depth_8 + add_7;  stochastic_depth_8 = add_7 = None
        features_7_0_block_0_0 = getattr(getattr(getattr(getattr(self.features, "7"), "0").block, "0"), "0")(add_8);  add_8 = None
        features_7_0_block_0_1 = getattr(getattr(getattr(getattr(self.features, "7"), "0").block, "0"), "1")(features_7_0_block_0_0);  features_7_0_block_0_0 = None
        features_7_0_block_0_2 = getattr(getattr(getattr(getattr(self.features, "7"), "0").block, "0"), "2")(features_7_0_block_0_1);  features_7_0_block_0_1 = None
        features_7_0_block_1_0 = getattr(getattr(getattr(getattr(self.features, "7"), "0").block, "1"), "0")(features_7_0_block_0_2);  features_7_0_block_0_2 = None
        features_7_0_block_1_1 = getattr(getattr(getattr(getattr(self.features, "7"), "0").block, "1"), "1")(features_7_0_block_1_0);  features_7_0_block_1_0 = None
        features_7_0_block_1_2 = getattr(getattr(getattr(getattr(self.features, "7"), "0").block, "1"), "2")(features_7_0_block_1_1);  features_7_0_block_1_1 = None
        features_7_0_block_2_avgpool = getattr(getattr(getattr(self.features, "7"), "0").block, "2").avgpool(features_7_0_block_1_2)
        features_7_0_block_2_fc1 = getattr(getattr(getattr(self.features, "7"), "0").block, "2").fc1(features_7_0_block_2_avgpool);  features_7_0_block_2_avgpool = None
        features_7_0_block_2_activation = getattr(getattr(getattr(self.features, "7"), "0").block, "2").activation(features_7_0_block_2_fc1);  features_7_0_block_2_fc1 = None
        features_7_0_block_2_fc2 = getattr(getattr(getattr(self.features, "7"), "0").block, "2").fc2(features_7_0_block_2_activation);  features_7_0_block_2_activation = None
        features_7_0_block_2_scale_activation = getattr(getattr(getattr(self.features, "7"), "0").block, "2").scale_activation(features_7_0_block_2_fc2);  features_7_0_block_2_fc2 = None
        mul_15 = features_7_0_block_2_scale_activation * features_7_0_block_1_2;  features_7_0_block_2_scale_activation = features_7_0_block_1_2 = None
        features_7_0_block_3_0 = getattr(getattr(getattr(getattr(self.features, "7"), "0").block, "3"), "0")(mul_15);  mul_15 = None
        features_7_0_block_3_1 = getattr(getattr(getattr(getattr(self.features, "7"), "0").block, "3"), "1")(features_7_0_block_3_0);  features_7_0_block_3_0 = None
        features_8_0 = getattr(getattr(self.features, "8"), "0")(features_7_0_block_3_1);  features_7_0_block_3_1 = None
        features_8_1 = getattr(getattr(self.features, "8"), "1")(features_8_0);  features_8_0 = None
        features_8_2 = getattr(getattr(self.features, "8"), "2")(features_8_1);  features_8_1 = None
        avgpool = self.avgpool(features_8_2);  features_8_2 = None
        flatten = torch.flatten(avgpool, 1);  avgpool = None
        classifier_0 = getattr(self.classifier, "0")(flatten);  flatten = None
        classifier_1 = getattr(self.classifier, "1")(classifier_0);  classifier_0 = None
        return classifier_1
```

## Relax program

```python
from tvm.script import ir as I
from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(inp_0: R.Tensor((1, 3, 224, 224), dtype="float32")) -> R.Tensor((1, 1000), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((1, 32, 112, 112), dtype="float32") = R.nn.conv2d(inp_0, metadata["relax.expr.Constant"][0], strides=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv1: R.Tuple(R.Tensor((1, 32, 112, 112), dtype="float32"), R.Tensor((32,), dtype="float32"), R.Tensor((32,), dtype="float32")) = R.nn.batch_norm(lv, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2], metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv2: R.Tensor((1, 32, 112, 112), dtype="float32") = lv1[0]
            lv3: R.Tensor((1, 32, 112, 112), dtype="float32") = R.nn.silu(lv2)
            lv4: R.Tensor((1, 32, 112, 112), dtype="float32") = R.nn.conv2d(lv3, metadata["relax.expr.Constant"][5], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=32, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv5: R.Tuple(R.Tensor((1, 32, 112, 112), dtype="float32"), R.Tensor((32,), dtype="float32"), R.Tensor((32,), dtype="float32")) = R.nn.batch_norm(lv4, metadata["relax.expr.Constant"][6], metadata["relax.expr.Constant"][7], metadata["relax.expr.Constant"][8], metadata["relax.expr.Constant"][9], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv6: R.Tensor((1, 32, 112, 112), dtype="float32") = lv5[0]
            lv7: R.Tensor((1, 32, 112, 112), dtype="float32") = R.nn.silu(lv6)
            lv8: R.Tensor((1, 32, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv7, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv9: R.Tensor((1, 8, 1, 1), dtype="float32") = R.nn.conv2d(lv8, metadata["relax.expr.Constant"][10], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv10: R.Tensor((1, 8, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][11], R.shape([1, 8, 1, 1]))
            lv11: R.Tensor((1, 8, 1, 1), dtype="float32") = R.add(lv9, lv10)
            lv12: R.Tensor((1, 8, 1, 1), dtype="float32") = R.nn.silu(lv11)
            lv13: R.Tensor((1, 32, 1, 1), dtype="float32") = R.nn.conv2d(lv12, metadata["relax.expr.Constant"][12], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv14: R.Tensor((1, 32, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][13], R.shape([1, 32, 1, 1]))
            lv15: R.Tensor((1, 32, 1, 1), dtype="float32") = R.add(lv13, lv14)
            lv16: R.Tensor((1, 32, 1, 1), dtype="float32") = R.sigmoid(lv15)
            lv17: R.Tensor((1, 32, 112, 112), dtype="float32") = R.multiply(lv16, lv7)
            lv18: R.Tensor((1, 16, 112, 112), dtype="float32") = R.nn.conv2d(lv17, metadata["relax.expr.Constant"][14], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv19: R.Tuple(R.Tensor((1, 16, 112, 112), dtype="float32"), R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")) = R.nn.batch_norm(lv18, metadata["relax.expr.Constant"][15], metadata["relax.expr.Constant"][16], metadata["relax.expr.Constant"][17], metadata["relax.expr.Constant"][18], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv20: R.Tensor((1, 16, 112, 112), dtype="float32") = lv19[0]
            lv21: R.Tensor((1, 96, 112, 112), dtype="float32") = R.nn.conv2d(lv20, metadata["relax.expr.Constant"][19], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv22: R.Tuple(R.Tensor((1, 96, 112, 112), dtype="float32"), R.Tensor((96,), dtype="float32"), R.Tensor((96,), dtype="float32")) = R.nn.batch_norm(lv21, metadata["relax.expr.Constant"][20], metadata["relax.expr.Constant"][21], metadata["relax.expr.Constant"][22], metadata["relax.expr.Constant"][23], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv23: R.Tensor((1, 96, 112, 112), dtype="float32") = lv22[0]
            lv24: R.Tensor((1, 96, 112, 112), dtype="float32") = R.nn.silu(lv23)
            lv25: R.Tensor((1, 96, 56, 56), dtype="float32") = R.nn.conv2d(lv24, metadata["relax.expr.Constant"][24], strides=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=96, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv26: R.Tuple(R.Tensor((1, 96, 56, 56), dtype="float32"), R.Tensor((96,), dtype="float32"), R.Tensor((96,), dtype="float32")) = R.nn.batch_norm(lv25, metadata["relax.expr.Constant"][25], metadata["relax.expr.Constant"][26], metadata["relax.expr.Constant"][27], metadata["relax.expr.Constant"][28], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv27: R.Tensor((1, 96, 56, 56), dtype="float32") = lv26[0]
            lv28: R.Tensor((1, 96, 56, 56), dtype="float32") = R.nn.silu(lv27)
            lv29: R.Tensor((1, 96, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv28, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv30: R.Tensor((1, 4, 1, 1), dtype="float32") = R.nn.conv2d(lv29, metadata["relax.expr.Constant"][29], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv31: R.Tensor((1, 4, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][30], R.shape([1, 4, 1, 1]))
            lv32: R.Tensor((1, 4, 1, 1), dtype="float32") = R.add(lv30, lv31)
            lv33: R.Tensor((1, 4, 1, 1), dtype="float32") = R.nn.silu(lv32)
            lv34: R.Tensor((1, 96, 1, 1), dtype="float32") = R.nn.conv2d(lv33, metadata["relax.expr.Constant"][31], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv35: R.Tensor((1, 96, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][32], R.shape([1, 96, 1, 1]))
            lv36: R.Tensor((1, 96, 1, 1), dtype="float32") = R.add(lv34, lv35)
            lv37: R.Tensor((1, 96, 1, 1), dtype="float32") = R.sigmoid(lv36)
            lv38: R.Tensor((1, 96, 56, 56), dtype="float32") = R.multiply(lv37, lv28)
            lv39: R.Tensor((1, 24, 56, 56), dtype="float32") = R.nn.conv2d(lv38, metadata["relax.expr.Constant"][33], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv40: R.Tuple(R.Tensor((1, 24, 56, 56), dtype="float32"), R.Tensor((24,), dtype="float32"), R.Tensor((24,), dtype="float32")) = R.nn.batch_norm(lv39, metadata["relax.expr.Constant"][34], metadata["relax.expr.Constant"][35], metadata["relax.expr.Constant"][36], metadata["relax.expr.Constant"][37], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv41: R.Tensor((1, 24, 56, 56), dtype="float32") = lv40[0]
            lv42: R.Tensor((1, 144, 56, 56), dtype="float32") = R.nn.conv2d(lv41, metadata["relax.expr.Constant"][38], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv43: R.Tuple(R.Tensor((1, 144, 56, 56), dtype="float32"), R.Tensor((144,), dtype="float32"), R.Tensor((144,), dtype="float32")) = R.nn.batch_norm(lv42, metadata["relax.expr.Constant"][39], metadata["relax.expr.Constant"][40], metadata["relax.expr.Constant"][41], metadata["relax.expr.Constant"][42], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv44: R.Tensor((1, 144, 56, 56), dtype="float32") = lv43[0]
            lv45: R.Tensor((1, 144, 56, 56), dtype="float32") = R.nn.silu(lv44)
            lv46: R.Tensor((1, 144, 56, 56), dtype="float32") = R.nn.conv2d(lv45, metadata["relax.expr.Constant"][43], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=144, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv47: R.Tuple(R.Tensor((1, 144, 56, 56), dtype="float32"), R.Tensor((144,), dtype="float32"), R.Tensor((144,), dtype="float32")) = R.nn.batch_norm(lv46, metadata["relax.expr.Constant"][44], metadata["relax.expr.Constant"][45], metadata["relax.expr.Constant"][46], metadata["relax.expr.Constant"][47], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv48: R.Tensor((1, 144, 56, 56), dtype="float32") = lv47[0]
            lv49: R.Tensor((1, 144, 56, 56), dtype="float32") = R.nn.silu(lv48)
            lv50: R.Tensor((1, 144, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv49, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv51: R.Tensor((1, 6, 1, 1), dtype="float32") = R.nn.conv2d(lv50, metadata["relax.expr.Constant"][48], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv52: R.Tensor((1, 6, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][49], R.shape([1, 6, 1, 1]))
            lv53: R.Tensor((1, 6, 1, 1), dtype="float32") = R.add(lv51, lv52)
            lv54: R.Tensor((1, 6, 1, 1), dtype="float32") = R.nn.silu(lv53)
            lv55: R.Tensor((1, 144, 1, 1), dtype="float32") = R.nn.conv2d(lv54, metadata["relax.expr.Constant"][50], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv56: R.Tensor((1, 144, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][51], R.shape([1, 144, 1, 1]))
            lv57: R.Tensor((1, 144, 1, 1), dtype="float32") = R.add(lv55, lv56)
            lv58: R.Tensor((1, 144, 1, 1), dtype="float32") = R.sigmoid(lv57)
            lv59: R.Tensor((1, 144, 56, 56), dtype="float32") = R.multiply(lv58, lv49)
            lv60: R.Tensor((1, 24, 56, 56), dtype="float32") = R.nn.conv2d(lv59, metadata["relax.expr.Constant"][52], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv61: R.Tuple(R.Tensor((1, 24, 56, 56), dtype="float32"), R.Tensor((24,), dtype="float32"), R.Tensor((24,), dtype="float32")) = R.nn.batch_norm(lv60, metadata["relax.expr.Constant"][53], metadata["relax.expr.Constant"][54], metadata["relax.expr.Constant"][55], metadata["relax.expr.Constant"][56], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv62: R.Tensor((1, 24, 56, 56), dtype="float32") = lv61[0]
            lv63: R.Tensor((1, 24, 56, 56), dtype="float32") = lv62
            lv64: R.Tensor((1, 24, 56, 56), dtype="float32") = R.add(lv63, lv41)
            lv65: R.Tensor((1, 144, 56, 56), dtype="float32") = R.nn.conv2d(lv64, metadata["relax.expr.Constant"][57], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv66: R.Tuple(R.Tensor((1, 144, 56, 56), dtype="float32"), R.Tensor((144,), dtype="float32"), R.Tensor((144,), dtype="float32")) = R.nn.batch_norm(lv65, metadata["relax.expr.Constant"][58], metadata["relax.expr.Constant"][59], metadata["relax.expr.Constant"][60], metadata["relax.expr.Constant"][61], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv67: R.Tensor((1, 144, 56, 56), dtype="float32") = lv66[0]
            lv68: R.Tensor((1, 144, 56, 56), dtype="float32") = R.nn.silu(lv67)
            lv69: R.Tensor((1, 144, 28, 28), dtype="float32") = R.nn.conv2d(lv68, metadata["relax.expr.Constant"][62], strides=[2, 2], padding=[2, 2, 2, 2], dilation=[1, 1], groups=144, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv70: R.Tuple(R.Tensor((1, 144, 28, 28), dtype="float32"), R.Tensor((144,), dtype="float32"), R.Tensor((144,), dtype="float32")) = R.nn.batch_norm(lv69, metadata["relax.expr.Constant"][63], metadata["relax.expr.Constant"][64], metadata["relax.expr.Constant"][65], metadata["relax.expr.Constant"][66], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv71: R.Tensor((1, 144, 28, 28), dtype="float32") = lv70[0]
            lv72: R.Tensor((1, 144, 28, 28), dtype="float32") = R.nn.silu(lv71)
            lv73: R.Tensor((1, 144, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv72, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv74: R.Tensor((1, 6, 1, 1), dtype="float32") = R.nn.conv2d(lv73, metadata["relax.expr.Constant"][67], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv75: R.Tensor((1, 6, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][68], R.shape([1, 6, 1, 1]))
            lv76: R.Tensor((1, 6, 1, 1), dtype="float32") = R.add(lv74, lv75)
            lv77: R.Tensor((1, 6, 1, 1), dtype="float32") = R.nn.silu(lv76)
            lv78: R.Tensor((1, 144, 1, 1), dtype="float32") = R.nn.conv2d(lv77, metadata["relax.expr.Constant"][69], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv79: R.Tensor((1, 144, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][70], R.shape([1, 144, 1, 1]))
            lv80: R.Tensor((1, 144, 1, 1), dtype="float32") = R.add(lv78, lv79)
            lv81: R.Tensor((1, 144, 1, 1), dtype="float32") = R.sigmoid(lv80)
            lv82: R.Tensor((1, 144, 28, 28), dtype="float32") = R.multiply(lv81, lv72)
            lv83: R.Tensor((1, 40, 28, 28), dtype="float32") = R.nn.conv2d(lv82, metadata["relax.expr.Constant"][71], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv84: R.Tuple(R.Tensor((1, 40, 28, 28), dtype="float32"), R.Tensor((40,), dtype="float32"), R.Tensor((40,), dtype="float32")) = R.nn.batch_norm(lv83, metadata["relax.expr.Constant"][72], metadata["relax.expr.Constant"][73], metadata["relax.expr.Constant"][74], metadata["relax.expr.Constant"][75], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv85: R.Tensor((1, 40, 28, 28), dtype="float32") = lv84[0]
            lv86: R.Tensor((1, 240, 28, 28), dtype="float32") = R.nn.conv2d(lv85, metadata["relax.expr.Constant"][76], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv87: R.Tuple(R.Tensor((1, 240, 28, 28), dtype="float32"), R.Tensor((240,), dtype="float32"), R.Tensor((240,), dtype="float32")) = R.nn.batch_norm(lv86, metadata["relax.expr.Constant"][77], metadata["relax.expr.Constant"][78], metadata["relax.expr.Constant"][79], metadata["relax.expr.Constant"][80], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv88: R.Tensor((1, 240, 28, 28), dtype="float32") = lv87[0]
            lv89: R.Tensor((1, 240, 28, 28), dtype="float32") = R.nn.silu(lv88)
            lv90: R.Tensor((1, 240, 28, 28), dtype="float32") = R.nn.conv2d(lv89, metadata["relax.expr.Constant"][81], strides=[1, 1], padding=[2, 2, 2, 2], dilation=[1, 1], groups=240, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv91: R.Tuple(R.Tensor((1, 240, 28, 28), dtype="float32"), R.Tensor((240,), dtype="float32"), R.Tensor((240,), dtype="float32")) = R.nn.batch_norm(lv90, metadata["relax.expr.Constant"][82], metadata["relax.expr.Constant"][83], metadata["relax.expr.Constant"][84], metadata["relax.expr.Constant"][85], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv92: R.Tensor((1, 240, 28, 28), dtype="float32") = lv91[0]
            lv93: R.Tensor((1, 240, 28, 28), dtype="float32") = R.nn.silu(lv92)
            lv94: R.Tensor((1, 240, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv93, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv95: R.Tensor((1, 10, 1, 1), dtype="float32") = R.nn.conv2d(lv94, metadata["relax.expr.Constant"][86], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv96: R.Tensor((1, 10, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][87], R.shape([1, 10, 1, 1]))
            lv97: R.Tensor((1, 10, 1, 1), dtype="float32") = R.add(lv95, lv96)
            lv98: R.Tensor((1, 10, 1, 1), dtype="float32") = R.nn.silu(lv97)
            lv99: R.Tensor((1, 240, 1, 1), dtype="float32") = R.nn.conv2d(lv98, metadata["relax.expr.Constant"][88], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv100: R.Tensor((1, 240, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][89], R.shape([1, 240, 1, 1]))
            lv101: R.Tensor((1, 240, 1, 1), dtype="float32") = R.add(lv99, lv100)
            lv102: R.Tensor((1, 240, 1, 1), dtype="float32") = R.sigmoid(lv101)
            lv103: R.Tensor((1, 240, 28, 28), dtype="float32") = R.multiply(lv102, lv93)
            lv104: R.Tensor((1, 40, 28, 28), dtype="float32") = R.nn.conv2d(lv103, metadata["relax.expr.Constant"][90], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv105: R.Tuple(R.Tensor((1, 40, 28, 28), dtype="float32"), R.Tensor((40,), dtype="float32"), R.Tensor((40,), dtype="float32")) = R.nn.batch_norm(lv104, metadata["relax.expr.Constant"][91], metadata["relax.expr.Constant"][92], metadata["relax.expr.Constant"][93], metadata["relax.expr.Constant"][94], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv106: R.Tensor((1, 40, 28, 28), dtype="float32") = lv105[0]
            lv107: R.Tensor((1, 40, 28, 28), dtype="float32") = lv106
            lv108: R.Tensor((1, 40, 28, 28), dtype="float32") = R.add(lv107, lv85)
            lv109: R.Tensor((1, 240, 28, 28), dtype="float32") = R.nn.conv2d(lv108, metadata["relax.expr.Constant"][95], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv110: R.Tuple(R.Tensor((1, 240, 28, 28), dtype="float32"), R.Tensor((240,), dtype="float32"), R.Tensor((240,), dtype="float32")) = R.nn.batch_norm(lv109, metadata["relax.expr.Constant"][96], metadata["relax.expr.Constant"][97], metadata["relax.expr.Constant"][98], metadata["relax.expr.Constant"][99], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv111: R.Tensor((1, 240, 28, 28), dtype="float32") = lv110[0]
            lv112: R.Tensor((1, 240, 28, 28), dtype="float32") = R.nn.silu(lv111)
            lv113: R.Tensor((1, 240, 14, 14), dtype="float32") = R.nn.conv2d(lv112, metadata["relax.expr.Constant"][100], strides=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=240, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv114: R.Tuple(R.Tensor((1, 240, 14, 14), dtype="float32"), R.Tensor((240,), dtype="float32"), R.Tensor((240,), dtype="float32")) = R.nn.batch_norm(lv113, metadata["relax.expr.Constant"][101], metadata["relax.expr.Constant"][102], metadata["relax.expr.Constant"][103], metadata["relax.expr.Constant"][104], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv115: R.Tensor((1, 240, 14, 14), dtype="float32") = lv114[0]
            lv116: R.Tensor((1, 240, 14, 14), dtype="float32") = R.nn.silu(lv115)
            lv117: R.Tensor((1, 240, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv116, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv118: R.Tensor((1, 10, 1, 1), dtype="float32") = R.nn.conv2d(lv117, metadata["relax.expr.Constant"][105], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv119: R.Tensor((1, 10, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][106], R.shape([1, 10, 1, 1]))
            lv120: R.Tensor((1, 10, 1, 1), dtype="float32") = R.add(lv118, lv119)
            lv121: R.Tensor((1, 10, 1, 1), dtype="float32") = R.nn.silu(lv120)
            lv122: R.Tensor((1, 240, 1, 1), dtype="float32") = R.nn.conv2d(lv121, metadata["relax.expr.Constant"][107], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv123: R.Tensor((1, 240, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][108], R.shape([1, 240, 1, 1]))
            lv124: R.Tensor((1, 240, 1, 1), dtype="float32") = R.add(lv122, lv123)
            lv125: R.Tensor((1, 240, 1, 1), dtype="float32") = R.sigmoid(lv124)
            lv126: R.Tensor((1, 240, 14, 14), dtype="float32") = R.multiply(lv125, lv116)
            lv127: R.Tensor((1, 80, 14, 14), dtype="float32") = R.nn.conv2d(lv126, metadata["relax.expr.Constant"][109], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv128: R.Tuple(R.Tensor((1, 80, 14, 14), dtype="float32"), R.Tensor((80,), dtype="float32"), R.Tensor((80,), dtype="float32")) = R.nn.batch_norm(lv127, metadata["relax.expr.Constant"][110], metadata["relax.expr.Constant"][111], metadata["relax.expr.Constant"][112], metadata["relax.expr.Constant"][113], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv129: R.Tensor((1, 80, 14, 14), dtype="float32") = lv128[0]
            lv130: R.Tensor((1, 480, 14, 14), dtype="float32") = R.nn.conv2d(lv129, metadata["relax.expr.Constant"][114], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv131: R.Tuple(R.Tensor((1, 480, 14, 14), dtype="float32"), R.Tensor((480,), dtype="float32"), R.Tensor((480,), dtype="float32")) = R.nn.batch_norm(lv130, metadata["relax.expr.Constant"][115], metadata["relax.expr.Constant"][116], metadata["relax.expr.Constant"][117], metadata["relax.expr.Constant"][118], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv132: R.Tensor((1, 480, 14, 14), dtype="float32") = lv131[0]
            lv133: R.Tensor((1, 480, 14, 14), dtype="float32") = R.nn.silu(lv132)
            lv134: R.Tensor((1, 480, 14, 14), dtype="float32") = R.nn.conv2d(lv133, metadata["relax.expr.Constant"][119], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=480, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv135: R.Tuple(R.Tensor((1, 480, 14, 14), dtype="float32"), R.Tensor((480,), dtype="float32"), R.Tensor((480,), dtype="float32")) = R.nn.batch_norm(lv134, metadata["relax.expr.Constant"][120], metadata["relax.expr.Constant"][121], metadata["relax.expr.Constant"][122], metadata["relax.expr.Constant"][123], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv136: R.Tensor((1, 480, 14, 14), dtype="float32") = lv135[0]
            lv137: R.Tensor((1, 480, 14, 14), dtype="float32") = R.nn.silu(lv136)
            lv138: R.Tensor((1, 480, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv137, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv139: R.Tensor((1, 20, 1, 1), dtype="float32") = R.nn.conv2d(lv138, metadata["relax.expr.Constant"][124], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv140: R.Tensor((1, 20, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][125], R.shape([1, 20, 1, 1]))
            lv141: R.Tensor((1, 20, 1, 1), dtype="float32") = R.add(lv139, lv140)
            lv142: R.Tensor((1, 20, 1, 1), dtype="float32") = R.nn.silu(lv141)
            lv143: R.Tensor((1, 480, 1, 1), dtype="float32") = R.nn.conv2d(lv142, metadata["relax.expr.Constant"][126], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv144: R.Tensor((1, 480, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][127], R.shape([1, 480, 1, 1]))
            lv145: R.Tensor((1, 480, 1, 1), dtype="float32") = R.add(lv143, lv144)
            lv146: R.Tensor((1, 480, 1, 1), dtype="float32") = R.sigmoid(lv145)
            lv147: R.Tensor((1, 480, 14, 14), dtype="float32") = R.multiply(lv146, lv137)
            lv148: R.Tensor((1, 80, 14, 14), dtype="float32") = R.nn.conv2d(lv147, metadata["relax.expr.Constant"][128], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv149: R.Tuple(R.Tensor((1, 80, 14, 14), dtype="float32"), R.Tensor((80,), dtype="float32"), R.Tensor((80,), dtype="float32")) = R.nn.batch_norm(lv148, metadata["relax.expr.Constant"][129], metadata["relax.expr.Constant"][130], metadata["relax.expr.Constant"][131], metadata["relax.expr.Constant"][132], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv150: R.Tensor((1, 80, 14, 14), dtype="float32") = lv149[0]
            lv151: R.Tensor((1, 80, 14, 14), dtype="float32") = lv150
            lv152: R.Tensor((1, 80, 14, 14), dtype="float32") = R.add(lv151, lv129)
            lv153: R.Tensor((1, 480, 14, 14), dtype="float32") = R.nn.conv2d(lv152, metadata["relax.expr.Constant"][133], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv154: R.Tuple(R.Tensor((1, 480, 14, 14), dtype="float32"), R.Tensor((480,), dtype="float32"), R.Tensor((480,), dtype="float32")) = R.nn.batch_norm(lv153, metadata["relax.expr.Constant"][134], metadata["relax.expr.Constant"][135], metadata["relax.expr.Constant"][136], metadata["relax.expr.Constant"][137], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv155: R.Tensor((1, 480, 14, 14), dtype="float32") = lv154[0]
            lv156: R.Tensor((1, 480, 14, 14), dtype="float32") = R.nn.silu(lv155)
            lv157: R.Tensor((1, 480, 14, 14), dtype="float32") = R.nn.conv2d(lv156, metadata["relax.expr.Constant"][138], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=480, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv158: R.Tuple(R.Tensor((1, 480, 14, 14), dtype="float32"), R.Tensor((480,), dtype="float32"), R.Tensor((480,), dtype="float32")) = R.nn.batch_norm(lv157, metadata["relax.expr.Constant"][139], metadata["relax.expr.Constant"][140], metadata["relax.expr.Constant"][141], metadata["relax.expr.Constant"][142], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv159: R.Tensor((1, 480, 14, 14), dtype="float32") = lv158[0]
            lv160: R.Tensor((1, 480, 14, 14), dtype="float32") = R.nn.silu(lv159)
            lv161: R.Tensor((1, 480, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv160, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv162: R.Tensor((1, 20, 1, 1), dtype="float32") = R.nn.conv2d(lv161, metadata["relax.expr.Constant"][143], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv163: R.Tensor((1, 20, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][144], R.shape([1, 20, 1, 1]))
            lv164: R.Tensor((1, 20, 1, 1), dtype="float32") = R.add(lv162, lv163)
            lv165: R.Tensor((1, 20, 1, 1), dtype="float32") = R.nn.silu(lv164)
            lv166: R.Tensor((1, 480, 1, 1), dtype="float32") = R.nn.conv2d(lv165, metadata["relax.expr.Constant"][145], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv167: R.Tensor((1, 480, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][146], R.shape([1, 480, 1, 1]))
            lv168: R.Tensor((1, 480, 1, 1), dtype="float32") = R.add(lv166, lv167)
            lv169: R.Tensor((1, 480, 1, 1), dtype="float32") = R.sigmoid(lv168)
            lv170: R.Tensor((1, 480, 14, 14), dtype="float32") = R.multiply(lv169, lv160)
            lv171: R.Tensor((1, 80, 14, 14), dtype="float32") = R.nn.conv2d(lv170, metadata["relax.expr.Constant"][147], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv172: R.Tuple(R.Tensor((1, 80, 14, 14), dtype="float32"), R.Tensor((80,), dtype="float32"), R.Tensor((80,), dtype="float32")) = R.nn.batch_norm(lv171, metadata["relax.expr.Constant"][148], metadata["relax.expr.Constant"][149], metadata["relax.expr.Constant"][150], metadata["relax.expr.Constant"][151], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv173: R.Tensor((1, 80, 14, 14), dtype="float32") = lv172[0]
            lv174: R.Tensor((1, 80, 14, 14), dtype="float32") = lv173
            lv175: R.Tensor((1, 80, 14, 14), dtype="float32") = R.add(lv174, lv152)
            lv176: R.Tensor((1, 480, 14, 14), dtype="float32") = R.nn.conv2d(lv175, metadata["relax.expr.Constant"][152], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv177: R.Tuple(R.Tensor((1, 480, 14, 14), dtype="float32"), R.Tensor((480,), dtype="float32"), R.Tensor((480,), dtype="float32")) = R.nn.batch_norm(lv176, metadata["relax.expr.Constant"][153], metadata["relax.expr.Constant"][154], metadata["relax.expr.Constant"][155], metadata["relax.expr.Constant"][156], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv178: R.Tensor((1, 480, 14, 14), dtype="float32") = lv177[0]
            lv179: R.Tensor((1, 480, 14, 14), dtype="float32") = R.nn.silu(lv178)
            lv180: R.Tensor((1, 480, 14, 14), dtype="float32") = R.nn.conv2d(lv179, metadata["relax.expr.Constant"][157], strides=[1, 1], padding=[2, 2, 2, 2], dilation=[1, 1], groups=480, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv181: R.Tuple(R.Tensor((1, 480, 14, 14), dtype="float32"), R.Tensor((480,), dtype="float32"), R.Tensor((480,), dtype="float32")) = R.nn.batch_norm(lv180, metadata["relax.expr.Constant"][158], metadata["relax.expr.Constant"][159], metadata["relax.expr.Constant"][160], metadata["relax.expr.Constant"][161], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv182: R.Tensor((1, 480, 14, 14), dtype="float32") = lv181[0]
            lv183: R.Tensor((1, 480, 14, 14), dtype="float32") = R.nn.silu(lv182)
            lv184: R.Tensor((1, 480, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv183, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv185: R.Tensor((1, 20, 1, 1), dtype="float32") = R.nn.conv2d(lv184, metadata["relax.expr.Constant"][162], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv186: R.Tensor((1, 20, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][163], R.shape([1, 20, 1, 1]))
            lv187: R.Tensor((1, 20, 1, 1), dtype="float32") = R.add(lv185, lv186)
            lv188: R.Tensor((1, 20, 1, 1), dtype="float32") = R.nn.silu(lv187)
            lv189: R.Tensor((1, 480, 1, 1), dtype="float32") = R.nn.conv2d(lv188, metadata["relax.expr.Constant"][164], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv190: R.Tensor((1, 480, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][165], R.shape([1, 480, 1, 1]))
            lv191: R.Tensor((1, 480, 1, 1), dtype="float32") = R.add(lv189, lv190)
            lv192: R.Tensor((1, 480, 1, 1), dtype="float32") = R.sigmoid(lv191)
            lv193: R.Tensor((1, 480, 14, 14), dtype="float32") = R.multiply(lv192, lv183)
            lv194: R.Tensor((1, 112, 14, 14), dtype="float32") = R.nn.conv2d(lv193, metadata["relax.expr.Constant"][166], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv195: R.Tuple(R.Tensor((1, 112, 14, 14), dtype="float32"), R.Tensor((112,), dtype="float32"), R.Tensor((112,), dtype="float32")) = R.nn.batch_norm(lv194, metadata["relax.expr.Constant"][167], metadata["relax.expr.Constant"][168], metadata["relax.expr.Constant"][169], metadata["relax.expr.Constant"][170], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv196: R.Tensor((1, 112, 14, 14), dtype="float32") = lv195[0]
            lv197: R.Tensor((1, 672, 14, 14), dtype="float32") = R.nn.conv2d(lv196, metadata["relax.expr.Constant"][171], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv198: R.Tuple(R.Tensor((1, 672, 14, 14), dtype="float32"), R.Tensor((672,), dtype="float32"), R.Tensor((672,), dtype="float32")) = R.nn.batch_norm(lv197, metadata["relax.expr.Constant"][172], metadata["relax.expr.Constant"][173], metadata["relax.expr.Constant"][174], metadata["relax.expr.Constant"][175], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv199: R.Tensor((1, 672, 14, 14), dtype="float32") = lv198[0]
            lv200: R.Tensor((1, 672, 14, 14), dtype="float32") = R.nn.silu(lv199)
            lv201: R.Tensor((1, 672, 14, 14), dtype="float32") = R.nn.conv2d(lv200, metadata["relax.expr.Constant"][176], strides=[1, 1], padding=[2, 2, 2, 2], dilation=[1, 1], groups=672, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv202: R.Tuple(R.Tensor((1, 672, 14, 14), dtype="float32"), R.Tensor((672,), dtype="float32"), R.Tensor((672,), dtype="float32")) = R.nn.batch_norm(lv201, metadata["relax.expr.Constant"][177], metadata["relax.expr.Constant"][178], metadata["relax.expr.Constant"][179], metadata["relax.expr.Constant"][180], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv203: R.Tensor((1, 672, 14, 14), dtype="float32") = lv202[0]
            lv204: R.Tensor((1, 672, 14, 14), dtype="float32") = R.nn.silu(lv203)
            lv205: R.Tensor((1, 672, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv204, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv206: R.Tensor((1, 28, 1, 1), dtype="float32") = R.nn.conv2d(lv205, metadata["relax.expr.Constant"][181], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv207: R.Tensor((1, 28, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][182], R.shape([1, 28, 1, 1]))
            lv208: R.Tensor((1, 28, 1, 1), dtype="float32") = R.add(lv206, lv207)
            lv209: R.Tensor((1, 28, 1, 1), dtype="float32") = R.nn.silu(lv208)
            lv210: R.Tensor((1, 672, 1, 1), dtype="float32") = R.nn.conv2d(lv209, metadata["relax.expr.Constant"][183], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv211: R.Tensor((1, 672, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][184], R.shape([1, 672, 1, 1]))
            lv212: R.Tensor((1, 672, 1, 1), dtype="float32") = R.add(lv210, lv211)
            lv213: R.Tensor((1, 672, 1, 1), dtype="float32") = R.sigmoid(lv212)
            lv214: R.Tensor((1, 672, 14, 14), dtype="float32") = R.multiply(lv213, lv204)
            lv215: R.Tensor((1, 112, 14, 14), dtype="float32") = R.nn.conv2d(lv214, metadata["relax.expr.Constant"][185], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv216: R.Tuple(R.Tensor((1, 112, 14, 14), dtype="float32"), R.Tensor((112,), dtype="float32"), R.Tensor((112,), dtype="float32")) = R.nn.batch_norm(lv215, metadata["relax.expr.Constant"][186], metadata["relax.expr.Constant"][187], metadata["relax.expr.Constant"][188], metadata["relax.expr.Constant"][189], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv217: R.Tensor((1, 112, 14, 14), dtype="float32") = lv216[0]
            lv218: R.Tensor((1, 112, 14, 14), dtype="float32") = lv217
            lv219: R.Tensor((1, 112, 14, 14), dtype="float32") = R.add(lv218, lv196)
            lv220: R.Tensor((1, 672, 14, 14), dtype="float32") = R.nn.conv2d(lv219, metadata["relax.expr.Constant"][190], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv221: R.Tuple(R.Tensor((1, 672, 14, 14), dtype="float32"), R.Tensor((672,), dtype="float32"), R.Tensor((672,), dtype="float32")) = R.nn.batch_norm(lv220, metadata["relax.expr.Constant"][191], metadata["relax.expr.Constant"][192], metadata["relax.expr.Constant"][193], metadata["relax.expr.Constant"][194], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv222: R.Tensor((1, 672, 14, 14), dtype="float32") = lv221[0]
            lv223: R.Tensor((1, 672, 14, 14), dtype="float32") = R.nn.silu(lv222)
            lv224: R.Tensor((1, 672, 14, 14), dtype="float32") = R.nn.conv2d(lv223, metadata["relax.expr.Constant"][195], strides=[1, 1], padding=[2, 2, 2, 2], dilation=[1, 1], groups=672, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv225: R.Tuple(R.Tensor((1, 672, 14, 14), dtype="float32"), R.Tensor((672,), dtype="float32"), R.Tensor((672,), dtype="float32")) = R.nn.batch_norm(lv224, metadata["relax.expr.Constant"][196], metadata["relax.expr.Constant"][197], metadata["relax.expr.Constant"][198], metadata["relax.expr.Constant"][199], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv226: R.Tensor((1, 672, 14, 14), dtype="float32") = lv225[0]
            lv227: R.Tensor((1, 672, 14, 14), dtype="float32") = R.nn.silu(lv226)
            lv228: R.Tensor((1, 672, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv227, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv229: R.Tensor((1, 28, 1, 1), dtype="float32") = R.nn.conv2d(lv228, metadata["relax.expr.Constant"][200], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv230: R.Tensor((1, 28, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][201], R.shape([1, 28, 1, 1]))
            lv231: R.Tensor((1, 28, 1, 1), dtype="float32") = R.add(lv229, lv230)
            lv232: R.Tensor((1, 28, 1, 1), dtype="float32") = R.nn.silu(lv231)
            lv233: R.Tensor((1, 672, 1, 1), dtype="float32") = R.nn.conv2d(lv232, metadata["relax.expr.Constant"][202], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv234: R.Tensor((1, 672, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][203], R.shape([1, 672, 1, 1]))
            lv235: R.Tensor((1, 672, 1, 1), dtype="float32") = R.add(lv233, lv234)
            lv236: R.Tensor((1, 672, 1, 1), dtype="float32") = R.sigmoid(lv235)
            lv237: R.Tensor((1, 672, 14, 14), dtype="float32") = R.multiply(lv236, lv227)
            lv238: R.Tensor((1, 112, 14, 14), dtype="float32") = R.nn.conv2d(lv237, metadata["relax.expr.Constant"][204], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv239: R.Tuple(R.Tensor((1, 112, 14, 14), dtype="float32"), R.Tensor((112,), dtype="float32"), R.Tensor((112,), dtype="float32")) = R.nn.batch_norm(lv238, metadata["relax.expr.Constant"][205], metadata["relax.expr.Constant"][206], metadata["relax.expr.Constant"][207], metadata["relax.expr.Constant"][208], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv240: R.Tensor((1, 112, 14, 14), dtype="float32") = lv239[0]
            lv241: R.Tensor((1, 112, 14, 14), dtype="float32") = lv240
            lv242: R.Tensor((1, 112, 14, 14), dtype="float32") = R.add(lv241, lv219)
            lv243: R.Tensor((1, 672, 14, 14), dtype="float32") = R.nn.conv2d(lv242, metadata["relax.expr.Constant"][209], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv244: R.Tuple(R.Tensor((1, 672, 14, 14), dtype="float32"), R.Tensor((672,), dtype="float32"), R.Tensor((672,), dtype="float32")) = R.nn.batch_norm(lv243, metadata["relax.expr.Constant"][210], metadata["relax.expr.Constant"][211], metadata["relax.expr.Constant"][212], metadata["relax.expr.Constant"][213], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv245: R.Tensor((1, 672, 14, 14), dtype="float32") = lv244[0]
            lv246: R.Tensor((1, 672, 14, 14), dtype="float32") = R.nn.silu(lv245)
            lv247: R.Tensor((1, 672, 7, 7), dtype="float32") = R.nn.conv2d(lv246, metadata["relax.expr.Constant"][214], strides=[2, 2], padding=[2, 2, 2, 2], dilation=[1, 1], groups=672, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv248: R.Tuple(R.Tensor((1, 672, 7, 7), dtype="float32"), R.Tensor((672,), dtype="float32"), R.Tensor((672,), dtype="float32")) = R.nn.batch_norm(lv247, metadata["relax.expr.Constant"][215], metadata["relax.expr.Constant"][216], metadata["relax.expr.Constant"][217], metadata["relax.expr.Constant"][218], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv249: R.Tensor((1, 672, 7, 7), dtype="float32") = lv248[0]
            lv250: R.Tensor((1, 672, 7, 7), dtype="float32") = R.nn.silu(lv249)
            lv251: R.Tensor((1, 672, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv250, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv252: R.Tensor((1, 28, 1, 1), dtype="float32") = R.nn.conv2d(lv251, metadata["relax.expr.Constant"][219], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv253: R.Tensor((1, 28, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][220], R.shape([1, 28, 1, 1]))
            lv254: R.Tensor((1, 28, 1, 1), dtype="float32") = R.add(lv252, lv253)
            lv255: R.Tensor((1, 28, 1, 1), dtype="float32") = R.nn.silu(lv254)
            lv256: R.Tensor((1, 672, 1, 1), dtype="float32") = R.nn.conv2d(lv255, metadata["relax.expr.Constant"][221], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv257: R.Tensor((1, 672, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][222], R.shape([1, 672, 1, 1]))
            lv258: R.Tensor((1, 672, 1, 1), dtype="float32") = R.add(lv256, lv257)
            lv259: R.Tensor((1, 672, 1, 1), dtype="float32") = R.sigmoid(lv258)
            lv260: R.Tensor((1, 672, 7, 7), dtype="float32") = R.multiply(lv259, lv250)
            lv261: R.Tensor((1, 192, 7, 7), dtype="float32") = R.nn.conv2d(lv260, metadata["relax.expr.Constant"][223], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv262: R.Tuple(R.Tensor((1, 192, 7, 7), dtype="float32"), R.Tensor((192,), dtype="float32"), R.Tensor((192,), dtype="float32")) = R.nn.batch_norm(lv261, metadata["relax.expr.Constant"][224], metadata["relax.expr.Constant"][225], metadata["relax.expr.Constant"][226], metadata["relax.expr.Constant"][227], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv263: R.Tensor((1, 192, 7, 7), dtype="float32") = lv262[0]
            lv264: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.conv2d(lv263, metadata["relax.expr.Constant"][228], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv265: R.Tuple(R.Tensor((1, 1152, 7, 7), dtype="float32"), R.Tensor((1152,), dtype="float32"), R.Tensor((1152,), dtype="float32")) = R.nn.batch_norm(lv264, metadata["relax.expr.Constant"][229], metadata["relax.expr.Constant"][230], metadata["relax.expr.Constant"][231], metadata["relax.expr.Constant"][232], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv266: R.Tensor((1, 1152, 7, 7), dtype="float32") = lv265[0]
            lv267: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.silu(lv266)
            lv268: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.conv2d(lv267, metadata["relax.expr.Constant"][233], strides=[1, 1], padding=[2, 2, 2, 2], dilation=[1, 1], groups=1152, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv269: R.Tuple(R.Tensor((1, 1152, 7, 7), dtype="float32"), R.Tensor((1152,), dtype="float32"), R.Tensor((1152,), dtype="float32")) = R.nn.batch_norm(lv268, metadata["relax.expr.Constant"][234], metadata["relax.expr.Constant"][235], metadata["relax.expr.Constant"][236], metadata["relax.expr.Constant"][237], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv270: R.Tensor((1, 1152, 7, 7), dtype="float32") = lv269[0]
            lv271: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.silu(lv270)
            lv272: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv271, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv273: R.Tensor((1, 48, 1, 1), dtype="float32") = R.nn.conv2d(lv272, metadata["relax.expr.Constant"][238], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv274: R.Tensor((1, 48, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][239], R.shape([1, 48, 1, 1]))
            lv275: R.Tensor((1, 48, 1, 1), dtype="float32") = R.add(lv273, lv274)
            lv276: R.Tensor((1, 48, 1, 1), dtype="float32") = R.nn.silu(lv275)
            lv277: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.nn.conv2d(lv276, metadata["relax.expr.Constant"][240], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv278: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][241], R.shape([1, 1152, 1, 1]))
            lv279: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.add(lv277, lv278)
            lv280: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.sigmoid(lv279)
            lv281: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.multiply(lv280, lv271)
            lv282: R.Tensor((1, 192, 7, 7), dtype="float32") = R.nn.conv2d(lv281, metadata["relax.expr.Constant"][242], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv283: R.Tuple(R.Tensor((1, 192, 7, 7), dtype="float32"), R.Tensor((192,), dtype="float32"), R.Tensor((192,), dtype="float32")) = R.nn.batch_norm(lv282, metadata["relax.expr.Constant"][243], metadata["relax.expr.Constant"][244], metadata["relax.expr.Constant"][245], metadata["relax.expr.Constant"][246], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv284: R.Tensor((1, 192, 7, 7), dtype="float32") = lv283[0]
            lv285: R.Tensor((1, 192, 7, 7), dtype="float32") = lv284
            lv286: R.Tensor((1, 192, 7, 7), dtype="float32") = R.add(lv285, lv263)
            lv287: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.conv2d(lv286, metadata["relax.expr.Constant"][247], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv288: R.Tuple(R.Tensor((1, 1152, 7, 7), dtype="float32"), R.Tensor((1152,), dtype="float32"), R.Tensor((1152,), dtype="float32")) = R.nn.batch_norm(lv287, metadata["relax.expr.Constant"][248], metadata["relax.expr.Constant"][249], metadata["relax.expr.Constant"][250], metadata["relax.expr.Constant"][251], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv289: R.Tensor((1, 1152, 7, 7), dtype="float32") = lv288[0]
            lv290: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.silu(lv289)
            lv291: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.conv2d(lv290, metadata["relax.expr.Constant"][252], strides=[1, 1], padding=[2, 2, 2, 2], dilation=[1, 1], groups=1152, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv292: R.Tuple(R.Tensor((1, 1152, 7, 7), dtype="float32"), R.Tensor((1152,), dtype="float32"), R.Tensor((1152,), dtype="float32")) = R.nn.batch_norm(lv291, metadata["relax.expr.Constant"][253], metadata["relax.expr.Constant"][254], metadata["relax.expr.Constant"][255], metadata["relax.expr.Constant"][256], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv293: R.Tensor((1, 1152, 7, 7), dtype="float32") = lv292[0]
            lv294: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.silu(lv293)
            lv295: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv294, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv296: R.Tensor((1, 48, 1, 1), dtype="float32") = R.nn.conv2d(lv295, metadata["relax.expr.Constant"][257], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv297: R.Tensor((1, 48, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][258], R.shape([1, 48, 1, 1]))
            lv298: R.Tensor((1, 48, 1, 1), dtype="float32") = R.add(lv296, lv297)
            lv299: R.Tensor((1, 48, 1, 1), dtype="float32") = R.nn.silu(lv298)
            lv300: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.nn.conv2d(lv299, metadata["relax.expr.Constant"][259], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv301: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][260], R.shape([1, 1152, 1, 1]))
            lv302: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.add(lv300, lv301)
            lv303: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.sigmoid(lv302)
            lv304: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.multiply(lv303, lv294)
            lv305: R.Tensor((1, 192, 7, 7), dtype="float32") = R.nn.conv2d(lv304, metadata["relax.expr.Constant"][261], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv306: R.Tuple(R.Tensor((1, 192, 7, 7), dtype="float32"), R.Tensor((192,), dtype="float32"), R.Tensor((192,), dtype="float32")) = R.nn.batch_norm(lv305, metadata["relax.expr.Constant"][262], metadata["relax.expr.Constant"][263], metadata["relax.expr.Constant"][264], metadata["relax.expr.Constant"][265], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv307: R.Tensor((1, 192, 7, 7), dtype="float32") = lv306[0]
            lv308: R.Tensor((1, 192, 7, 7), dtype="float32") = lv307
            lv309: R.Tensor((1, 192, 7, 7), dtype="float32") = R.add(lv308, lv286)
            lv310: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.conv2d(lv309, metadata["relax.expr.Constant"][266], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv311: R.Tuple(R.Tensor((1, 1152, 7, 7), dtype="float32"), R.Tensor((1152,), dtype="float32"), R.Tensor((1152,), dtype="float32")) = R.nn.batch_norm(lv310, metadata["relax.expr.Constant"][267], metadata["relax.expr.Constant"][268], metadata["relax.expr.Constant"][269], metadata["relax.expr.Constant"][270], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv312: R.Tensor((1, 1152, 7, 7), dtype="float32") = lv311[0]
            lv313: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.silu(lv312)
            lv314: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.conv2d(lv313, metadata["relax.expr.Constant"][271], strides=[1, 1], padding=[2, 2, 2, 2], dilation=[1, 1], groups=1152, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv315: R.Tuple(R.Tensor((1, 1152, 7, 7), dtype="float32"), R.Tensor((1152,), dtype="float32"), R.Tensor((1152,), dtype="float32")) = R.nn.batch_norm(lv314, metadata["relax.expr.Constant"][272], metadata["relax.expr.Constant"][273], metadata["relax.expr.Constant"][274], metadata["relax.expr.Constant"][275], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv316: R.Tensor((1, 1152, 7, 7), dtype="float32") = lv315[0]
            lv317: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.silu(lv316)
            lv318: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv317, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv319: R.Tensor((1, 48, 1, 1), dtype="float32") = R.nn.conv2d(lv318, metadata["relax.expr.Constant"][276], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv320: R.Tensor((1, 48, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][277], R.shape([1, 48, 1, 1]))
            lv321: R.Tensor((1, 48, 1, 1), dtype="float32") = R.add(lv319, lv320)
            lv322: R.Tensor((1, 48, 1, 1), dtype="float32") = R.nn.silu(lv321)
            lv323: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.nn.conv2d(lv322, metadata["relax.expr.Constant"][278], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv324: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][279], R.shape([1, 1152, 1, 1]))
            lv325: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.add(lv323, lv324)
            lv326: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.sigmoid(lv325)
            lv327: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.multiply(lv326, lv317)
            lv328: R.Tensor((1, 192, 7, 7), dtype="float32") = R.nn.conv2d(lv327, metadata["relax.expr.Constant"][280], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv329: R.Tuple(R.Tensor((1, 192, 7, 7), dtype="float32"), R.Tensor((192,), dtype="float32"), R.Tensor((192,), dtype="float32")) = R.nn.batch_norm(lv328, metadata["relax.expr.Constant"][281], metadata["relax.expr.Constant"][282], metadata["relax.expr.Constant"][283], metadata["relax.expr.Constant"][284], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv330: R.Tensor((1, 192, 7, 7), dtype="float32") = lv329[0]
            lv331: R.Tensor((1, 192, 7, 7), dtype="float32") = lv330
            lv332: R.Tensor((1, 192, 7, 7), dtype="float32") = R.add(lv331, lv309)
            lv333: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.conv2d(lv332, metadata["relax.expr.Constant"][285], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv334: R.Tuple(R.Tensor((1, 1152, 7, 7), dtype="float32"), R.Tensor((1152,), dtype="float32"), R.Tensor((1152,), dtype="float32")) = R.nn.batch_norm(lv333, metadata["relax.expr.Constant"][286], metadata["relax.expr.Constant"][287], metadata["relax.expr.Constant"][288], metadata["relax.expr.Constant"][289], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv335: R.Tensor((1, 1152, 7, 7), dtype="float32") = lv334[0]
            lv336: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.silu(lv335)
            lv337: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.conv2d(lv336, metadata["relax.expr.Constant"][290], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1152, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv338: R.Tuple(R.Tensor((1, 1152, 7, 7), dtype="float32"), R.Tensor((1152,), dtype="float32"), R.Tensor((1152,), dtype="float32")) = R.nn.batch_norm(lv337, metadata["relax.expr.Constant"][291], metadata["relax.expr.Constant"][292], metadata["relax.expr.Constant"][293], metadata["relax.expr.Constant"][294], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv339: R.Tensor((1, 1152, 7, 7), dtype="float32") = lv338[0]
            lv340: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.nn.silu(lv339)
            lv341: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv340, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv342: R.Tensor((1, 48, 1, 1), dtype="float32") = R.nn.conv2d(lv341, metadata["relax.expr.Constant"][295], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv343: R.Tensor((1, 48, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][296], R.shape([1, 48, 1, 1]))
            lv344: R.Tensor((1, 48, 1, 1), dtype="float32") = R.add(lv342, lv343)
            lv345: R.Tensor((1, 48, 1, 1), dtype="float32") = R.nn.silu(lv344)
            lv346: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.nn.conv2d(lv345, metadata["relax.expr.Constant"][297], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv347: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][298], R.shape([1, 1152, 1, 1]))
            lv348: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.add(lv346, lv347)
            lv349: R.Tensor((1, 1152, 1, 1), dtype="float32") = R.sigmoid(lv348)
            lv350: R.Tensor((1, 1152, 7, 7), dtype="float32") = R.multiply(lv349, lv340)
            lv351: R.Tensor((1, 320, 7, 7), dtype="float32") = R.nn.conv2d(lv350, metadata["relax.expr.Constant"][299], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv352: R.Tuple(R.Tensor((1, 320, 7, 7), dtype="float32"), R.Tensor((320,), dtype="float32"), R.Tensor((320,), dtype="float32")) = R.nn.batch_norm(lv351, metadata["relax.expr.Constant"][300], metadata["relax.expr.Constant"][301], metadata["relax.expr.Constant"][302], metadata["relax.expr.Constant"][303], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv353: R.Tensor((1, 320, 7, 7), dtype="float32") = lv352[0]
            lv354: R.Tensor((1, 1280, 7, 7), dtype="float32") = R.nn.conv2d(lv353, metadata["relax.expr.Constant"][304], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv355: R.Tuple(R.Tensor((1, 1280, 7, 7), dtype="float32"), R.Tensor((1280,), dtype="float32"), R.Tensor((1280,), dtype="float32")) = R.nn.batch_norm(lv354, metadata["relax.expr.Constant"][305], metadata["relax.expr.Constant"][306], metadata["relax.expr.Constant"][307], metadata["relax.expr.Constant"][308], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            lv356: R.Tensor((1, 1280, 7, 7), dtype="float32") = lv355[0]
            lv357: R.Tensor((1, 1280, 7, 7), dtype="float32") = R.nn.silu(lv356)
            lv358: R.Tensor((1, 1280, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv357, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv359: R.Tensor((1, 1280), dtype="float32") = R.reshape(lv358, R.shape([1, 1280]))
            lv360: R.Tensor((1280, 1000), dtype="float32") = R.permute_dims(metadata["relax.expr.Constant"][309], axes=None)
            lv361: R.Tensor((1, 1000), dtype="float32") = R.matmul(lv359, lv360, out_dtype="float32")
            lv362: R.Tensor((1, 1000), dtype="float32") = R.add(lv361, metadata["relax.expr.Constant"][310])
            gv: R.Tensor((1, 1000), dtype="float32") = lv362
            R.output(gv)
        return gv
```
