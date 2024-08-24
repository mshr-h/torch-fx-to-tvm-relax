# MobileNet V4

- [timm/mobilenetv4_conv_small.e1200_r224_in1k · Hugging Face](https://huggingface.co/timm/mobilenetv4_conv_small.e1200_r224_in1k)

## Testing

```sh
```

## FX Graph

```
graph():
    %x : torch.Tensor [num_users=1] = placeholder[target=x]
    %conv_stem : [num_users=3] = call_module[target=conv_stem](args = (%x,), kwargs = {})
    %getattr_1 : [num_users=1] = call_function[target=builtins.getattr](args = (%conv_stem, ndim), kwargs = {})
    %eq : [num_users=1] = call_function[target=operator.eq](args = (%getattr_1, 4), kwargs = {})
    %getattr_2 : [num_users=0] = call_function[target=builtins.getattr](args = (%conv_stem, ndim), kwargs = {})
    %_assert : [num_users=0] = call_function[target=torch._assert](args = (%eq, expected 4D input (got Proxy(getattr_2)D input)), kwargs = {})
    %bn1_weight : [num_users=1] = get_attr[target=bn1.weight]
    %bn1_bias : [num_users=1] = get_attr[target=bn1.bias]
    %bn1_running_mean : [num_users=1] = get_attr[target=bn1.running_mean]
    %bn1_running_var : [num_users=1] = get_attr[target=bn1.running_var]
    %batch_norm : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%conv_stem, %bn1_running_mean, %bn1_running_var), kwargs = {weight: %bn1_weight, bias: %bn1_bias, training: False, momentum: 0.1, eps: 1e-05})
    %bn1_drop : [num_users=1] = call_module[target=bn1.drop](args = (%batch_norm,), kwargs = {})
    %bn1_act : [num_users=1] = call_module[target=bn1.act](args = (%bn1_drop,), kwargs = {})
    %blocks_0_0_conv : [num_users=3] = call_module[target=blocks.0.0.conv](args = (%bn1_act,), kwargs = {})
    %getattr_3 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_0_0_conv, ndim), kwargs = {})
    %eq_1 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_3, 4), kwargs = {})
    %getattr_4 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_0_0_conv, ndim), kwargs = {})
    %_assert_1 : [num_users=0] = call_function[target=torch._assert](args = (%eq_1, expected 4D input (got Proxy(getattr_4)D input)), kwargs = {})
    %blocks_0_0_bn1_weight : [num_users=1] = get_attr[target=blocks.0.0.bn1.weight]
    %blocks_0_0_bn1_bias : [num_users=1] = get_attr[target=blocks.0.0.bn1.bias]
    %blocks_0_0_bn1_running_mean : [num_users=1] = get_attr[target=blocks.0.0.bn1.running_mean]
    %blocks_0_0_bn1_running_var : [num_users=1] = get_attr[target=blocks.0.0.bn1.running_var]
    %batch_norm_1 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_0_0_conv, %blocks_0_0_bn1_running_mean, %blocks_0_0_bn1_running_var), kwargs = {weight: %blocks_0_0_bn1_weight, bias: %blocks_0_0_bn1_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_0_0_bn1_drop : [num_users=1] = call_module[target=blocks.0.0.bn1.drop](args = (%batch_norm_1,), kwargs = {})
    %blocks_0_0_bn1_act : [num_users=1] = call_module[target=blocks.0.0.bn1.act](args = (%blocks_0_0_bn1_drop,), kwargs = {})
    %blocks_0_0_aa : [num_users=1] = call_module[target=blocks.0.0.aa](args = (%blocks_0_0_bn1_act,), kwargs = {})
    %blocks_0_1_conv : [num_users=3] = call_module[target=blocks.0.1.conv](args = (%blocks_0_0_aa,), kwargs = {})
    %getattr_5 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_0_1_conv, ndim), kwargs = {})
    %eq_2 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_5, 4), kwargs = {})
    %getattr_6 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_0_1_conv, ndim), kwargs = {})
    %_assert_2 : [num_users=0] = call_function[target=torch._assert](args = (%eq_2, expected 4D input (got Proxy(getattr_6)D input)), kwargs = {})
    %blocks_0_1_bn1_weight : [num_users=1] = get_attr[target=blocks.0.1.bn1.weight]
    %blocks_0_1_bn1_bias : [num_users=1] = get_attr[target=blocks.0.1.bn1.bias]
    %blocks_0_1_bn1_running_mean : [num_users=1] = get_attr[target=blocks.0.1.bn1.running_mean]
    %blocks_0_1_bn1_running_var : [num_users=1] = get_attr[target=blocks.0.1.bn1.running_var]
    %batch_norm_2 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_0_1_conv, %blocks_0_1_bn1_running_mean, %blocks_0_1_bn1_running_var), kwargs = {weight: %blocks_0_1_bn1_weight, bias: %blocks_0_1_bn1_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_0_1_bn1_drop : [num_users=1] = call_module[target=blocks.0.1.bn1.drop](args = (%batch_norm_2,), kwargs = {})
    %blocks_0_1_bn1_act : [num_users=1] = call_module[target=blocks.0.1.bn1.act](args = (%blocks_0_1_bn1_drop,), kwargs = {})
    %blocks_0_1_aa : [num_users=1] = call_module[target=blocks.0.1.aa](args = (%blocks_0_1_bn1_act,), kwargs = {})
    %blocks_1_0_conv : [num_users=3] = call_module[target=blocks.1.0.conv](args = (%blocks_0_1_aa,), kwargs = {})
    %getattr_7 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_1_0_conv, ndim), kwargs = {})
    %eq_3 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_7, 4), kwargs = {})
    %getattr_8 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_1_0_conv, ndim), kwargs = {})
    %_assert_3 : [num_users=0] = call_function[target=torch._assert](args = (%eq_3, expected 4D input (got Proxy(getattr_8)D input)), kwargs = {})
    %blocks_1_0_bn1_weight : [num_users=1] = get_attr[target=blocks.1.0.bn1.weight]
    %blocks_1_0_bn1_bias : [num_users=1] = get_attr[target=blocks.1.0.bn1.bias]
    %blocks_1_0_bn1_running_mean : [num_users=1] = get_attr[target=blocks.1.0.bn1.running_mean]
    %blocks_1_0_bn1_running_var : [num_users=1] = get_attr[target=blocks.1.0.bn1.running_var]
    %batch_norm_3 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_1_0_conv, %blocks_1_0_bn1_running_mean, %blocks_1_0_bn1_running_var), kwargs = {weight: %blocks_1_0_bn1_weight, bias: %blocks_1_0_bn1_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_1_0_bn1_drop : [num_users=1] = call_module[target=blocks.1.0.bn1.drop](args = (%batch_norm_3,), kwargs = {})
    %blocks_1_0_bn1_act : [num_users=1] = call_module[target=blocks.1.0.bn1.act](args = (%blocks_1_0_bn1_drop,), kwargs = {})
    %blocks_1_0_aa : [num_users=1] = call_module[target=blocks.1.0.aa](args = (%blocks_1_0_bn1_act,), kwargs = {})
    %blocks_1_1_conv : [num_users=3] = call_module[target=blocks.1.1.conv](args = (%blocks_1_0_aa,), kwargs = {})
    %getattr_9 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_1_1_conv, ndim), kwargs = {})
    %eq_4 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_9, 4), kwargs = {})
    %getattr_10 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_1_1_conv, ndim), kwargs = {})
    %_assert_4 : [num_users=0] = call_function[target=torch._assert](args = (%eq_4, expected 4D input (got Proxy(getattr_10)D input)), kwargs = {})
    %blocks_1_1_bn1_weight : [num_users=1] = get_attr[target=blocks.1.1.bn1.weight]
    %blocks_1_1_bn1_bias : [num_users=1] = get_attr[target=blocks.1.1.bn1.bias]
    %blocks_1_1_bn1_running_mean : [num_users=1] = get_attr[target=blocks.1.1.bn1.running_mean]
    %blocks_1_1_bn1_running_var : [num_users=1] = get_attr[target=blocks.1.1.bn1.running_var]
    %batch_norm_4 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_1_1_conv, %blocks_1_1_bn1_running_mean, %blocks_1_1_bn1_running_var), kwargs = {weight: %blocks_1_1_bn1_weight, bias: %blocks_1_1_bn1_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_1_1_bn1_drop : [num_users=1] = call_module[target=blocks.1.1.bn1.drop](args = (%batch_norm_4,), kwargs = {})
    %blocks_1_1_bn1_act : [num_users=1] = call_module[target=blocks.1.1.bn1.act](args = (%blocks_1_1_bn1_drop,), kwargs = {})
    %blocks_1_1_aa : [num_users=1] = call_module[target=blocks.1.1.aa](args = (%blocks_1_1_bn1_act,), kwargs = {})
    %blocks_2_0_dw_start_conv : [num_users=3] = call_module[target=blocks.2.0.dw_start.conv](args = (%blocks_1_1_aa,), kwargs = {})
    %getattr_11 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_0_dw_start_conv, ndim), kwargs = {})
    %eq_5 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_11, 4), kwargs = {})
    %getattr_12 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_0_dw_start_conv, ndim), kwargs = {})
    %_assert_5 : [num_users=0] = call_function[target=torch._assert](args = (%eq_5, expected 4D input (got Proxy(getattr_12)D input)), kwargs = {})
    %blocks_2_0_dw_start_bn_weight : [num_users=1] = get_attr[target=blocks.2.0.dw_start.bn.weight]
    %blocks_2_0_dw_start_bn_bias : [num_users=1] = get_attr[target=blocks.2.0.dw_start.bn.bias]
    %blocks_2_0_dw_start_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.0.dw_start.bn.running_mean]
    %blocks_2_0_dw_start_bn_running_var : [num_users=1] = get_attr[target=blocks.2.0.dw_start.bn.running_var]
    %batch_norm_5 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_0_dw_start_conv, %blocks_2_0_dw_start_bn_running_mean, %blocks_2_0_dw_start_bn_running_var), kwargs = {weight: %blocks_2_0_dw_start_bn_weight, bias: %blocks_2_0_dw_start_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_0_dw_start_bn_drop : [num_users=1] = call_module[target=blocks.2.0.dw_start.bn.drop](args = (%batch_norm_5,), kwargs = {})
    %blocks_2_0_dw_start_bn_act : [num_users=1] = call_module[target=blocks.2.0.dw_start.bn.act](args = (%blocks_2_0_dw_start_bn_drop,), kwargs = {})
    %blocks_2_0_pw_exp_conv : [num_users=3] = call_module[target=blocks.2.0.pw_exp.conv](args = (%blocks_2_0_dw_start_bn_act,), kwargs = {})
    %getattr_13 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_0_pw_exp_conv, ndim), kwargs = {})
    %eq_6 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_13, 4), kwargs = {})
    %getattr_14 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_0_pw_exp_conv, ndim), kwargs = {})
    %_assert_6 : [num_users=0] = call_function[target=torch._assert](args = (%eq_6, expected 4D input (got Proxy(getattr_14)D input)), kwargs = {})
    %blocks_2_0_pw_exp_bn_weight : [num_users=1] = get_attr[target=blocks.2.0.pw_exp.bn.weight]
    %blocks_2_0_pw_exp_bn_bias : [num_users=1] = get_attr[target=blocks.2.0.pw_exp.bn.bias]
    %blocks_2_0_pw_exp_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.0.pw_exp.bn.running_mean]
    %blocks_2_0_pw_exp_bn_running_var : [num_users=1] = get_attr[target=blocks.2.0.pw_exp.bn.running_var]
    %batch_norm_6 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_0_pw_exp_conv, %blocks_2_0_pw_exp_bn_running_mean, %blocks_2_0_pw_exp_bn_running_var), kwargs = {weight: %blocks_2_0_pw_exp_bn_weight, bias: %blocks_2_0_pw_exp_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_0_pw_exp_bn_drop : [num_users=1] = call_module[target=blocks.2.0.pw_exp.bn.drop](args = (%batch_norm_6,), kwargs = {})
    %blocks_2_0_pw_exp_bn_act : [num_users=1] = call_module[target=blocks.2.0.pw_exp.bn.act](args = (%blocks_2_0_pw_exp_bn_drop,), kwargs = {})
    %blocks_2_0_dw_mid_conv : [num_users=3] = call_module[target=blocks.2.0.dw_mid.conv](args = (%blocks_2_0_pw_exp_bn_act,), kwargs = {})
    %getattr_15 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_0_dw_mid_conv, ndim), kwargs = {})
    %eq_7 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_15, 4), kwargs = {})
    %getattr_16 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_0_dw_mid_conv, ndim), kwargs = {})
    %_assert_7 : [num_users=0] = call_function[target=torch._assert](args = (%eq_7, expected 4D input (got Proxy(getattr_16)D input)), kwargs = {})
    %blocks_2_0_dw_mid_bn_weight : [num_users=1] = get_attr[target=blocks.2.0.dw_mid.bn.weight]
    %blocks_2_0_dw_mid_bn_bias : [num_users=1] = get_attr[target=blocks.2.0.dw_mid.bn.bias]
    %blocks_2_0_dw_mid_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.0.dw_mid.bn.running_mean]
    %blocks_2_0_dw_mid_bn_running_var : [num_users=1] = get_attr[target=blocks.2.0.dw_mid.bn.running_var]
    %batch_norm_7 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_0_dw_mid_conv, %blocks_2_0_dw_mid_bn_running_mean, %blocks_2_0_dw_mid_bn_running_var), kwargs = {weight: %blocks_2_0_dw_mid_bn_weight, bias: %blocks_2_0_dw_mid_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_0_dw_mid_bn_drop : [num_users=1] = call_module[target=blocks.2.0.dw_mid.bn.drop](args = (%batch_norm_7,), kwargs = {})
    %blocks_2_0_dw_mid_bn_act : [num_users=1] = call_module[target=blocks.2.0.dw_mid.bn.act](args = (%blocks_2_0_dw_mid_bn_drop,), kwargs = {})
    %blocks_2_0_se : [num_users=1] = call_module[target=blocks.2.0.se](args = (%blocks_2_0_dw_mid_bn_act,), kwargs = {})
    %blocks_2_0_pw_proj_conv : [num_users=3] = call_module[target=blocks.2.0.pw_proj.conv](args = (%blocks_2_0_se,), kwargs = {})
    %getattr_17 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_0_pw_proj_conv, ndim), kwargs = {})
    %eq_8 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_17, 4), kwargs = {})
    %getattr_18 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_0_pw_proj_conv, ndim), kwargs = {})
    %_assert_8 : [num_users=0] = call_function[target=torch._assert](args = (%eq_8, expected 4D input (got Proxy(getattr_18)D input)), kwargs = {})
    %blocks_2_0_pw_proj_bn_weight : [num_users=1] = get_attr[target=blocks.2.0.pw_proj.bn.weight]
    %blocks_2_0_pw_proj_bn_bias : [num_users=1] = get_attr[target=blocks.2.0.pw_proj.bn.bias]
    %blocks_2_0_pw_proj_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.0.pw_proj.bn.running_mean]
    %blocks_2_0_pw_proj_bn_running_var : [num_users=1] = get_attr[target=blocks.2.0.pw_proj.bn.running_var]
    %batch_norm_8 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_0_pw_proj_conv, %blocks_2_0_pw_proj_bn_running_mean, %blocks_2_0_pw_proj_bn_running_var), kwargs = {weight: %blocks_2_0_pw_proj_bn_weight, bias: %blocks_2_0_pw_proj_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_0_pw_proj_bn_drop : [num_users=1] = call_module[target=blocks.2.0.pw_proj.bn.drop](args = (%batch_norm_8,), kwargs = {})
    %blocks_2_0_pw_proj_bn_act : [num_users=1] = call_module[target=blocks.2.0.pw_proj.bn.act](args = (%blocks_2_0_pw_proj_bn_drop,), kwargs = {})
    %blocks_2_0_dw_end : [num_users=1] = call_module[target=blocks.2.0.dw_end](args = (%blocks_2_0_pw_proj_bn_act,), kwargs = {})
    %blocks_2_0_layer_scale : [num_users=2] = call_module[target=blocks.2.0.layer_scale](args = (%blocks_2_0_dw_end,), kwargs = {})
    %blocks_2_1_dw_start : [num_users=1] = call_module[target=blocks.2.1.dw_start](args = (%blocks_2_0_layer_scale,), kwargs = {})
    %blocks_2_1_pw_exp_conv : [num_users=3] = call_module[target=blocks.2.1.pw_exp.conv](args = (%blocks_2_1_dw_start,), kwargs = {})
    %getattr_19 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_1_pw_exp_conv, ndim), kwargs = {})
    %eq_9 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_19, 4), kwargs = {})
    %getattr_20 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_1_pw_exp_conv, ndim), kwargs = {})
    %_assert_9 : [num_users=0] = call_function[target=torch._assert](args = (%eq_9, expected 4D input (got Proxy(getattr_20)D input)), kwargs = {})
    %blocks_2_1_pw_exp_bn_weight : [num_users=1] = get_attr[target=blocks.2.1.pw_exp.bn.weight]
    %blocks_2_1_pw_exp_bn_bias : [num_users=1] = get_attr[target=blocks.2.1.pw_exp.bn.bias]
    %blocks_2_1_pw_exp_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.1.pw_exp.bn.running_mean]
    %blocks_2_1_pw_exp_bn_running_var : [num_users=1] = get_attr[target=blocks.2.1.pw_exp.bn.running_var]
    %batch_norm_9 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_1_pw_exp_conv, %blocks_2_1_pw_exp_bn_running_mean, %blocks_2_1_pw_exp_bn_running_var), kwargs = {weight: %blocks_2_1_pw_exp_bn_weight, bias: %blocks_2_1_pw_exp_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_1_pw_exp_bn_drop : [num_users=1] = call_module[target=blocks.2.1.pw_exp.bn.drop](args = (%batch_norm_9,), kwargs = {})
    %blocks_2_1_pw_exp_bn_act : [num_users=1] = call_module[target=blocks.2.1.pw_exp.bn.act](args = (%blocks_2_1_pw_exp_bn_drop,), kwargs = {})
    %blocks_2_1_dw_mid_conv : [num_users=3] = call_module[target=blocks.2.1.dw_mid.conv](args = (%blocks_2_1_pw_exp_bn_act,), kwargs = {})
    %getattr_21 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_1_dw_mid_conv, ndim), kwargs = {})
    %eq_10 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_21, 4), kwargs = {})
    %getattr_22 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_1_dw_mid_conv, ndim), kwargs = {})
    %_assert_10 : [num_users=0] = call_function[target=torch._assert](args = (%eq_10, expected 4D input (got Proxy(getattr_22)D input)), kwargs = {})
    %blocks_2_1_dw_mid_bn_weight : [num_users=1] = get_attr[target=blocks.2.1.dw_mid.bn.weight]
    %blocks_2_1_dw_mid_bn_bias : [num_users=1] = get_attr[target=blocks.2.1.dw_mid.bn.bias]
    %blocks_2_1_dw_mid_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.1.dw_mid.bn.running_mean]
    %blocks_2_1_dw_mid_bn_running_var : [num_users=1] = get_attr[target=blocks.2.1.dw_mid.bn.running_var]
    %batch_norm_10 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_1_dw_mid_conv, %blocks_2_1_dw_mid_bn_running_mean, %blocks_2_1_dw_mid_bn_running_var), kwargs = {weight: %blocks_2_1_dw_mid_bn_weight, bias: %blocks_2_1_dw_mid_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_1_dw_mid_bn_drop : [num_users=1] = call_module[target=blocks.2.1.dw_mid.bn.drop](args = (%batch_norm_10,), kwargs = {})
    %blocks_2_1_dw_mid_bn_act : [num_users=1] = call_module[target=blocks.2.1.dw_mid.bn.act](args = (%blocks_2_1_dw_mid_bn_drop,), kwargs = {})
    %blocks_2_1_se : [num_users=1] = call_module[target=blocks.2.1.se](args = (%blocks_2_1_dw_mid_bn_act,), kwargs = {})
    %blocks_2_1_pw_proj_conv : [num_users=3] = call_module[target=blocks.2.1.pw_proj.conv](args = (%blocks_2_1_se,), kwargs = {})
    %getattr_23 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_1_pw_proj_conv, ndim), kwargs = {})
    %eq_11 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_23, 4), kwargs = {})
    %getattr_24 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_1_pw_proj_conv, ndim), kwargs = {})
    %_assert_11 : [num_users=0] = call_function[target=torch._assert](args = (%eq_11, expected 4D input (got Proxy(getattr_24)D input)), kwargs = {})
    %blocks_2_1_pw_proj_bn_weight : [num_users=1] = get_attr[target=blocks.2.1.pw_proj.bn.weight]
    %blocks_2_1_pw_proj_bn_bias : [num_users=1] = get_attr[target=blocks.2.1.pw_proj.bn.bias]
    %blocks_2_1_pw_proj_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.1.pw_proj.bn.running_mean]
    %blocks_2_1_pw_proj_bn_running_var : [num_users=1] = get_attr[target=blocks.2.1.pw_proj.bn.running_var]
    %batch_norm_11 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_1_pw_proj_conv, %blocks_2_1_pw_proj_bn_running_mean, %blocks_2_1_pw_proj_bn_running_var), kwargs = {weight: %blocks_2_1_pw_proj_bn_weight, bias: %blocks_2_1_pw_proj_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_1_pw_proj_bn_drop : [num_users=1] = call_module[target=blocks.2.1.pw_proj.bn.drop](args = (%batch_norm_11,), kwargs = {})
    %blocks_2_1_pw_proj_bn_act : [num_users=1] = call_module[target=blocks.2.1.pw_proj.bn.act](args = (%blocks_2_1_pw_proj_bn_drop,), kwargs = {})
    %blocks_2_1_dw_end : [num_users=1] = call_module[target=blocks.2.1.dw_end](args = (%blocks_2_1_pw_proj_bn_act,), kwargs = {})
    %blocks_2_1_layer_scale : [num_users=1] = call_module[target=blocks.2.1.layer_scale](args = (%blocks_2_1_dw_end,), kwargs = {})
    %blocks_2_1_drop_path : [num_users=1] = call_module[target=blocks.2.1.drop_path](args = (%blocks_2_1_layer_scale,), kwargs = {})
    %add : [num_users=2] = call_function[target=operator.add](args = (%blocks_2_1_drop_path, %blocks_2_0_layer_scale), kwargs = {})
    %blocks_2_2_dw_start : [num_users=1] = call_module[target=blocks.2.2.dw_start](args = (%add,), kwargs = {})
    %blocks_2_2_pw_exp_conv : [num_users=3] = call_module[target=blocks.2.2.pw_exp.conv](args = (%blocks_2_2_dw_start,), kwargs = {})
    %getattr_25 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_2_pw_exp_conv, ndim), kwargs = {})
    %eq_12 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_25, 4), kwargs = {})
    %getattr_26 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_2_pw_exp_conv, ndim), kwargs = {})
    %_assert_12 : [num_users=0] = call_function[target=torch._assert](args = (%eq_12, expected 4D input (got Proxy(getattr_26)D input)), kwargs = {})
    %blocks_2_2_pw_exp_bn_weight : [num_users=1] = get_attr[target=blocks.2.2.pw_exp.bn.weight]
    %blocks_2_2_pw_exp_bn_bias : [num_users=1] = get_attr[target=blocks.2.2.pw_exp.bn.bias]
    %blocks_2_2_pw_exp_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.2.pw_exp.bn.running_mean]
    %blocks_2_2_pw_exp_bn_running_var : [num_users=1] = get_attr[target=blocks.2.2.pw_exp.bn.running_var]
    %batch_norm_12 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_2_pw_exp_conv, %blocks_2_2_pw_exp_bn_running_mean, %blocks_2_2_pw_exp_bn_running_var), kwargs = {weight: %blocks_2_2_pw_exp_bn_weight, bias: %blocks_2_2_pw_exp_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_2_pw_exp_bn_drop : [num_users=1] = call_module[target=blocks.2.2.pw_exp.bn.drop](args = (%batch_norm_12,), kwargs = {})
    %blocks_2_2_pw_exp_bn_act : [num_users=1] = call_module[target=blocks.2.2.pw_exp.bn.act](args = (%blocks_2_2_pw_exp_bn_drop,), kwargs = {})
    %blocks_2_2_dw_mid_conv : [num_users=3] = call_module[target=blocks.2.2.dw_mid.conv](args = (%blocks_2_2_pw_exp_bn_act,), kwargs = {})
    %getattr_27 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_2_dw_mid_conv, ndim), kwargs = {})
    %eq_13 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_27, 4), kwargs = {})
    %getattr_28 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_2_dw_mid_conv, ndim), kwargs = {})
    %_assert_13 : [num_users=0] = call_function[target=torch._assert](args = (%eq_13, expected 4D input (got Proxy(getattr_28)D input)), kwargs = {})
    %blocks_2_2_dw_mid_bn_weight : [num_users=1] = get_attr[target=blocks.2.2.dw_mid.bn.weight]
    %blocks_2_2_dw_mid_bn_bias : [num_users=1] = get_attr[target=blocks.2.2.dw_mid.bn.bias]
    %blocks_2_2_dw_mid_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.2.dw_mid.bn.running_mean]
    %blocks_2_2_dw_mid_bn_running_var : [num_users=1] = get_attr[target=blocks.2.2.dw_mid.bn.running_var]
    %batch_norm_13 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_2_dw_mid_conv, %blocks_2_2_dw_mid_bn_running_mean, %blocks_2_2_dw_mid_bn_running_var), kwargs = {weight: %blocks_2_2_dw_mid_bn_weight, bias: %blocks_2_2_dw_mid_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_2_dw_mid_bn_drop : [num_users=1] = call_module[target=blocks.2.2.dw_mid.bn.drop](args = (%batch_norm_13,), kwargs = {})
    %blocks_2_2_dw_mid_bn_act : [num_users=1] = call_module[target=blocks.2.2.dw_mid.bn.act](args = (%blocks_2_2_dw_mid_bn_drop,), kwargs = {})
    %blocks_2_2_se : [num_users=1] = call_module[target=blocks.2.2.se](args = (%blocks_2_2_dw_mid_bn_act,), kwargs = {})
    %blocks_2_2_pw_proj_conv : [num_users=3] = call_module[target=blocks.2.2.pw_proj.conv](args = (%blocks_2_2_se,), kwargs = {})
    %getattr_29 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_2_pw_proj_conv, ndim), kwargs = {})
    %eq_14 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_29, 4), kwargs = {})
    %getattr_30 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_2_pw_proj_conv, ndim), kwargs = {})
    %_assert_14 : [num_users=0] = call_function[target=torch._assert](args = (%eq_14, expected 4D input (got Proxy(getattr_30)D input)), kwargs = {})
    %blocks_2_2_pw_proj_bn_weight : [num_users=1] = get_attr[target=blocks.2.2.pw_proj.bn.weight]
    %blocks_2_2_pw_proj_bn_bias : [num_users=1] = get_attr[target=blocks.2.2.pw_proj.bn.bias]
    %blocks_2_2_pw_proj_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.2.pw_proj.bn.running_mean]
    %blocks_2_2_pw_proj_bn_running_var : [num_users=1] = get_attr[target=blocks.2.2.pw_proj.bn.running_var]
    %batch_norm_14 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_2_pw_proj_conv, %blocks_2_2_pw_proj_bn_running_mean, %blocks_2_2_pw_proj_bn_running_var), kwargs = {weight: %blocks_2_2_pw_proj_bn_weight, bias: %blocks_2_2_pw_proj_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_2_pw_proj_bn_drop : [num_users=1] = call_module[target=blocks.2.2.pw_proj.bn.drop](args = (%batch_norm_14,), kwargs = {})
    %blocks_2_2_pw_proj_bn_act : [num_users=1] = call_module[target=blocks.2.2.pw_proj.bn.act](args = (%blocks_2_2_pw_proj_bn_drop,), kwargs = {})
    %blocks_2_2_dw_end : [num_users=1] = call_module[target=blocks.2.2.dw_end](args = (%blocks_2_2_pw_proj_bn_act,), kwargs = {})
    %blocks_2_2_layer_scale : [num_users=1] = call_module[target=blocks.2.2.layer_scale](args = (%blocks_2_2_dw_end,), kwargs = {})
    %blocks_2_2_drop_path : [num_users=1] = call_module[target=blocks.2.2.drop_path](args = (%blocks_2_2_layer_scale,), kwargs = {})
    %add_1 : [num_users=2] = call_function[target=operator.add](args = (%blocks_2_2_drop_path, %add), kwargs = {})
    %blocks_2_3_dw_start : [num_users=1] = call_module[target=blocks.2.3.dw_start](args = (%add_1,), kwargs = {})
    %blocks_2_3_pw_exp_conv : [num_users=3] = call_module[target=blocks.2.3.pw_exp.conv](args = (%blocks_2_3_dw_start,), kwargs = {})
    %getattr_31 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_3_pw_exp_conv, ndim), kwargs = {})
    %eq_15 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_31, 4), kwargs = {})
    %getattr_32 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_3_pw_exp_conv, ndim), kwargs = {})
    %_assert_15 : [num_users=0] = call_function[target=torch._assert](args = (%eq_15, expected 4D input (got Proxy(getattr_32)D input)), kwargs = {})
    %blocks_2_3_pw_exp_bn_weight : [num_users=1] = get_attr[target=blocks.2.3.pw_exp.bn.weight]
    %blocks_2_3_pw_exp_bn_bias : [num_users=1] = get_attr[target=blocks.2.3.pw_exp.bn.bias]
    %blocks_2_3_pw_exp_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.3.pw_exp.bn.running_mean]
    %blocks_2_3_pw_exp_bn_running_var : [num_users=1] = get_attr[target=blocks.2.3.pw_exp.bn.running_var]
    %batch_norm_15 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_3_pw_exp_conv, %blocks_2_3_pw_exp_bn_running_mean, %blocks_2_3_pw_exp_bn_running_var), kwargs = {weight: %blocks_2_3_pw_exp_bn_weight, bias: %blocks_2_3_pw_exp_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_3_pw_exp_bn_drop : [num_users=1] = call_module[target=blocks.2.3.pw_exp.bn.drop](args = (%batch_norm_15,), kwargs = {})
    %blocks_2_3_pw_exp_bn_act : [num_users=1] = call_module[target=blocks.2.3.pw_exp.bn.act](args = (%blocks_2_3_pw_exp_bn_drop,), kwargs = {})
    %blocks_2_3_dw_mid_conv : [num_users=3] = call_module[target=blocks.2.3.dw_mid.conv](args = (%blocks_2_3_pw_exp_bn_act,), kwargs = {})
    %getattr_33 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_3_dw_mid_conv, ndim), kwargs = {})
    %eq_16 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_33, 4), kwargs = {})
    %getattr_34 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_3_dw_mid_conv, ndim), kwargs = {})
    %_assert_16 : [num_users=0] = call_function[target=torch._assert](args = (%eq_16, expected 4D input (got Proxy(getattr_34)D input)), kwargs = {})
    %blocks_2_3_dw_mid_bn_weight : [num_users=1] = get_attr[target=blocks.2.3.dw_mid.bn.weight]
    %blocks_2_3_dw_mid_bn_bias : [num_users=1] = get_attr[target=blocks.2.3.dw_mid.bn.bias]
    %blocks_2_3_dw_mid_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.3.dw_mid.bn.running_mean]
    %blocks_2_3_dw_mid_bn_running_var : [num_users=1] = get_attr[target=blocks.2.3.dw_mid.bn.running_var]
    %batch_norm_16 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_3_dw_mid_conv, %blocks_2_3_dw_mid_bn_running_mean, %blocks_2_3_dw_mid_bn_running_var), kwargs = {weight: %blocks_2_3_dw_mid_bn_weight, bias: %blocks_2_3_dw_mid_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_3_dw_mid_bn_drop : [num_users=1] = call_module[target=blocks.2.3.dw_mid.bn.drop](args = (%batch_norm_16,), kwargs = {})
    %blocks_2_3_dw_mid_bn_act : [num_users=1] = call_module[target=blocks.2.3.dw_mid.bn.act](args = (%blocks_2_3_dw_mid_bn_drop,), kwargs = {})
    %blocks_2_3_se : [num_users=1] = call_module[target=blocks.2.3.se](args = (%blocks_2_3_dw_mid_bn_act,), kwargs = {})
    %blocks_2_3_pw_proj_conv : [num_users=3] = call_module[target=blocks.2.3.pw_proj.conv](args = (%blocks_2_3_se,), kwargs = {})
    %getattr_35 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_3_pw_proj_conv, ndim), kwargs = {})
    %eq_17 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_35, 4), kwargs = {})
    %getattr_36 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_3_pw_proj_conv, ndim), kwargs = {})
    %_assert_17 : [num_users=0] = call_function[target=torch._assert](args = (%eq_17, expected 4D input (got Proxy(getattr_36)D input)), kwargs = {})
    %blocks_2_3_pw_proj_bn_weight : [num_users=1] = get_attr[target=blocks.2.3.pw_proj.bn.weight]
    %blocks_2_3_pw_proj_bn_bias : [num_users=1] = get_attr[target=blocks.2.3.pw_proj.bn.bias]
    %blocks_2_3_pw_proj_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.3.pw_proj.bn.running_mean]
    %blocks_2_3_pw_proj_bn_running_var : [num_users=1] = get_attr[target=blocks.2.3.pw_proj.bn.running_var]
    %batch_norm_17 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_3_pw_proj_conv, %blocks_2_3_pw_proj_bn_running_mean, %blocks_2_3_pw_proj_bn_running_var), kwargs = {weight: %blocks_2_3_pw_proj_bn_weight, bias: %blocks_2_3_pw_proj_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_3_pw_proj_bn_drop : [num_users=1] = call_module[target=blocks.2.3.pw_proj.bn.drop](args = (%batch_norm_17,), kwargs = {})
    %blocks_2_3_pw_proj_bn_act : [num_users=1] = call_module[target=blocks.2.3.pw_proj.bn.act](args = (%blocks_2_3_pw_proj_bn_drop,), kwargs = {})
    %blocks_2_3_dw_end : [num_users=1] = call_module[target=blocks.2.3.dw_end](args = (%blocks_2_3_pw_proj_bn_act,), kwargs = {})
    %blocks_2_3_layer_scale : [num_users=1] = call_module[target=blocks.2.3.layer_scale](args = (%blocks_2_3_dw_end,), kwargs = {})
    %blocks_2_3_drop_path : [num_users=1] = call_module[target=blocks.2.3.drop_path](args = (%blocks_2_3_layer_scale,), kwargs = {})
    %add_2 : [num_users=2] = call_function[target=operator.add](args = (%blocks_2_3_drop_path, %add_1), kwargs = {})
    %blocks_2_4_dw_start : [num_users=1] = call_module[target=blocks.2.4.dw_start](args = (%add_2,), kwargs = {})
    %blocks_2_4_pw_exp_conv : [num_users=3] = call_module[target=blocks.2.4.pw_exp.conv](args = (%blocks_2_4_dw_start,), kwargs = {})
    %getattr_37 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_4_pw_exp_conv, ndim), kwargs = {})
    %eq_18 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_37, 4), kwargs = {})
    %getattr_38 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_4_pw_exp_conv, ndim), kwargs = {})
    %_assert_18 : [num_users=0] = call_function[target=torch._assert](args = (%eq_18, expected 4D input (got Proxy(getattr_38)D input)), kwargs = {})
    %blocks_2_4_pw_exp_bn_weight : [num_users=1] = get_attr[target=blocks.2.4.pw_exp.bn.weight]
    %blocks_2_4_pw_exp_bn_bias : [num_users=1] = get_attr[target=blocks.2.4.pw_exp.bn.bias]
    %blocks_2_4_pw_exp_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.4.pw_exp.bn.running_mean]
    %blocks_2_4_pw_exp_bn_running_var : [num_users=1] = get_attr[target=blocks.2.4.pw_exp.bn.running_var]
    %batch_norm_18 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_4_pw_exp_conv, %blocks_2_4_pw_exp_bn_running_mean, %blocks_2_4_pw_exp_bn_running_var), kwargs = {weight: %blocks_2_4_pw_exp_bn_weight, bias: %blocks_2_4_pw_exp_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_4_pw_exp_bn_drop : [num_users=1] = call_module[target=blocks.2.4.pw_exp.bn.drop](args = (%batch_norm_18,), kwargs = {})
    %blocks_2_4_pw_exp_bn_act : [num_users=1] = call_module[target=blocks.2.4.pw_exp.bn.act](args = (%blocks_2_4_pw_exp_bn_drop,), kwargs = {})
    %blocks_2_4_dw_mid_conv : [num_users=3] = call_module[target=blocks.2.4.dw_mid.conv](args = (%blocks_2_4_pw_exp_bn_act,), kwargs = {})
    %getattr_39 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_4_dw_mid_conv, ndim), kwargs = {})
    %eq_19 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_39, 4), kwargs = {})
    %getattr_40 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_4_dw_mid_conv, ndim), kwargs = {})
    %_assert_19 : [num_users=0] = call_function[target=torch._assert](args = (%eq_19, expected 4D input (got Proxy(getattr_40)D input)), kwargs = {})
    %blocks_2_4_dw_mid_bn_weight : [num_users=1] = get_attr[target=blocks.2.4.dw_mid.bn.weight]
    %blocks_2_4_dw_mid_bn_bias : [num_users=1] = get_attr[target=blocks.2.4.dw_mid.bn.bias]
    %blocks_2_4_dw_mid_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.4.dw_mid.bn.running_mean]
    %blocks_2_4_dw_mid_bn_running_var : [num_users=1] = get_attr[target=blocks.2.4.dw_mid.bn.running_var]
    %batch_norm_19 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_4_dw_mid_conv, %blocks_2_4_dw_mid_bn_running_mean, %blocks_2_4_dw_mid_bn_running_var), kwargs = {weight: %blocks_2_4_dw_mid_bn_weight, bias: %blocks_2_4_dw_mid_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_4_dw_mid_bn_drop : [num_users=1] = call_module[target=blocks.2.4.dw_mid.bn.drop](args = (%batch_norm_19,), kwargs = {})
    %blocks_2_4_dw_mid_bn_act : [num_users=1] = call_module[target=blocks.2.4.dw_mid.bn.act](args = (%blocks_2_4_dw_mid_bn_drop,), kwargs = {})
    %blocks_2_4_se : [num_users=1] = call_module[target=blocks.2.4.se](args = (%blocks_2_4_dw_mid_bn_act,), kwargs = {})
    %blocks_2_4_pw_proj_conv : [num_users=3] = call_module[target=blocks.2.4.pw_proj.conv](args = (%blocks_2_4_se,), kwargs = {})
    %getattr_41 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_4_pw_proj_conv, ndim), kwargs = {})
    %eq_20 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_41, 4), kwargs = {})
    %getattr_42 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_4_pw_proj_conv, ndim), kwargs = {})
    %_assert_20 : [num_users=0] = call_function[target=torch._assert](args = (%eq_20, expected 4D input (got Proxy(getattr_42)D input)), kwargs = {})
    %blocks_2_4_pw_proj_bn_weight : [num_users=1] = get_attr[target=blocks.2.4.pw_proj.bn.weight]
    %blocks_2_4_pw_proj_bn_bias : [num_users=1] = get_attr[target=blocks.2.4.pw_proj.bn.bias]
    %blocks_2_4_pw_proj_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.4.pw_proj.bn.running_mean]
    %blocks_2_4_pw_proj_bn_running_var : [num_users=1] = get_attr[target=blocks.2.4.pw_proj.bn.running_var]
    %batch_norm_20 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_4_pw_proj_conv, %blocks_2_4_pw_proj_bn_running_mean, %blocks_2_4_pw_proj_bn_running_var), kwargs = {weight: %blocks_2_4_pw_proj_bn_weight, bias: %blocks_2_4_pw_proj_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_4_pw_proj_bn_drop : [num_users=1] = call_module[target=blocks.2.4.pw_proj.bn.drop](args = (%batch_norm_20,), kwargs = {})
    %blocks_2_4_pw_proj_bn_act : [num_users=1] = call_module[target=blocks.2.4.pw_proj.bn.act](args = (%blocks_2_4_pw_proj_bn_drop,), kwargs = {})
    %blocks_2_4_dw_end : [num_users=1] = call_module[target=blocks.2.4.dw_end](args = (%blocks_2_4_pw_proj_bn_act,), kwargs = {})
    %blocks_2_4_layer_scale : [num_users=1] = call_module[target=blocks.2.4.layer_scale](args = (%blocks_2_4_dw_end,), kwargs = {})
    %blocks_2_4_drop_path : [num_users=1] = call_module[target=blocks.2.4.drop_path](args = (%blocks_2_4_layer_scale,), kwargs = {})
    %add_3 : [num_users=2] = call_function[target=operator.add](args = (%blocks_2_4_drop_path, %add_2), kwargs = {})
    %blocks_2_5_dw_start_conv : [num_users=3] = call_module[target=blocks.2.5.dw_start.conv](args = (%add_3,), kwargs = {})
    %getattr_43 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_5_dw_start_conv, ndim), kwargs = {})
    %eq_21 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_43, 4), kwargs = {})
    %getattr_44 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_5_dw_start_conv, ndim), kwargs = {})
    %_assert_21 : [num_users=0] = call_function[target=torch._assert](args = (%eq_21, expected 4D input (got Proxy(getattr_44)D input)), kwargs = {})
    %blocks_2_5_dw_start_bn_weight : [num_users=1] = get_attr[target=blocks.2.5.dw_start.bn.weight]
    %blocks_2_5_dw_start_bn_bias : [num_users=1] = get_attr[target=blocks.2.5.dw_start.bn.bias]
    %blocks_2_5_dw_start_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.5.dw_start.bn.running_mean]
    %blocks_2_5_dw_start_bn_running_var : [num_users=1] = get_attr[target=blocks.2.5.dw_start.bn.running_var]
    %batch_norm_21 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_5_dw_start_conv, %blocks_2_5_dw_start_bn_running_mean, %blocks_2_5_dw_start_bn_running_var), kwargs = {weight: %blocks_2_5_dw_start_bn_weight, bias: %blocks_2_5_dw_start_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_5_dw_start_bn_drop : [num_users=1] = call_module[target=blocks.2.5.dw_start.bn.drop](args = (%batch_norm_21,), kwargs = {})
    %blocks_2_5_dw_start_bn_act : [num_users=1] = call_module[target=blocks.2.5.dw_start.bn.act](args = (%blocks_2_5_dw_start_bn_drop,), kwargs = {})
    %blocks_2_5_pw_exp_conv : [num_users=3] = call_module[target=blocks.2.5.pw_exp.conv](args = (%blocks_2_5_dw_start_bn_act,), kwargs = {})
    %getattr_45 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_5_pw_exp_conv, ndim), kwargs = {})
    %eq_22 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_45, 4), kwargs = {})
    %getattr_46 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_5_pw_exp_conv, ndim), kwargs = {})
    %_assert_22 : [num_users=0] = call_function[target=torch._assert](args = (%eq_22, expected 4D input (got Proxy(getattr_46)D input)), kwargs = {})
    %blocks_2_5_pw_exp_bn_weight : [num_users=1] = get_attr[target=blocks.2.5.pw_exp.bn.weight]
    %blocks_2_5_pw_exp_bn_bias : [num_users=1] = get_attr[target=blocks.2.5.pw_exp.bn.bias]
    %blocks_2_5_pw_exp_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.5.pw_exp.bn.running_mean]
    %blocks_2_5_pw_exp_bn_running_var : [num_users=1] = get_attr[target=blocks.2.5.pw_exp.bn.running_var]
    %batch_norm_22 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_5_pw_exp_conv, %blocks_2_5_pw_exp_bn_running_mean, %blocks_2_5_pw_exp_bn_running_var), kwargs = {weight: %blocks_2_5_pw_exp_bn_weight, bias: %blocks_2_5_pw_exp_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_5_pw_exp_bn_drop : [num_users=1] = call_module[target=blocks.2.5.pw_exp.bn.drop](args = (%batch_norm_22,), kwargs = {})
    %blocks_2_5_pw_exp_bn_act : [num_users=1] = call_module[target=blocks.2.5.pw_exp.bn.act](args = (%blocks_2_5_pw_exp_bn_drop,), kwargs = {})
    %blocks_2_5_dw_mid : [num_users=1] = call_module[target=blocks.2.5.dw_mid](args = (%blocks_2_5_pw_exp_bn_act,), kwargs = {})
    %blocks_2_5_se : [num_users=1] = call_module[target=blocks.2.5.se](args = (%blocks_2_5_dw_mid,), kwargs = {})
    %blocks_2_5_pw_proj_conv : [num_users=3] = call_module[target=blocks.2.5.pw_proj.conv](args = (%blocks_2_5_se,), kwargs = {})
    %getattr_47 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_2_5_pw_proj_conv, ndim), kwargs = {})
    %eq_23 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_47, 4), kwargs = {})
    %getattr_48 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_2_5_pw_proj_conv, ndim), kwargs = {})
    %_assert_23 : [num_users=0] = call_function[target=torch._assert](args = (%eq_23, expected 4D input (got Proxy(getattr_48)D input)), kwargs = {})
    %blocks_2_5_pw_proj_bn_weight : [num_users=1] = get_attr[target=blocks.2.5.pw_proj.bn.weight]
    %blocks_2_5_pw_proj_bn_bias : [num_users=1] = get_attr[target=blocks.2.5.pw_proj.bn.bias]
    %blocks_2_5_pw_proj_bn_running_mean : [num_users=1] = get_attr[target=blocks.2.5.pw_proj.bn.running_mean]
    %blocks_2_5_pw_proj_bn_running_var : [num_users=1] = get_attr[target=blocks.2.5.pw_proj.bn.running_var]
    %batch_norm_23 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_2_5_pw_proj_conv, %blocks_2_5_pw_proj_bn_running_mean, %blocks_2_5_pw_proj_bn_running_var), kwargs = {weight: %blocks_2_5_pw_proj_bn_weight, bias: %blocks_2_5_pw_proj_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_2_5_pw_proj_bn_drop : [num_users=1] = call_module[target=blocks.2.5.pw_proj.bn.drop](args = (%batch_norm_23,), kwargs = {})
    %blocks_2_5_pw_proj_bn_act : [num_users=1] = call_module[target=blocks.2.5.pw_proj.bn.act](args = (%blocks_2_5_pw_proj_bn_drop,), kwargs = {})
    %blocks_2_5_dw_end : [num_users=1] = call_module[target=blocks.2.5.dw_end](args = (%blocks_2_5_pw_proj_bn_act,), kwargs = {})
    %blocks_2_5_layer_scale : [num_users=1] = call_module[target=blocks.2.5.layer_scale](args = (%blocks_2_5_dw_end,), kwargs = {})
    %blocks_2_5_drop_path : [num_users=1] = call_module[target=blocks.2.5.drop_path](args = (%blocks_2_5_layer_scale,), kwargs = {})
    %add_4 : [num_users=1] = call_function[target=operator.add](args = (%blocks_2_5_drop_path, %add_3), kwargs = {})
    %blocks_3_0_dw_start_conv : [num_users=3] = call_module[target=blocks.3.0.dw_start.conv](args = (%add_4,), kwargs = {})
    %getattr_49 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_0_dw_start_conv, ndim), kwargs = {})
    %eq_24 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_49, 4), kwargs = {})
    %getattr_50 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_0_dw_start_conv, ndim), kwargs = {})
    %_assert_24 : [num_users=0] = call_function[target=torch._assert](args = (%eq_24, expected 4D input (got Proxy(getattr_50)D input)), kwargs = {})
    %blocks_3_0_dw_start_bn_weight : [num_users=1] = get_attr[target=blocks.3.0.dw_start.bn.weight]
    %blocks_3_0_dw_start_bn_bias : [num_users=1] = get_attr[target=blocks.3.0.dw_start.bn.bias]
    %blocks_3_0_dw_start_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.0.dw_start.bn.running_mean]
    %blocks_3_0_dw_start_bn_running_var : [num_users=1] = get_attr[target=blocks.3.0.dw_start.bn.running_var]
    %batch_norm_24 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_0_dw_start_conv, %blocks_3_0_dw_start_bn_running_mean, %blocks_3_0_dw_start_bn_running_var), kwargs = {weight: %blocks_3_0_dw_start_bn_weight, bias: %blocks_3_0_dw_start_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_0_dw_start_bn_drop : [num_users=1] = call_module[target=blocks.3.0.dw_start.bn.drop](args = (%batch_norm_24,), kwargs = {})
    %blocks_3_0_dw_start_bn_act : [num_users=1] = call_module[target=blocks.3.0.dw_start.bn.act](args = (%blocks_3_0_dw_start_bn_drop,), kwargs = {})
    %blocks_3_0_pw_exp_conv : [num_users=3] = call_module[target=blocks.3.0.pw_exp.conv](args = (%blocks_3_0_dw_start_bn_act,), kwargs = {})
    %getattr_51 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_0_pw_exp_conv, ndim), kwargs = {})
    %eq_25 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_51, 4), kwargs = {})
    %getattr_52 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_0_pw_exp_conv, ndim), kwargs = {})
    %_assert_25 : [num_users=0] = call_function[target=torch._assert](args = (%eq_25, expected 4D input (got Proxy(getattr_52)D input)), kwargs = {})
    %blocks_3_0_pw_exp_bn_weight : [num_users=1] = get_attr[target=blocks.3.0.pw_exp.bn.weight]
    %blocks_3_0_pw_exp_bn_bias : [num_users=1] = get_attr[target=blocks.3.0.pw_exp.bn.bias]
    %blocks_3_0_pw_exp_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.0.pw_exp.bn.running_mean]
    %blocks_3_0_pw_exp_bn_running_var : [num_users=1] = get_attr[target=blocks.3.0.pw_exp.bn.running_var]
    %batch_norm_25 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_0_pw_exp_conv, %blocks_3_0_pw_exp_bn_running_mean, %blocks_3_0_pw_exp_bn_running_var), kwargs = {weight: %blocks_3_0_pw_exp_bn_weight, bias: %blocks_3_0_pw_exp_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_0_pw_exp_bn_drop : [num_users=1] = call_module[target=blocks.3.0.pw_exp.bn.drop](args = (%batch_norm_25,), kwargs = {})
    %blocks_3_0_pw_exp_bn_act : [num_users=1] = call_module[target=blocks.3.0.pw_exp.bn.act](args = (%blocks_3_0_pw_exp_bn_drop,), kwargs = {})
    %blocks_3_0_dw_mid_conv : [num_users=3] = call_module[target=blocks.3.0.dw_mid.conv](args = (%blocks_3_0_pw_exp_bn_act,), kwargs = {})
    %getattr_53 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_0_dw_mid_conv, ndim), kwargs = {})
    %eq_26 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_53, 4), kwargs = {})
    %getattr_54 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_0_dw_mid_conv, ndim), kwargs = {})
    %_assert_26 : [num_users=0] = call_function[target=torch._assert](args = (%eq_26, expected 4D input (got Proxy(getattr_54)D input)), kwargs = {})
    %blocks_3_0_dw_mid_bn_weight : [num_users=1] = get_attr[target=blocks.3.0.dw_mid.bn.weight]
    %blocks_3_0_dw_mid_bn_bias : [num_users=1] = get_attr[target=blocks.3.0.dw_mid.bn.bias]
    %blocks_3_0_dw_mid_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.0.dw_mid.bn.running_mean]
    %blocks_3_0_dw_mid_bn_running_var : [num_users=1] = get_attr[target=blocks.3.0.dw_mid.bn.running_var]
    %batch_norm_26 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_0_dw_mid_conv, %blocks_3_0_dw_mid_bn_running_mean, %blocks_3_0_dw_mid_bn_running_var), kwargs = {weight: %blocks_3_0_dw_mid_bn_weight, bias: %blocks_3_0_dw_mid_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_0_dw_mid_bn_drop : [num_users=1] = call_module[target=blocks.3.0.dw_mid.bn.drop](args = (%batch_norm_26,), kwargs = {})
    %blocks_3_0_dw_mid_bn_act : [num_users=1] = call_module[target=blocks.3.0.dw_mid.bn.act](args = (%blocks_3_0_dw_mid_bn_drop,), kwargs = {})
    %blocks_3_0_se : [num_users=1] = call_module[target=blocks.3.0.se](args = (%blocks_3_0_dw_mid_bn_act,), kwargs = {})
    %blocks_3_0_pw_proj_conv : [num_users=3] = call_module[target=blocks.3.0.pw_proj.conv](args = (%blocks_3_0_se,), kwargs = {})
    %getattr_55 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_0_pw_proj_conv, ndim), kwargs = {})
    %eq_27 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_55, 4), kwargs = {})
    %getattr_56 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_0_pw_proj_conv, ndim), kwargs = {})
    %_assert_27 : [num_users=0] = call_function[target=torch._assert](args = (%eq_27, expected 4D input (got Proxy(getattr_56)D input)), kwargs = {})
    %blocks_3_0_pw_proj_bn_weight : [num_users=1] = get_attr[target=blocks.3.0.pw_proj.bn.weight]
    %blocks_3_0_pw_proj_bn_bias : [num_users=1] = get_attr[target=blocks.3.0.pw_proj.bn.bias]
    %blocks_3_0_pw_proj_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.0.pw_proj.bn.running_mean]
    %blocks_3_0_pw_proj_bn_running_var : [num_users=1] = get_attr[target=blocks.3.0.pw_proj.bn.running_var]
    %batch_norm_27 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_0_pw_proj_conv, %blocks_3_0_pw_proj_bn_running_mean, %blocks_3_0_pw_proj_bn_running_var), kwargs = {weight: %blocks_3_0_pw_proj_bn_weight, bias: %blocks_3_0_pw_proj_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_0_pw_proj_bn_drop : [num_users=1] = call_module[target=blocks.3.0.pw_proj.bn.drop](args = (%batch_norm_27,), kwargs = {})
    %blocks_3_0_pw_proj_bn_act : [num_users=1] = call_module[target=blocks.3.0.pw_proj.bn.act](args = (%blocks_3_0_pw_proj_bn_drop,), kwargs = {})
    %blocks_3_0_dw_end : [num_users=1] = call_module[target=blocks.3.0.dw_end](args = (%blocks_3_0_pw_proj_bn_act,), kwargs = {})
    %blocks_3_0_layer_scale : [num_users=2] = call_module[target=blocks.3.0.layer_scale](args = (%blocks_3_0_dw_end,), kwargs = {})
    %blocks_3_1_dw_start_conv : [num_users=3] = call_module[target=blocks.3.1.dw_start.conv](args = (%blocks_3_0_layer_scale,), kwargs = {})
    %getattr_57 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_1_dw_start_conv, ndim), kwargs = {})
    %eq_28 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_57, 4), kwargs = {})
    %getattr_58 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_1_dw_start_conv, ndim), kwargs = {})
    %_assert_28 : [num_users=0] = call_function[target=torch._assert](args = (%eq_28, expected 4D input (got Proxy(getattr_58)D input)), kwargs = {})
    %blocks_3_1_dw_start_bn_weight : [num_users=1] = get_attr[target=blocks.3.1.dw_start.bn.weight]
    %blocks_3_1_dw_start_bn_bias : [num_users=1] = get_attr[target=blocks.3.1.dw_start.bn.bias]
    %blocks_3_1_dw_start_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.1.dw_start.bn.running_mean]
    %blocks_3_1_dw_start_bn_running_var : [num_users=1] = get_attr[target=blocks.3.1.dw_start.bn.running_var]
    %batch_norm_28 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_1_dw_start_conv, %blocks_3_1_dw_start_bn_running_mean, %blocks_3_1_dw_start_bn_running_var), kwargs = {weight: %blocks_3_1_dw_start_bn_weight, bias: %blocks_3_1_dw_start_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_1_dw_start_bn_drop : [num_users=1] = call_module[target=blocks.3.1.dw_start.bn.drop](args = (%batch_norm_28,), kwargs = {})
    %blocks_3_1_dw_start_bn_act : [num_users=1] = call_module[target=blocks.3.1.dw_start.bn.act](args = (%blocks_3_1_dw_start_bn_drop,), kwargs = {})
    %blocks_3_1_pw_exp_conv : [num_users=3] = call_module[target=blocks.3.1.pw_exp.conv](args = (%blocks_3_1_dw_start_bn_act,), kwargs = {})
    %getattr_59 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_1_pw_exp_conv, ndim), kwargs = {})
    %eq_29 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_59, 4), kwargs = {})
    %getattr_60 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_1_pw_exp_conv, ndim), kwargs = {})
    %_assert_29 : [num_users=0] = call_function[target=torch._assert](args = (%eq_29, expected 4D input (got Proxy(getattr_60)D input)), kwargs = {})
    %blocks_3_1_pw_exp_bn_weight : [num_users=1] = get_attr[target=blocks.3.1.pw_exp.bn.weight]
    %blocks_3_1_pw_exp_bn_bias : [num_users=1] = get_attr[target=blocks.3.1.pw_exp.bn.bias]
    %blocks_3_1_pw_exp_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.1.pw_exp.bn.running_mean]
    %blocks_3_1_pw_exp_bn_running_var : [num_users=1] = get_attr[target=blocks.3.1.pw_exp.bn.running_var]
    %batch_norm_29 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_1_pw_exp_conv, %blocks_3_1_pw_exp_bn_running_mean, %blocks_3_1_pw_exp_bn_running_var), kwargs = {weight: %blocks_3_1_pw_exp_bn_weight, bias: %blocks_3_1_pw_exp_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_1_pw_exp_bn_drop : [num_users=1] = call_module[target=blocks.3.1.pw_exp.bn.drop](args = (%batch_norm_29,), kwargs = {})
    %blocks_3_1_pw_exp_bn_act : [num_users=1] = call_module[target=blocks.3.1.pw_exp.bn.act](args = (%blocks_3_1_pw_exp_bn_drop,), kwargs = {})
    %blocks_3_1_dw_mid_conv : [num_users=3] = call_module[target=blocks.3.1.dw_mid.conv](args = (%blocks_3_1_pw_exp_bn_act,), kwargs = {})
    %getattr_61 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_1_dw_mid_conv, ndim), kwargs = {})
    %eq_30 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_61, 4), kwargs = {})
    %getattr_62 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_1_dw_mid_conv, ndim), kwargs = {})
    %_assert_30 : [num_users=0] = call_function[target=torch._assert](args = (%eq_30, expected 4D input (got Proxy(getattr_62)D input)), kwargs = {})
    %blocks_3_1_dw_mid_bn_weight : [num_users=1] = get_attr[target=blocks.3.1.dw_mid.bn.weight]
    %blocks_3_1_dw_mid_bn_bias : [num_users=1] = get_attr[target=blocks.3.1.dw_mid.bn.bias]
    %blocks_3_1_dw_mid_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.1.dw_mid.bn.running_mean]
    %blocks_3_1_dw_mid_bn_running_var : [num_users=1] = get_attr[target=blocks.3.1.dw_mid.bn.running_var]
    %batch_norm_30 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_1_dw_mid_conv, %blocks_3_1_dw_mid_bn_running_mean, %blocks_3_1_dw_mid_bn_running_var), kwargs = {weight: %blocks_3_1_dw_mid_bn_weight, bias: %blocks_3_1_dw_mid_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_1_dw_mid_bn_drop : [num_users=1] = call_module[target=blocks.3.1.dw_mid.bn.drop](args = (%batch_norm_30,), kwargs = {})
    %blocks_3_1_dw_mid_bn_act : [num_users=1] = call_module[target=blocks.3.1.dw_mid.bn.act](args = (%blocks_3_1_dw_mid_bn_drop,), kwargs = {})
    %blocks_3_1_se : [num_users=1] = call_module[target=blocks.3.1.se](args = (%blocks_3_1_dw_mid_bn_act,), kwargs = {})
    %blocks_3_1_pw_proj_conv : [num_users=3] = call_module[target=blocks.3.1.pw_proj.conv](args = (%blocks_3_1_se,), kwargs = {})
    %getattr_63 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_1_pw_proj_conv, ndim), kwargs = {})
    %eq_31 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_63, 4), kwargs = {})
    %getattr_64 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_1_pw_proj_conv, ndim), kwargs = {})
    %_assert_31 : [num_users=0] = call_function[target=torch._assert](args = (%eq_31, expected 4D input (got Proxy(getattr_64)D input)), kwargs = {})
    %blocks_3_1_pw_proj_bn_weight : [num_users=1] = get_attr[target=blocks.3.1.pw_proj.bn.weight]
    %blocks_3_1_pw_proj_bn_bias : [num_users=1] = get_attr[target=blocks.3.1.pw_proj.bn.bias]
    %blocks_3_1_pw_proj_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.1.pw_proj.bn.running_mean]
    %blocks_3_1_pw_proj_bn_running_var : [num_users=1] = get_attr[target=blocks.3.1.pw_proj.bn.running_var]
    %batch_norm_31 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_1_pw_proj_conv, %blocks_3_1_pw_proj_bn_running_mean, %blocks_3_1_pw_proj_bn_running_var), kwargs = {weight: %blocks_3_1_pw_proj_bn_weight, bias: %blocks_3_1_pw_proj_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_1_pw_proj_bn_drop : [num_users=1] = call_module[target=blocks.3.1.pw_proj.bn.drop](args = (%batch_norm_31,), kwargs = {})
    %blocks_3_1_pw_proj_bn_act : [num_users=1] = call_module[target=blocks.3.1.pw_proj.bn.act](args = (%blocks_3_1_pw_proj_bn_drop,), kwargs = {})
    %blocks_3_1_dw_end : [num_users=1] = call_module[target=blocks.3.1.dw_end](args = (%blocks_3_1_pw_proj_bn_act,), kwargs = {})
    %blocks_3_1_layer_scale : [num_users=1] = call_module[target=blocks.3.1.layer_scale](args = (%blocks_3_1_dw_end,), kwargs = {})
    %blocks_3_1_drop_path : [num_users=1] = call_module[target=blocks.3.1.drop_path](args = (%blocks_3_1_layer_scale,), kwargs = {})
    %add_5 : [num_users=2] = call_function[target=operator.add](args = (%blocks_3_1_drop_path, %blocks_3_0_layer_scale), kwargs = {})
    %blocks_3_2_dw_start : [num_users=1] = call_module[target=blocks.3.2.dw_start](args = (%add_5,), kwargs = {})
    %blocks_3_2_pw_exp_conv : [num_users=3] = call_module[target=blocks.3.2.pw_exp.conv](args = (%blocks_3_2_dw_start,), kwargs = {})
    %getattr_65 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_2_pw_exp_conv, ndim), kwargs = {})
    %eq_32 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_65, 4), kwargs = {})
    %getattr_66 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_2_pw_exp_conv, ndim), kwargs = {})
    %_assert_32 : [num_users=0] = call_function[target=torch._assert](args = (%eq_32, expected 4D input (got Proxy(getattr_66)D input)), kwargs = {})
    %blocks_3_2_pw_exp_bn_weight : [num_users=1] = get_attr[target=blocks.3.2.pw_exp.bn.weight]
    %blocks_3_2_pw_exp_bn_bias : [num_users=1] = get_attr[target=blocks.3.2.pw_exp.bn.bias]
    %blocks_3_2_pw_exp_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.2.pw_exp.bn.running_mean]
    %blocks_3_2_pw_exp_bn_running_var : [num_users=1] = get_attr[target=blocks.3.2.pw_exp.bn.running_var]
    %batch_norm_32 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_2_pw_exp_conv, %blocks_3_2_pw_exp_bn_running_mean, %blocks_3_2_pw_exp_bn_running_var), kwargs = {weight: %blocks_3_2_pw_exp_bn_weight, bias: %blocks_3_2_pw_exp_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_2_pw_exp_bn_drop : [num_users=1] = call_module[target=blocks.3.2.pw_exp.bn.drop](args = (%batch_norm_32,), kwargs = {})
    %blocks_3_2_pw_exp_bn_act : [num_users=1] = call_module[target=blocks.3.2.pw_exp.bn.act](args = (%blocks_3_2_pw_exp_bn_drop,), kwargs = {})
    %blocks_3_2_dw_mid_conv : [num_users=3] = call_module[target=blocks.3.2.dw_mid.conv](args = (%blocks_3_2_pw_exp_bn_act,), kwargs = {})
    %getattr_67 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_2_dw_mid_conv, ndim), kwargs = {})
    %eq_33 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_67, 4), kwargs = {})
    %getattr_68 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_2_dw_mid_conv, ndim), kwargs = {})
    %_assert_33 : [num_users=0] = call_function[target=torch._assert](args = (%eq_33, expected 4D input (got Proxy(getattr_68)D input)), kwargs = {})
    %blocks_3_2_dw_mid_bn_weight : [num_users=1] = get_attr[target=blocks.3.2.dw_mid.bn.weight]
    %blocks_3_2_dw_mid_bn_bias : [num_users=1] = get_attr[target=blocks.3.2.dw_mid.bn.bias]
    %blocks_3_2_dw_mid_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.2.dw_mid.bn.running_mean]
    %blocks_3_2_dw_mid_bn_running_var : [num_users=1] = get_attr[target=blocks.3.2.dw_mid.bn.running_var]
    %batch_norm_33 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_2_dw_mid_conv, %blocks_3_2_dw_mid_bn_running_mean, %blocks_3_2_dw_mid_bn_running_var), kwargs = {weight: %blocks_3_2_dw_mid_bn_weight, bias: %blocks_3_2_dw_mid_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_2_dw_mid_bn_drop : [num_users=1] = call_module[target=blocks.3.2.dw_mid.bn.drop](args = (%batch_norm_33,), kwargs = {})
    %blocks_3_2_dw_mid_bn_act : [num_users=1] = call_module[target=blocks.3.2.dw_mid.bn.act](args = (%blocks_3_2_dw_mid_bn_drop,), kwargs = {})
    %blocks_3_2_se : [num_users=1] = call_module[target=blocks.3.2.se](args = (%blocks_3_2_dw_mid_bn_act,), kwargs = {})
    %blocks_3_2_pw_proj_conv : [num_users=3] = call_module[target=blocks.3.2.pw_proj.conv](args = (%blocks_3_2_se,), kwargs = {})
    %getattr_69 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_2_pw_proj_conv, ndim), kwargs = {})
    %eq_34 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_69, 4), kwargs = {})
    %getattr_70 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_2_pw_proj_conv, ndim), kwargs = {})
    %_assert_34 : [num_users=0] = call_function[target=torch._assert](args = (%eq_34, expected 4D input (got Proxy(getattr_70)D input)), kwargs = {})
    %blocks_3_2_pw_proj_bn_weight : [num_users=1] = get_attr[target=blocks.3.2.pw_proj.bn.weight]
    %blocks_3_2_pw_proj_bn_bias : [num_users=1] = get_attr[target=blocks.3.2.pw_proj.bn.bias]
    %blocks_3_2_pw_proj_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.2.pw_proj.bn.running_mean]
    %blocks_3_2_pw_proj_bn_running_var : [num_users=1] = get_attr[target=blocks.3.2.pw_proj.bn.running_var]
    %batch_norm_34 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_2_pw_proj_conv, %blocks_3_2_pw_proj_bn_running_mean, %blocks_3_2_pw_proj_bn_running_var), kwargs = {weight: %blocks_3_2_pw_proj_bn_weight, bias: %blocks_3_2_pw_proj_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_2_pw_proj_bn_drop : [num_users=1] = call_module[target=blocks.3.2.pw_proj.bn.drop](args = (%batch_norm_34,), kwargs = {})
    %blocks_3_2_pw_proj_bn_act : [num_users=1] = call_module[target=blocks.3.2.pw_proj.bn.act](args = (%blocks_3_2_pw_proj_bn_drop,), kwargs = {})
    %blocks_3_2_dw_end : [num_users=1] = call_module[target=blocks.3.2.dw_end](args = (%blocks_3_2_pw_proj_bn_act,), kwargs = {})
    %blocks_3_2_layer_scale : [num_users=1] = call_module[target=blocks.3.2.layer_scale](args = (%blocks_3_2_dw_end,), kwargs = {})
    %blocks_3_2_drop_path : [num_users=1] = call_module[target=blocks.3.2.drop_path](args = (%blocks_3_2_layer_scale,), kwargs = {})
    %add_6 : [num_users=2] = call_function[target=operator.add](args = (%blocks_3_2_drop_path, %add_5), kwargs = {})
    %blocks_3_3_dw_start : [num_users=1] = call_module[target=blocks.3.3.dw_start](args = (%add_6,), kwargs = {})
    %blocks_3_3_pw_exp_conv : [num_users=3] = call_module[target=blocks.3.3.pw_exp.conv](args = (%blocks_3_3_dw_start,), kwargs = {})
    %getattr_71 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_3_pw_exp_conv, ndim), kwargs = {})
    %eq_35 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_71, 4), kwargs = {})
    %getattr_72 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_3_pw_exp_conv, ndim), kwargs = {})
    %_assert_35 : [num_users=0] = call_function[target=torch._assert](args = (%eq_35, expected 4D input (got Proxy(getattr_72)D input)), kwargs = {})
    %blocks_3_3_pw_exp_bn_weight : [num_users=1] = get_attr[target=blocks.3.3.pw_exp.bn.weight]
    %blocks_3_3_pw_exp_bn_bias : [num_users=1] = get_attr[target=blocks.3.3.pw_exp.bn.bias]
    %blocks_3_3_pw_exp_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.3.pw_exp.bn.running_mean]
    %blocks_3_3_pw_exp_bn_running_var : [num_users=1] = get_attr[target=blocks.3.3.pw_exp.bn.running_var]
    %batch_norm_35 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_3_pw_exp_conv, %blocks_3_3_pw_exp_bn_running_mean, %blocks_3_3_pw_exp_bn_running_var), kwargs = {weight: %blocks_3_3_pw_exp_bn_weight, bias: %blocks_3_3_pw_exp_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_3_pw_exp_bn_drop : [num_users=1] = call_module[target=blocks.3.3.pw_exp.bn.drop](args = (%batch_norm_35,), kwargs = {})
    %blocks_3_3_pw_exp_bn_act : [num_users=1] = call_module[target=blocks.3.3.pw_exp.bn.act](args = (%blocks_3_3_pw_exp_bn_drop,), kwargs = {})
    %blocks_3_3_dw_mid_conv : [num_users=3] = call_module[target=blocks.3.3.dw_mid.conv](args = (%blocks_3_3_pw_exp_bn_act,), kwargs = {})
    %getattr_73 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_3_dw_mid_conv, ndim), kwargs = {})
    %eq_36 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_73, 4), kwargs = {})
    %getattr_74 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_3_dw_mid_conv, ndim), kwargs = {})
    %_assert_36 : [num_users=0] = call_function[target=torch._assert](args = (%eq_36, expected 4D input (got Proxy(getattr_74)D input)), kwargs = {})
    %blocks_3_3_dw_mid_bn_weight : [num_users=1] = get_attr[target=blocks.3.3.dw_mid.bn.weight]
    %blocks_3_3_dw_mid_bn_bias : [num_users=1] = get_attr[target=blocks.3.3.dw_mid.bn.bias]
    %blocks_3_3_dw_mid_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.3.dw_mid.bn.running_mean]
    %blocks_3_3_dw_mid_bn_running_var : [num_users=1] = get_attr[target=blocks.3.3.dw_mid.bn.running_var]
    %batch_norm_36 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_3_dw_mid_conv, %blocks_3_3_dw_mid_bn_running_mean, %blocks_3_3_dw_mid_bn_running_var), kwargs = {weight: %blocks_3_3_dw_mid_bn_weight, bias: %blocks_3_3_dw_mid_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_3_dw_mid_bn_drop : [num_users=1] = call_module[target=blocks.3.3.dw_mid.bn.drop](args = (%batch_norm_36,), kwargs = {})
    %blocks_3_3_dw_mid_bn_act : [num_users=1] = call_module[target=blocks.3.3.dw_mid.bn.act](args = (%blocks_3_3_dw_mid_bn_drop,), kwargs = {})
    %blocks_3_3_se : [num_users=1] = call_module[target=blocks.3.3.se](args = (%blocks_3_3_dw_mid_bn_act,), kwargs = {})
    %blocks_3_3_pw_proj_conv : [num_users=3] = call_module[target=blocks.3.3.pw_proj.conv](args = (%blocks_3_3_se,), kwargs = {})
    %getattr_75 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_3_pw_proj_conv, ndim), kwargs = {})
    %eq_37 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_75, 4), kwargs = {})
    %getattr_76 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_3_pw_proj_conv, ndim), kwargs = {})
    %_assert_37 : [num_users=0] = call_function[target=torch._assert](args = (%eq_37, expected 4D input (got Proxy(getattr_76)D input)), kwargs = {})
    %blocks_3_3_pw_proj_bn_weight : [num_users=1] = get_attr[target=blocks.3.3.pw_proj.bn.weight]
    %blocks_3_3_pw_proj_bn_bias : [num_users=1] = get_attr[target=blocks.3.3.pw_proj.bn.bias]
    %blocks_3_3_pw_proj_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.3.pw_proj.bn.running_mean]
    %blocks_3_3_pw_proj_bn_running_var : [num_users=1] = get_attr[target=blocks.3.3.pw_proj.bn.running_var]
    %batch_norm_37 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_3_pw_proj_conv, %blocks_3_3_pw_proj_bn_running_mean, %blocks_3_3_pw_proj_bn_running_var), kwargs = {weight: %blocks_3_3_pw_proj_bn_weight, bias: %blocks_3_3_pw_proj_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_3_pw_proj_bn_drop : [num_users=1] = call_module[target=blocks.3.3.pw_proj.bn.drop](args = (%batch_norm_37,), kwargs = {})
    %blocks_3_3_pw_proj_bn_act : [num_users=1] = call_module[target=blocks.3.3.pw_proj.bn.act](args = (%blocks_3_3_pw_proj_bn_drop,), kwargs = {})
    %blocks_3_3_dw_end : [num_users=1] = call_module[target=blocks.3.3.dw_end](args = (%blocks_3_3_pw_proj_bn_act,), kwargs = {})
    %blocks_3_3_layer_scale : [num_users=1] = call_module[target=blocks.3.3.layer_scale](args = (%blocks_3_3_dw_end,), kwargs = {})
    %blocks_3_3_drop_path : [num_users=1] = call_module[target=blocks.3.3.drop_path](args = (%blocks_3_3_layer_scale,), kwargs = {})
    %add_7 : [num_users=2] = call_function[target=operator.add](args = (%blocks_3_3_drop_path, %add_6), kwargs = {})
    %blocks_3_4_dw_start : [num_users=1] = call_module[target=blocks.3.4.dw_start](args = (%add_7,), kwargs = {})
    %blocks_3_4_pw_exp_conv : [num_users=3] = call_module[target=blocks.3.4.pw_exp.conv](args = (%blocks_3_4_dw_start,), kwargs = {})
    %getattr_77 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_4_pw_exp_conv, ndim), kwargs = {})
    %eq_38 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_77, 4), kwargs = {})
    %getattr_78 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_4_pw_exp_conv, ndim), kwargs = {})
    %_assert_38 : [num_users=0] = call_function[target=torch._assert](args = (%eq_38, expected 4D input (got Proxy(getattr_78)D input)), kwargs = {})
    %blocks_3_4_pw_exp_bn_weight : [num_users=1] = get_attr[target=blocks.3.4.pw_exp.bn.weight]
    %blocks_3_4_pw_exp_bn_bias : [num_users=1] = get_attr[target=blocks.3.4.pw_exp.bn.bias]
    %blocks_3_4_pw_exp_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.4.pw_exp.bn.running_mean]
    %blocks_3_4_pw_exp_bn_running_var : [num_users=1] = get_attr[target=blocks.3.4.pw_exp.bn.running_var]
    %batch_norm_38 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_4_pw_exp_conv, %blocks_3_4_pw_exp_bn_running_mean, %blocks_3_4_pw_exp_bn_running_var), kwargs = {weight: %blocks_3_4_pw_exp_bn_weight, bias: %blocks_3_4_pw_exp_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_4_pw_exp_bn_drop : [num_users=1] = call_module[target=blocks.3.4.pw_exp.bn.drop](args = (%batch_norm_38,), kwargs = {})
    %blocks_3_4_pw_exp_bn_act : [num_users=1] = call_module[target=blocks.3.4.pw_exp.bn.act](args = (%blocks_3_4_pw_exp_bn_drop,), kwargs = {})
    %blocks_3_4_dw_mid_conv : [num_users=3] = call_module[target=blocks.3.4.dw_mid.conv](args = (%blocks_3_4_pw_exp_bn_act,), kwargs = {})
    %getattr_79 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_4_dw_mid_conv, ndim), kwargs = {})
    %eq_39 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_79, 4), kwargs = {})
    %getattr_80 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_4_dw_mid_conv, ndim), kwargs = {})
    %_assert_39 : [num_users=0] = call_function[target=torch._assert](args = (%eq_39, expected 4D input (got Proxy(getattr_80)D input)), kwargs = {})
    %blocks_3_4_dw_mid_bn_weight : [num_users=1] = get_attr[target=blocks.3.4.dw_mid.bn.weight]
    %blocks_3_4_dw_mid_bn_bias : [num_users=1] = get_attr[target=blocks.3.4.dw_mid.bn.bias]
    %blocks_3_4_dw_mid_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.4.dw_mid.bn.running_mean]
    %blocks_3_4_dw_mid_bn_running_var : [num_users=1] = get_attr[target=blocks.3.4.dw_mid.bn.running_var]
    %batch_norm_39 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_4_dw_mid_conv, %blocks_3_4_dw_mid_bn_running_mean, %blocks_3_4_dw_mid_bn_running_var), kwargs = {weight: %blocks_3_4_dw_mid_bn_weight, bias: %blocks_3_4_dw_mid_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_4_dw_mid_bn_drop : [num_users=1] = call_module[target=blocks.3.4.dw_mid.bn.drop](args = (%batch_norm_39,), kwargs = {})
    %blocks_3_4_dw_mid_bn_act : [num_users=1] = call_module[target=blocks.3.4.dw_mid.bn.act](args = (%blocks_3_4_dw_mid_bn_drop,), kwargs = {})
    %blocks_3_4_se : [num_users=1] = call_module[target=blocks.3.4.se](args = (%blocks_3_4_dw_mid_bn_act,), kwargs = {})
    %blocks_3_4_pw_proj_conv : [num_users=3] = call_module[target=blocks.3.4.pw_proj.conv](args = (%blocks_3_4_se,), kwargs = {})
    %getattr_81 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_4_pw_proj_conv, ndim), kwargs = {})
    %eq_40 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_81, 4), kwargs = {})
    %getattr_82 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_4_pw_proj_conv, ndim), kwargs = {})
    %_assert_40 : [num_users=0] = call_function[target=torch._assert](args = (%eq_40, expected 4D input (got Proxy(getattr_82)D input)), kwargs = {})
    %blocks_3_4_pw_proj_bn_weight : [num_users=1] = get_attr[target=blocks.3.4.pw_proj.bn.weight]
    %blocks_3_4_pw_proj_bn_bias : [num_users=1] = get_attr[target=blocks.3.4.pw_proj.bn.bias]
    %blocks_3_4_pw_proj_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.4.pw_proj.bn.running_mean]
    %blocks_3_4_pw_proj_bn_running_var : [num_users=1] = get_attr[target=blocks.3.4.pw_proj.bn.running_var]
    %batch_norm_40 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_4_pw_proj_conv, %blocks_3_4_pw_proj_bn_running_mean, %blocks_3_4_pw_proj_bn_running_var), kwargs = {weight: %blocks_3_4_pw_proj_bn_weight, bias: %blocks_3_4_pw_proj_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_4_pw_proj_bn_drop : [num_users=1] = call_module[target=blocks.3.4.pw_proj.bn.drop](args = (%batch_norm_40,), kwargs = {})
    %blocks_3_4_pw_proj_bn_act : [num_users=1] = call_module[target=blocks.3.4.pw_proj.bn.act](args = (%blocks_3_4_pw_proj_bn_drop,), kwargs = {})
    %blocks_3_4_dw_end : [num_users=1] = call_module[target=blocks.3.4.dw_end](args = (%blocks_3_4_pw_proj_bn_act,), kwargs = {})
    %blocks_3_4_layer_scale : [num_users=1] = call_module[target=blocks.3.4.layer_scale](args = (%blocks_3_4_dw_end,), kwargs = {})
    %blocks_3_4_drop_path : [num_users=1] = call_module[target=blocks.3.4.drop_path](args = (%blocks_3_4_layer_scale,), kwargs = {})
    %add_8 : [num_users=2] = call_function[target=operator.add](args = (%blocks_3_4_drop_path, %add_7), kwargs = {})
    %blocks_3_5_dw_start : [num_users=1] = call_module[target=blocks.3.5.dw_start](args = (%add_8,), kwargs = {})
    %blocks_3_5_pw_exp_conv : [num_users=3] = call_module[target=blocks.3.5.pw_exp.conv](args = (%blocks_3_5_dw_start,), kwargs = {})
    %getattr_83 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_5_pw_exp_conv, ndim), kwargs = {})
    %eq_41 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_83, 4), kwargs = {})
    %getattr_84 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_5_pw_exp_conv, ndim), kwargs = {})
    %_assert_41 : [num_users=0] = call_function[target=torch._assert](args = (%eq_41, expected 4D input (got Proxy(getattr_84)D input)), kwargs = {})
    %blocks_3_5_pw_exp_bn_weight : [num_users=1] = get_attr[target=blocks.3.5.pw_exp.bn.weight]
    %blocks_3_5_pw_exp_bn_bias : [num_users=1] = get_attr[target=blocks.3.5.pw_exp.bn.bias]
    %blocks_3_5_pw_exp_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.5.pw_exp.bn.running_mean]
    %blocks_3_5_pw_exp_bn_running_var : [num_users=1] = get_attr[target=blocks.3.5.pw_exp.bn.running_var]
    %batch_norm_41 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_5_pw_exp_conv, %blocks_3_5_pw_exp_bn_running_mean, %blocks_3_5_pw_exp_bn_running_var), kwargs = {weight: %blocks_3_5_pw_exp_bn_weight, bias: %blocks_3_5_pw_exp_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_5_pw_exp_bn_drop : [num_users=1] = call_module[target=blocks.3.5.pw_exp.bn.drop](args = (%batch_norm_41,), kwargs = {})
    %blocks_3_5_pw_exp_bn_act : [num_users=1] = call_module[target=blocks.3.5.pw_exp.bn.act](args = (%blocks_3_5_pw_exp_bn_drop,), kwargs = {})
    %blocks_3_5_dw_mid_conv : [num_users=3] = call_module[target=blocks.3.5.dw_mid.conv](args = (%blocks_3_5_pw_exp_bn_act,), kwargs = {})
    %getattr_85 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_5_dw_mid_conv, ndim), kwargs = {})
    %eq_42 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_85, 4), kwargs = {})
    %getattr_86 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_5_dw_mid_conv, ndim), kwargs = {})
    %_assert_42 : [num_users=0] = call_function[target=torch._assert](args = (%eq_42, expected 4D input (got Proxy(getattr_86)D input)), kwargs = {})
    %blocks_3_5_dw_mid_bn_weight : [num_users=1] = get_attr[target=blocks.3.5.dw_mid.bn.weight]
    %blocks_3_5_dw_mid_bn_bias : [num_users=1] = get_attr[target=blocks.3.5.dw_mid.bn.bias]
    %blocks_3_5_dw_mid_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.5.dw_mid.bn.running_mean]
    %blocks_3_5_dw_mid_bn_running_var : [num_users=1] = get_attr[target=blocks.3.5.dw_mid.bn.running_var]
    %batch_norm_42 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_5_dw_mid_conv, %blocks_3_5_dw_mid_bn_running_mean, %blocks_3_5_dw_mid_bn_running_var), kwargs = {weight: %blocks_3_5_dw_mid_bn_weight, bias: %blocks_3_5_dw_mid_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_5_dw_mid_bn_drop : [num_users=1] = call_module[target=blocks.3.5.dw_mid.bn.drop](args = (%batch_norm_42,), kwargs = {})
    %blocks_3_5_dw_mid_bn_act : [num_users=1] = call_module[target=blocks.3.5.dw_mid.bn.act](args = (%blocks_3_5_dw_mid_bn_drop,), kwargs = {})
    %blocks_3_5_se : [num_users=1] = call_module[target=blocks.3.5.se](args = (%blocks_3_5_dw_mid_bn_act,), kwargs = {})
    %blocks_3_5_pw_proj_conv : [num_users=3] = call_module[target=blocks.3.5.pw_proj.conv](args = (%blocks_3_5_se,), kwargs = {})
    %getattr_87 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_3_5_pw_proj_conv, ndim), kwargs = {})
    %eq_43 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_87, 4), kwargs = {})
    %getattr_88 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_3_5_pw_proj_conv, ndim), kwargs = {})
    %_assert_43 : [num_users=0] = call_function[target=torch._assert](args = (%eq_43, expected 4D input (got Proxy(getattr_88)D input)), kwargs = {})
    %blocks_3_5_pw_proj_bn_weight : [num_users=1] = get_attr[target=blocks.3.5.pw_proj.bn.weight]
    %blocks_3_5_pw_proj_bn_bias : [num_users=1] = get_attr[target=blocks.3.5.pw_proj.bn.bias]
    %blocks_3_5_pw_proj_bn_running_mean : [num_users=1] = get_attr[target=blocks.3.5.pw_proj.bn.running_mean]
    %blocks_3_5_pw_proj_bn_running_var : [num_users=1] = get_attr[target=blocks.3.5.pw_proj.bn.running_var]
    %batch_norm_43 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_3_5_pw_proj_conv, %blocks_3_5_pw_proj_bn_running_mean, %blocks_3_5_pw_proj_bn_running_var), kwargs = {weight: %blocks_3_5_pw_proj_bn_weight, bias: %blocks_3_5_pw_proj_bn_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_3_5_pw_proj_bn_drop : [num_users=1] = call_module[target=blocks.3.5.pw_proj.bn.drop](args = (%batch_norm_43,), kwargs = {})
    %blocks_3_5_pw_proj_bn_act : [num_users=1] = call_module[target=blocks.3.5.pw_proj.bn.act](args = (%blocks_3_5_pw_proj_bn_drop,), kwargs = {})
    %blocks_3_5_dw_end : [num_users=1] = call_module[target=blocks.3.5.dw_end](args = (%blocks_3_5_pw_proj_bn_act,), kwargs = {})
    %blocks_3_5_layer_scale : [num_users=1] = call_module[target=blocks.3.5.layer_scale](args = (%blocks_3_5_dw_end,), kwargs = {})
    %blocks_3_5_drop_path : [num_users=1] = call_module[target=blocks.3.5.drop_path](args = (%blocks_3_5_layer_scale,), kwargs = {})
    %add_9 : [num_users=1] = call_function[target=operator.add](args = (%blocks_3_5_drop_path, %add_8), kwargs = {})
    %blocks_4_0_conv : [num_users=3] = call_module[target=blocks.4.0.conv](args = (%add_9,), kwargs = {})
    %getattr_89 : [num_users=1] = call_function[target=builtins.getattr](args = (%blocks_4_0_conv, ndim), kwargs = {})
    %eq_44 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_89, 4), kwargs = {})
    %getattr_90 : [num_users=0] = call_function[target=builtins.getattr](args = (%blocks_4_0_conv, ndim), kwargs = {})
    %_assert_44 : [num_users=0] = call_function[target=torch._assert](args = (%eq_44, expected 4D input (got Proxy(getattr_90)D input)), kwargs = {})
    %blocks_4_0_bn1_weight : [num_users=1] = get_attr[target=blocks.4.0.bn1.weight]
    %blocks_4_0_bn1_bias : [num_users=1] = get_attr[target=blocks.4.0.bn1.bias]
    %blocks_4_0_bn1_running_mean : [num_users=1] = get_attr[target=blocks.4.0.bn1.running_mean]
    %blocks_4_0_bn1_running_var : [num_users=1] = get_attr[target=blocks.4.0.bn1.running_var]
    %batch_norm_44 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%blocks_4_0_conv, %blocks_4_0_bn1_running_mean, %blocks_4_0_bn1_running_var), kwargs = {weight: %blocks_4_0_bn1_weight, bias: %blocks_4_0_bn1_bias, training: False, momentum: 0.1, eps: 1e-05})
    %blocks_4_0_bn1_drop : [num_users=1] = call_module[target=blocks.4.0.bn1.drop](args = (%batch_norm_44,), kwargs = {})
    %blocks_4_0_bn1_act : [num_users=1] = call_module[target=blocks.4.0.bn1.act](args = (%blocks_4_0_bn1_drop,), kwargs = {})
    %blocks_4_0_aa : [num_users=1] = call_module[target=blocks.4.0.aa](args = (%blocks_4_0_bn1_act,), kwargs = {})
    %global_pool_pool : [num_users=1] = call_module[target=global_pool.pool](args = (%blocks_4_0_aa,), kwargs = {})
    %global_pool_flatten : [num_users=1] = call_module[target=global_pool.flatten](args = (%global_pool_pool,), kwargs = {})
    %conv_head : [num_users=3] = call_module[target=conv_head](args = (%global_pool_flatten,), kwargs = {})
    %getattr_91 : [num_users=1] = call_function[target=builtins.getattr](args = (%conv_head, ndim), kwargs = {})
    %eq_45 : [num_users=1] = call_function[target=operator.eq](args = (%getattr_91, 4), kwargs = {})
    %getattr_92 : [num_users=0] = call_function[target=builtins.getattr](args = (%conv_head, ndim), kwargs = {})
    %_assert_45 : [num_users=0] = call_function[target=torch._assert](args = (%eq_45, expected 4D input (got Proxy(getattr_92)D input)), kwargs = {})
    %norm_head_weight : [num_users=1] = get_attr[target=norm_head.weight]
    %norm_head_bias : [num_users=1] = get_attr[target=norm_head.bias]
    %norm_head_running_mean : [num_users=1] = get_attr[target=norm_head.running_mean]
    %norm_head_running_var : [num_users=1] = get_attr[target=norm_head.running_var]
    %batch_norm_45 : [num_users=1] = call_function[target=torch.nn.functional.batch_norm](args = (%conv_head, %norm_head_running_mean, %norm_head_running_var), kwargs = {weight: %norm_head_weight, bias: %norm_head_bias, training: False, momentum: 0.1, eps: 1e-05})
    %norm_head_drop : [num_users=1] = call_module[target=norm_head.drop](args = (%batch_norm_45,), kwargs = {})
    %norm_head_act : [num_users=1] = call_module[target=norm_head.act](args = (%norm_head_drop,), kwargs = {})
    %act2 : [num_users=1] = call_module[target=act2](args = (%norm_head_act,), kwargs = {})
    %flatten : [num_users=1] = call_module[target=flatten](args = (%act2,), kwargs = {})
    %classifier_weight : [num_users=1] = get_attr[target=classifier.weight]
    %classifier_bias : [num_users=1] = get_attr[target=classifier.bias]
    %linear : [num_users=1] = call_function[target=torch._C._nn.linear](args = (%flatten, %classifier_weight, %classifier_bias), kwargs = {})
    return linear
```
