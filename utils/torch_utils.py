import torch
from model.base.fcn import DnnVAD


def load_match_dict(model, model_path):
    # model: single gpu model, please load dict before warp with nn.DataParallel
    pretrain_dict = torch.load(model_path)
    model_dict = model.state_dict()
    # the pretrain dict may be multi gpus, cleaning
    pretrain_dict = {k.replace('.module', ''): v for k, v in pretrain_dict.items()}
    # 1. filter out unnecessary keys
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                     k in model_dict and v.shape == model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrain_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def convert_onnx(model, input_size, model_name):
    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, input_size, requires_grad=True)
    out = model(dummy_input)
    print(out)
    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      model_name + ".onnx",  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    print('Model has been converted to ONNX')


# 将训练好的参数转换为可以用C++代码加载的参数
def convert_trace(model):
    model.eval()
    input_size = 14
    example = torch.rand(1, input_size)
    print(example.size())
    pred = model(example)
    print(pred)
    traced_script_module = torch.jit.trace(model, example)
    output = traced_script_module(torch.ones(1, 14))
    print(output)
    traced_script_module.save("traced_dnn_vad.pt")


if __name__ == '__main__':
    load_model_path = r"F:\workspace\GHT\projects\vad\code\torch_vad\checkpoints\dnn_vad_pref\19_000000.pth"
    model = DnnVAD()
    load_match_dict(model, load_model_path)
    input_size = 14
    model_name = "dnn_vad"

    convert_trace(model)
    # convert_onnx(model, 14, model_name)
