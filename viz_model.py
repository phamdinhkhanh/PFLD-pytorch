import torch
from models.pfld import PFLDInference, AuxiliaryNet
from torchviz import make_dot

model=PFLDInference()
x=torch.randn(64, 3, 112, 112)
features_for_auxiliarynet, landmarks=model(x)

# make_dot(landmarks).render("attached", format="png")

# torch.onnx.export(model,               # model being run
#                   x,                         # model input (or a tuple for multiple inputs)
#                   "pfld.onnx",   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=10,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                   dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
#                                 'output' : {0 : 'batch_size'}})

model2 = AuxiliaryNet()
print("features_for_auxiliarynet: ", features_for_auxiliarynet.size())
torch.onnx.export(model2,               # model being run
                  features_for_auxiliarynet, # model input (or a tuple for multiple inputs)
                  "auxiliary.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})

# Import .onnx model into lutzroeder to visualize model graph
# https://lutzroeder.github.io/netron/