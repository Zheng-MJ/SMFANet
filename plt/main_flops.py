
import torch
import argparse
from fvcore.nn import FlopCountAnalysis, flop_count_table
from models.SMFANet import SMFANet
from models.SAFMN import SAFMN
from models.ShuffleMixer import ShuffleMixer

def get_model(model_id = 0, scale = 4):
  if model_id == 0:
      model = SMFANet(dim=36, n_blocks=8, ffn_scale=2, upscaling_factor=scale)
  elif model_id == 1:
      model = SMFANet(dim=48, n_blocks=12, ffn_scale=2, upscaling_factor=scale)
  elif model_id == 2:
      model = SAFMN(dim=36, n_blocks=8, ffn_scale=2, upscaling_factor=scale)
  elif model_id == 3:
     model = ShuffleMixer(n_feats=64, kernel_size=7, n_blocks=5, mlp_ratio=2, upscaling_factor=scale)
  else:
     assert False, "Model ID ERRO"
  return model

def main(args):
  h, w = 1280, 720
  model = get_model(args.model_id, args.scale)
  dummy_input = torch.randn(1, 3, h // args.scale, w // args.scale)

  with torch.no_grad():
    flops = FlopCountAnalysis(model, dummy_input )
    print(model.__class__.__name__)
    print(flop_count_table(flops))
    y = model(dummy_input)
    print("output.shape", y.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Flops")
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--scale", default=4, type=int)
    args = parser.parse_args()
    main(args)
