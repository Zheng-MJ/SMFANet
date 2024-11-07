import torch
import argparse
from tqdm import tqdm
from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
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
  clip = 500
  h, w = 1280, 720
  model = get_model(args.model_id, args.scale)
  model = model.cuda()
  dummy_input = torch.randn(1, 3, h // args.scale, w // args.scale).cuda()

 
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  runtime = 0

  #  model.eval()
  with torch.no_grad():
    # print(model)
    for _ in tqdm(range(clip)):
        _ = model(dummy_input)

    for _ in tqdm(range(clip)):
        start.record()
        _ = model(dummy_input)
        end.record()
        torch.cuda.synchronize()
        runtime += start.elapsed_time(end)

    avg_time = runtime / clip
    max_memory = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2

    print(model.__class__.__name__)
    print(f'{clip} Number Frames x{args.scale} SR Per Frame Time: {avg_time :.6f} ms')
    print(f' x{args.scale}SR FPS: {(1000 / avg_time):.6f} FPS')
    print(f' Max Memery {max_memory:.6f} [M]')
    output = model(dummy_input)
    print(output.shape)
    print(flop_count_table(FlopCountAnalysis(model, dummy_input), activations=ActivationCountAnalysis(model, dummy_input)))
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Flops")
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--scale", default=4, type=int)
    args = parser.parse_args()
    main(args)
