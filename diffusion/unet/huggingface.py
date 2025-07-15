from diffusers import UNet2DModel, DDIMPipeline
import torch

from diffusion.unet.models import Model

def get_diffusion(args):
    pipeline = DDIMPipeline.from_pretrained(args.model_name_or_path)
    scheduler = pipeline.scheduler
    
    unet_ = pipeline.unet
    # model = Model()
    # #print('model',model)
    
    # states = torch.load(args.pretrained_model_path, map_location=args.device)
    # model = model.to(args.device)

    # new_state_dict = {}
    # for key,v in states[0].items():
    #     new_key = key[len("module."):]
    #     new_state_dict[new_key] = v
    # #print('states[0]',states[0])
    # model.load_state_dict(new_state_dict, strict=True)
    # unet_ = model
    unet_.to(args.device)
    unet_.eval()
    unet = lambda x,t: unet_(x, t).sample

    scheduler.set_timesteps(args.inference_steps)
    ts = scheduler.timesteps

    alpha_prod_ts = scheduler.alphas_cumprod[ts]
    alpha_prod_t_prevs = torch.cat([alpha_prod_ts[1:], torch.ones(1) * scheduler.final_alpha_cumprod])

    return unet, ts, alpha_prod_ts, alpha_prod_t_prevs