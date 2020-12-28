import sys

from model.vit.vision_transformer_origin import CONFIGS
from model.vit.vision_transformer_origin import VisionTransformer
from pipe.auto_pipe import AutoElasticPipe
from pipe.pipe_model_builder import count_parameters, OutputHead, get_ddp_ignored_params_name

if __name__ == "__main__":
    output_dim = 10
    img_size = 224

    pretrained_dir = "../model/vit/pretrained/ViT-B_16.npz"
    model_type = 'vit-B_16'
    # model_type = 'vit-L_32'
    # model_type = 'vit-H_14'
    config = CONFIGS[model_type]


    num_device = 8
    num_layers = config.transformer.num_layers
    print("num_layers = %d" % num_layers)

    dp_num = 2
    device_ranks = []
    for dp_idx in range(dp_num):
        start_rank = dp_idx * num_device
        for local_rank in range(num_device):
            global_rank = local_rank + start_rank
            device_ranks.append((local_rank, global_rank))
    world_size = dp_num * num_device
    for local_rank, global_rank in device_ranks:
        print("local rank %d, global_rank %d. " % (local_rank, global_rank))

        # create model
        print("Vision Transformer Configuration: " + str(config))
        model = VisionTransformer(config, img_size, zero_head=True, num_classes=output_dim, vis=False)
        #    model.load_from(np.load(pretrained_dir))
        model_size = count_parameters(model)
        print("model_size = " + str(model_size))
        output_head = OutputHead(config.hidden_size, output_dim)

        # auto pipe
        auto_pipe = AutoElasticPipe(world_size, local_rank, global_rank, model, output_head,
                                    num_device, num_layers, debug_mode=True)

        for num_frozen_layers in range(num_layers + 1):
            model, pipe_len, params_skip = auto_pipe.transform(num_frozen_layers)
        print("finished!")
    sys.exit()
