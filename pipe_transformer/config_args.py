from dataclasses import dataclass


@dataclass
class ConfigArgs:
    # switch
    b_auto_dp: bool = True
    b_freeze: bool = True
    b_auto_pipe: bool = True
    b_cache: bool = True

    # DP related
    is_infiniband: bool = True
    master_addr: str = "192.168.1.1"
    master_port: str = "11111"
    if_name: str = "ib0"
    num_nodes: int = 2
    node_rank: int = 0
    world_size: int = 16
    local_rank: int = 0

    # Pipe Related
    pipe_len_at_the_beginning: int = 8
    num_chunks_of_micro_batches: int = 32

    # CV model related
    batch_size: int = 256
    num_layer: int = 12
    output_dim: int = 10
    hidden_size: int = 768
    seq_len: int = 197

    is_debug_mode: bool = False

