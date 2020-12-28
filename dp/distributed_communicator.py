import numpy as np
import torch
import torch.distributed as dist


def dist_broadcast(src, object_list):
    """Broadcasts a given object to all parties."""
    dist.broadcast_object_list(src, object_list)
    return object_list


def dist_send(dest, msg, device_id):
    """Broadcasts a given object to all parties."""
    tensor_obj = torch.from_numpy(np.array(msg))
    tensor_obj = tensor_obj.to(device_id)
    dist.send(tensor_obj, dest)


def dist_receive(source, msg, device_id):
    tensor_obj = torch.from_numpy(np.array(msg)).to(device_id)
    dist.recv(tensor_obj, source)
    msg = tensor_obj.cpu().numpy()
    return msg
