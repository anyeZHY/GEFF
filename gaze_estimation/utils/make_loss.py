import numpy as np
import torch

def yaw_pitch_to_vec(gaze: torch.Tensor):
    """
    convert gaze = (yaw, pitch) to vector (x, y, z)
    """
    N, _ = gaze.size()
    yaw = gaze[:,0].view((N,-1))
    pitch = gaze[:,1].view((N,-1))
    z = torch.sin(pitch)
    y = torch.cos(pitch) * torch.sin(yaw)
    x = torch.cos(pitch) * torch.cos(yaw)
    xyz = torch.cat([x,y,z], dim=1)
    return xyz

# ============== Angular Error >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def angular_error(a, b):
    """
    Input:
    -a: of size (N,2) or (N,3)
    -b: of size (N,2) or (N,3)
    .. math:: \frac{1}{N} \sum_{i=1}^N \arccos(\langle a_i, b_i \rangle)
    """
    a = yaw_pitch_to_vec(a) if a.shape[1]==2 else a
    b = yaw_pitch_to_vec(b) if b.shape[1]==2 else b
    ab =torch.sum(torch.multiply(a, b), dim=1)
    a_norm = torch.linalg.norm(a, axis=1)
    b_norm = torch.linalg.norm(b, axis=1)
    a_norm = torch.clip(a_norm, min=1e-7, max=None)
    b_norm = torch.clip(b_norm, min=1e-6, max=None)

    similarity = torch.divide(ab, torch.multiply(a_norm, b_norm))
    pi = torch.acos(torch.zeros(1)).item() * 2
    return torch.mean(torch.arccos(similarity) * 180.0 / np.pi)

if __name__ == '__main__':
    a = torch.tensor([[0.5, 0.5],[0.1,0.3]])
    b = torch.tensor([[0.1, 0.4],[0.1,0.299999]])
    print(yaw_pitch_to_vec(a)-yaw_pitch_to_vec(b))
    print(angular_error(a, b))
