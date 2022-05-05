import torch
import torch.nn as nn
import torchvision
import resnet
import encoders

"""
models = {
  'face_en': resnet,
    'right_'
}
GEFF(models)
"""

class GEFF(nn.Module):
    def __init__(self, models, share_eye=False, similarity=False):
        """
        Args:
            models: dictionary
                - face_en: E_f => F_f
                - left_en: E_l => F_l
                - right_en: E_r => F_r
                - extractor: M_f, extract eyes' feature from F_f.
                             >> F_f^l, F_f^r = extractor(F_f)[0:half], another half
                - fussion_l: out_l = M_l(cat(F_l, F_f^l))
                - fussion_r: out_l = M_r(cat(F_r, F_f^r))
                    Why & How to fuse? Equation in Pic. => generate F_l^{fused}, F_r^{fused})
                - decoder: D(cat(F_f, F_l^{fused}, F_r^{fused})) ==> out, to be returned
                - eye_en: E_p. Use carfully
           share_eye: bool. If true: use E_p!
           similarity: bool. If ture:
                                return out, F_f^l, F_f^r, F_l, F_r, F_f
        """
        super(GEFF, self).__init__()
        #### Your code here
        pass
        self.face_en = models['face_en']
        self.t_l = torch.tensor(something ,requires_grad=True)
        self.share_eye=share_eye
        if share_eye:
            self.eye_en=models[]
        else:

    def forward(self, imags):
        pass
