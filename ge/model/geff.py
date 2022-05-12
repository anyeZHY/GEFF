import torch
import torch.nn as nn

"""
models = {
            'Face': resnet18(num_classes=face_dim).to(device),
            'Left': EyeEncoder(dim_features=eyes_dim).to(device),
            'Right': EyeEncoder(dim_features=eyes_dim).to(device),
            'Eye': EyeResEncoder(eyes_dim).to(device),
            'Extractor': MLP(face_dim, eyes_dim).to(device),
            'Fusion_l': MLP(channels=fusion_channel).to(device),
            'Fusion_r': MLP(channels=fusion_channel).to(device),
            'Decoder': MLP(channels=decoder_channel).to(device),
        }
GEFF(models)
"""


class GEFF(nn.Module):
    def __init__(self, models, t, share_eye=False, similarity=False):
        """
        Args:
            models: dictionary
                - face_en: E_f => F_f
                - left_en: E_l => F_l
                - right_en: E_r => F_r
                - extractor: M_f, extract eyes' feature from F_f.
                             >> F_f^l, F_f^r = extractor(F_f)[0:half], another half
                - fusion_l: out_l = M_l(cat(F_l, F_f^l))
                - fusion_r: out_l = M_r(cat(F_r, F_f^r))
                    Why & How to fuse? Equation in Pic. => generate F_l^{fused}, F_r^{fused})
                - decoder: D(cat(F_f, F_l^{fused}, F_r^{fused})) ==> out, to be returned
                - eye_en: E_p. Use carefully
           share_eye: bool. If true: use E_p!
           similarity: bool. If ture:
                                return out, F_f^l, F_f^r, F_l, F_r, F_f
        """
        super(GEFF, self).__init__()
        self.face_en = models['Face']
        self.share_eye = share_eye
        if share_eye:
            self.eye_en = models['Eye']
        else:
            self.left_en = models['Left']
            self.right_en = models['Right']
        self.t = t
        self.extractor = models['Extractor']
        self.left_fusion = models['Fusion_l']
        self.right_fusion = models['Fusion_r']
        self.decoder = models['Decoder']

    def forward(self, imgs, args, similarity=False, cur_epoch=1000):
        faces, lefts, rights = imgs['Face'], imgs['Left'], imgs['Right']
        F_face = self.face_en(faces)
        if args.pretrain and cur_epoch<30:
            F_face = F_face.detach()
        if self.share_eye:
            F_left = self.eye_en(lefts)
            F_right = self.eye_en(lefts)
        else:
            F_left = self.left_en(lefts)
            F_right = self.right_en(rights)
        F_eyes = self.extractor(F_face)
        _, D = F_eyes.shape
        F_lf = F_eyes[:, :D//2]
        F_rf = F_eyes[:, D//2:]
        out_l = self.left_fusion(torch.cat((F_left, F_lf), dim=1))
        out_r = self.right_fusion(torch.cat((F_right, F_rf), dim=1))
        out_r, out_l = torch.abs(out_r), torch.abs(out_l)
        w_l, w_r = 1/(1+torch.exp(-self.t*out_l)), 1/(1+torch.exp(-self.t*out_r))
        F_lfused = F_left * (1-w_l) + F_lf * w_l
        F_rfused = F_right * (1-w_r) + F_rf * w_r
        features = torch.cat((F_face, F_lfused, F_rfused), dim=1)
        gaze = self.decoder(features)
        if similarity:
            return gaze, F_lf, F_rf, F_left, F_right, F_face
        else:
            return gaze
