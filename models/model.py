import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial, reduce
from models.modules import ResNet101, ASPP


class DeepLabV3(nn.Module):
    def __init__(
        self,
        output_stride=16,
        norm_act='bn_sync',
        backbone_pretrained=False,
        classes=None,
        use_cosine=False,
        freeze_all_bn=False,
        freeze_backbone_bn=False,
    ):
        super().__init__()
        self.norm_act = norm_act
        if norm_act == 'iabn_sync':
            from inplace_abn import ABN, InPlaceABNSync
            norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01)
        elif norm_act == 'bn_sync':
            norm = nn.BatchNorm2d
        else:
            raise NotImplementedError

        self.freeze_all_bn = freeze_all_bn
        self.freeze_backbone_bn = freeze_backbone_bn

        self.classes = classes
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.use_cosine = use_cosine
        use_bias = not use_cosine

        # Network
        self.backbone = ResNet101(norm, norm_act, output_stride, backbone_pretrained)
        self.aspp = ASPP(2048, 256, 256, norm_act=norm_act, norm=norm, output_stride=output_stride)
        self.cls = nn.ModuleList([nn.Conv2d(256, c, kernel_size=1, bias=use_bias) for c in classes])  # cls[0]: an auxiliary classifier
        self._init_classifier()

    def forward(self, x, ret_intermediate=False, fake_features=None, region_bg=None):
        out_size = x.shape[-2:]  # spatial size
        x_b, x_pl, attentions = self.forward_before_class_prediction(x)
        sem_logits_small = self.forward_class_prediction(x_pl)

        logits_for_fake = None
        logits_for_extra_bg = None
        if fake_features is not None:
            logits_for_fake = self.forward_class_prediction(fake_features)

        if region_bg is not None:
            extra_bg = x_pl.permute(1, 0, 2, 3)[:, region_bg].unsqueeze(0).unsqueeze(3)
            if extra_bg.shape[2] > 0:
                logits_for_extra_bg = self.forward_class_prediction(extra_bg)

        sem_logits = F.interpolate(
            sem_logits_small, size=out_size,
            mode="bilinear", align_corners=False
        )

        if ret_intermediate:
            features = attentions
            features.append(x_b)
            features.append(x_pl)

            return sem_logits, features, [logits_for_fake, logits_for_extra_bg]
        else:
            return sem_logits, {}, [logits_for_fake, logits_for_extra_bg]

    def forward_before_class_prediction(self, x):
        x_b, attentions = self.backbone(x)
        x_pl = self.aspp(x_b)
        return x_b, x_pl, attentions

    def forward_class_prediction(self, x_pl):
        out = []
        for i, mod in enumerate(self.cls):
            out.append(mod(x_pl))  # [N, c, H, W]
        x_o = torch.cat(out, dim=1)  # [N, |Ct|, H, W]
        return x_o

    def _init_classifier(self):
        # Random Initialization
        for m in self.cls.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def freeze_bn(self, affine_freeze=False):
        if self.freeze_all_bn:
            for m in self.modules():
                if self.norm_act == 'bn_sync':
                    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        m.eval()
                        if affine_freeze is True:
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
                elif self.norm_act == 'iabn_sync':
                    from inplace_abn import ABN, InPlaceABNSync
                    if isinstance(m, (ABN)):
                        m.eval()
                        if affine_freeze is True:
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
            
        elif self.freeze_backbone_bn:
            for m in self.backbone.modules():
                if self.norm_act == 'bn_sync':
                    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        m.eval()
                        if affine_freeze is True:
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
                elif self.norm_act == 'iabn_sync':
                    from inplace_abn import ABN, InPlaceABNSync
                    if isinstance(m, (ABN)):
                        m.eval()
                        if affine_freeze is True:
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False

    def freeze_dropout(self):
        for m in self.modules():
            if isinstance(m, (nn.Dropout)):
                m.eval()

    def _load_pretrained_model(self, pretrained_path):
        pretrain_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
        self.load_state_dict(pretrain_dict['state_dict'], strict=False)

    def _set_bn_momentum(self, model=None, momentum=0.1):
        if model is not None:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = momentum
        else:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = momentum

    def get_backbone_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
                elif isinstance(m[1], (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if not (self.freeze_all_bn or self.freeze_backbone_bn):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_aspp_params(self):
        modules = [self.aspp]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
                elif isinstance(m[1], (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if not (self.freeze_all_bn):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_classifer_params(self):
        modules = [self.cls]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_old_classifer_params(self):
        modules = [self.cls[i] for i in range(0, len(self.cls) - 1)]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_new_classifer_params(self):
        modules = [self.cls[len(self.cls) - 1]]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
