import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry


class SAMSeg(nn.Module):
    def __init__(self, checkpoint: str, model_type: str = 'vit_h', img_size: int = 1024):
        super().__init__()
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.img_size = img_size
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = False
        n_blocks = len(self.sam.image_encoder.blocks)
        for block in self.sam.image_encoder.blocks[n_blocks - 4:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.sam.image_encoder.neck.parameters():
            p.requires_grad = True

    def _boxes_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract tight bounding boxes from binary masks.

        mask: [B, 1, H, W]  returns [B, 4] in pixel coords (x1, y1, x2, y2)
        Falls back to full-image box when mask is empty.
        """
        B, _, H, W = mask.shape
        boxes = torch.zeros(B, 4, dtype=torch.float32, device=mask.device)
        for i in range(B):
            m = mask[i, 0]
            if m.sum() > 0:
                rows = m.any(dim=1).nonzero(as_tuple=True)[0]
                cols = m.any(dim=0).nonzero(as_tuple=True)[0]
                boxes[i] = torch.tensor(
                    [cols[0].item(), rows[0].item(), cols[-1].item(), rows[-1].item()],
                    dtype=torch.float32, device=mask.device,
                )
            else:
                boxes[i] = torch.tensor([0, 0, W - 1, H - 1],
                                         dtype=torch.float32, device=mask.device)
        return boxes

    def forward(self, imgs: torch.Tensor, gt_mask: torch.Tensor = None) -> torch.Tensor:
        B, _, H, W = imgs.shape

        image_embeddings = self.sam.image_encoder(imgs)

        if gt_mask is not None:
            boxes = self._boxes_from_mask(gt_mask)
        else:
            boxes = torch.tensor(
                [[0, 0, W - 1, H - 1]], dtype=torch.float32, device=imgs.device
            ).repeat(B, 1)

        mask_list = []
        for i in range(B):
            sparse_emb, dense_emb = self.sam.prompt_encoder(
                points=None, boxes=boxes[i:i + 1], masks=None,
            )
            low_res_mask, _ = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i:i + 1],
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            mask_list.append(low_res_mask)

        low_res_masks = torch.cat(mask_list, dim=0)
        return F.interpolate(low_res_masks, size=(H, W), mode='bilinear', align_corners=False)
