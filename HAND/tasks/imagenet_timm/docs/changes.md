# Recent Changes

### July 27, 2022
* All runtime benchmark and validation result csv files are up-to-date!
* A few more weights & model defs added:
  * `darknetaa53` -  79.8 @ 256, 80.5 @ 288
  * `convnext_nano` - 80.8 @ 224, 81.5 @ 288
  * `cs3sedarknet_l` - 81.2 @ 256, 81.8 @ 288
  * `cs3darknet_x` - 81.8 @ 256, 82.2 @ 288
  * `cs3sedarknet_x` - 82.2 @ 256, 82.7 @ 288
  * `cs3edgenet_x` - 82.2 @ 256, 82.7 @ 288
  * `cs3se_edgenet_x` - 82.8 @ 256, 83.5 @ 320
* `cs3*` weights above all trained on TPU w/ `bits_and_tpu` branch. Thanks to TRC program!
* Add output_stride=8 and 16 support to ConvNeXt (dilation)
* deit3 models not being able to resize pos_emb fixed
* Version 0.6.7 PyPi release (/w above bug fixes and new weighs since 0.6.5)

### July 8, 2022
More models, more fixes
* Official research models (w/ weights) added:
  * EdgeNeXt from (https://github.com/mmaaz60/EdgeNeXt)
  * MobileViT-V2 from (https://github.com/apple/ml-cvnets)
  * DeiT III (Revenge of the ViT) from (https://github.com/facebookresearch/deit)
* My own models:
  * Small `ResNet` defs added by request with 1 block repeats for both basic and bottleneck (resnet10 and resnet14)
  * `CspNet` refactored with dataclass config, simplified CrossStage3 (`cs3`) option. These are closer to YOLO-v5+ backbone defs.
  * More relative position vit fiddling. Two `srelpos` (shared relative position) models trained, and a medium w/ class token.
  * Add an alternate downsample mode to EdgeNeXt and train a `small` model. Better than original small, but not their new USI trained weights.
* My own model weight results (all ImageNet-1k training)
  * `resnet10t` - 66.5 @ 176, 68.3 @ 224
  * `resnet14t` - 71.3 @ 176, 72.3 @ 224
  * `resnetaa50` - 80.6 @ 224 , 81.6 @ 288
  * `darknet53` -  80.0 @ 256, 80.5 @ 288
  * `cs3darknet_m` - 77.0 @ 256, 77.6 @ 288
  * `cs3darknet_focus_m` - 76.7 @ 256, 77.3 @ 288
  * `cs3darknet_l` - 80.4 @ 256, 80.9 @ 288
  * `cs3darknet_focus_l` - 80.3 @ 256, 80.9 @ 288
  * `vit_srelpos_small_patch16_224` - 81.1 @ 224, 82.1 @ 320
  * `vit_srelpos_medium_patch16_224` - 82.3 @ 224, 83.1 @ 320
  * `vit_relpos_small_patch16_cls_224` - 82.6 @ 224, 83.6 @ 320
  * `edgnext_small_rw` - 79.6 @ 224, 80.4 @ 320
* `cs3`, `darknet`, and `vit_*relpos` weights above all trained on TPU thanks to TRC program! Rest trained on overheating GPUs.
* Hugging Face Hub support fixes verified, demo notebook TBA
* Pretrained weights / configs can be loaded externally (ie from local disk) w/ support for head adaptation.
* Add support to change image extensions scanned by `timm` datasets/parsers. See (https://github.com/rwightman/pytorch-image-models/pull/1274#issuecomment-1178303103)
* Default ConvNeXt LayerNorm impl to use `F.layer_norm(x.permute(0, 2, 3, 1), ...).permute(0, 3, 1, 2)` via `LayerNorm2d` in all cases. 
  * a bit slower than previous custom impl on some hardware (ie Ampere w/ CL), but overall fewer regressions across wider HW / PyTorch version ranges. 
  * previous impl exists as `LayerNormExp2d` in `models/layers/norm.py`
* Numerous bug fixes
* Currently testing for imminent PyPi 0.6.x release
* LeViT pretraining of larger models still a WIP, they don't train well / easily without distillation. Time to add distill support (finally)?
* ImageNet-22k weight training + finetune ongoing, work on multi-weight support (slowly) chugging along (there are a LOT of weights, sigh) ...

### May 13, 2022
* Official Swin-V2 models and weights added from (https://github.com/microsoft/Swin-Transformer). Cleaned up to support torchscript.
* Some refactoring for existing `timm` Swin-V2-CR impl, will likely do a bit more to bring parts closer to official and decide whether to merge some aspects.
* More Vision Transformer relative position / residual post-norm experiments (all trained on TPU thanks to TRC program)
  * `vit_relpos_small_patch16_224` - 81.5 @ 224, 82.5 @ 320 -- rel pos, layer scale, no class token, avg pool
  * `vit_relpos_medium_patch16_rpn_224` - 82.3 @ 224, 83.1 @ 320 -- rel pos + res-post-norm, no class token, avg pool
  * `vit_relpos_medium_patch16_224` - 82.5 @ 224, 83.3 @ 320 -- rel pos, layer scale, no class token, avg pool
  * `vit_relpos_base_patch16_gapcls_224` - 82.8 @ 224, 83.9 @ 320 -- rel pos, layer scale, class token, avg pool (by mistake)
* Bring 512 dim, 8-head 'medium' ViT model variant back to life (after using in a pre DeiT 'small' model for first ViT impl back in 2020)
* Add ViT relative position support for switching btw existing impl and some additions in official Swin-V2 impl for future trials
* Sequencer2D impl (https://arxiv.org/abs/2205.01972), added via PR from author (https://github.com/okojoalg)

### May 2, 2022
* Vision Transformer experiments adding Relative Position (Swin-V2 log-coord) (`vision_transformer_relpos.py`) and Residual Post-Norm branches (from Swin-V2) (`vision_transformer*.py`)
  * `vit_relpos_base_patch32_plus_rpn_256` - 79.5 @ 256, 80.6 @ 320 -- rel pos + extended width + res-post-norm, no class token, avg pool
  * `vit_relpos_base_patch16_224` - 82.5 @ 224, 83.6 @ 320 -- rel pos, layer scale, no class token, avg pool
  * `vit_base_patch16_rpn_224` - 82.3 @ 224 -- rel pos + res-post-norm, no class token, avg pool
* Vision Transformer refactor to remove representation layer that was only used in initial vit and rarely used since with newer pretrain (ie `How to Train Your ViT`)
* `vit_*` models support removal of class token, use of global average pool, use of fc_norm (ala beit, mae).

### April 22, 2022
* `timm` models are now officially supported in [fast.ai](https://www.fast.ai/)! Just in time for the new Practical Deep Learning course. `timmdocs` documentation link updated to [timm.fast.ai](http://timm.fast.ai/).
* Two more model weights added in the TPU trained [series](https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-tpu-weights). Some In22k pretrain still in progress.
  * `seresnext101d_32x8d` - 83.69 @ 224, 84.35 @ 288
  * `seresnextaa101d_32x8d` (anti-aliased w/ AvgPool2d) - 83.85 @ 224, 84.57 @ 288

### March 23, 2022
* Add `ParallelBlock` and `LayerScale` option to base vit models to support model configs in [Three things everyone should know about ViT](https://arxiv.org/abs/2203.09795)
* `convnext_tiny_hnf` (head norm first) weights trained with (close to) A2 recipe, 82.2% top-1, could do better with more epochs.

### March 21, 2022
* Merge `norm_norm_norm`. **IMPORTANT** this update for a coming 0.6.x release will likely de-stabilize the master branch for a while. Branch [`0.5.x`](https://github.com/rwightman/pytorch-image-models/tree/0.5.x) or a previous 0.5.x release can be used if stability is required.
* Significant weights update (all TPU trained) as described in this [release](https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-tpu-weights)
  * `regnety_040` - 82.3 @ 224, 82.96 @ 288
  * `regnety_064` - 83.0 @ 224, 83.65 @ 288
  * `regnety_080` - 83.17 @ 224, 83.86 @ 288
  * `regnetv_040` - 82.44 @ 224, 83.18 @ 288   (timm pre-act)
  * `regnetv_064` - 83.1 @ 224, 83.71 @ 288   (timm pre-act)
  * `regnetz_040` - 83.67 @ 256, 84.25 @ 320
  * `regnetz_040h` - 83.77 @ 256, 84.5 @ 320 (w/ extra fc in head)
  * `resnetv2_50d_gn` - 80.8 @ 224, 81.96 @ 288 (pre-act GroupNorm)
  * `resnetv2_50d_evos` 80.77 @ 224, 82.04 @ 288 (pre-act EvoNormS)
  * `regnetz_c16_evos`  - 81.9 @ 256, 82.64 @ 320 (EvoNormS)
  * `regnetz_d8_evos`  - 83.42 @ 256, 84.04 @ 320 (EvoNormS)
  * `xception41p` - 82 @ 299   (timm pre-act)
  * `xception65` -  83.17 @ 299
  * `xception65p` -  83.14 @ 299   (timm pre-act)
  * `resnext101_64x4d` - 82.46 @ 224, 83.16 @ 288
  * `seresnext101_32x8d` - 83.57 @ 224, 84.270 @ 288
  * `resnetrs200` - 83.85 @ 256, 84.44 @ 320
* HuggingFace hub support fixed w/ initial groundwork for allowing alternative 'config sources' for pretrained model definitions and weights (generic local file / remote url support soon)
* SwinTransformer-V2 implementation added. Submitted by [Christoph Reich](https://github.com/ChristophReich1996). Training experiments and model changes by myself are ongoing so expect compat breaks.
* Swin-S3 (AutoFormerV2) models / weights added from https://github.com/microsoft/Cream/tree/main/AutoFormerV2
* MobileViT models w/ weights adapted from https://github.com/apple/ml-cvnets
* PoolFormer models w/ weights adapted from https://github.com/sail-sg/poolformer
* VOLO models w/ weights adapted from https://github.com/sail-sg/volo
* Significant work experimenting with non-BatchNorm norm layers such as EvoNorm, FilterResponseNorm, GroupNorm, etc
* Enhance support for alternate norm + act ('NormAct') layers added to a number of models, esp EfficientNet/MobileNetV3, RegNet, and aligned Xception
* Grouped conv support added to EfficientNet family
* Add 'group matching' API to all models to allow grouping model parameters for application of 'layer-wise' LR decay, lr scale added to LR scheduler
* Gradient checkpointing support added to many models
* `forward_head(x, pre_logits=False)` fn added to all models to allow separate calls of `forward_features` + `forward_head`
* All vision transformer and vision MLP models update to return non-pooled / non-token selected features from `foward_features`, for consistency with CNN models, token selection or pooling now applied in `forward_head`

### Feb 2, 2022
* [Chris Hughes](https://github.com/Chris-hughes10) posted an exhaustive run through of `timm` on his blog yesterday. Well worth a read. [Getting Started with PyTorch Image Models (timm): A Practitioner’s Guide](https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055)
* I'm currently prepping to merge the `norm_norm_norm` branch back to master (ver 0.6.x) in next week or so.
  * The changes are more extensive than usual and may destabilize and break some model API use (aiming for full backwards compat). So, beware `pip install git+https://github.com/rwightman/pytorch-image-models` installs!
  * `0.5.x` releases and a `0.5.x` branch will remain stable with a cherry pick or two until dust clears. Recommend sticking to pypi install for a bit if you want stable.

### Jan 14, 2022
* Version 0.5.4 w/ release to be pushed to pypi. It's been a while since last pypi update and riskier changes will be merged to main branch soon....
* Add ConvNeXT models /w weights from official impl (https://github.com/facebookresearch/ConvNeXt), a few perf tweaks, compatible with timm features
* Tried training a few small (~1.8-3M param) / mobile optimized models, a few are good so far, more on the way...
  * `mnasnet_small` - 65.6 top-1
  * `mobilenetv2_050` - 65.9
  * `lcnet_100/075/050` - 72.1 / 68.8 / 63.1
  * `semnasnet_075` - 73
  * `fbnetv3_b/d/g` - 79.1 / 79.7 / 82.0
* TinyNet models added by [rsomani95](https://github.com/rsomani95)
* LCNet added via MobileNetV3 architecture

### Nov 22, 2021
* A number of updated weights anew new model defs
  * `eca_halonext26ts` - 79.5 @ 256
  * `resnet50_gn` (new) - 80.1 @ 224, 81.3 @ 288
  * `resnet50` - 80.7 @ 224, 80.9 @ 288 (trained at 176, not replacing current a1 weights as default since these don't scale as well to higher res, [weights](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1h2_176-001a1197.pth))
  * `resnext50_32x4d` - 81.1 @ 224, 82.0 @ 288
  * `sebotnet33ts_256` (new) - 81.2 @ 224
  * `lamhalobotnet50ts_256` - 81.5 @ 256
  * `halonet50ts` - 81.7 @ 256
  * `halo2botnet50ts_256` - 82.0 @ 256
  * `resnet101` - 82.0 @ 224, 82.8 @ 288
  * `resnetv2_101` (new) - 82.1 @ 224, 83.0 @ 288
  * `resnet152` - 82.8 @ 224, 83.5 @ 288
  * `regnetz_d8` (new) - 83.5 @ 256, 84.0 @ 320
  * `regnetz_e8` (new) - 84.5 @ 256, 85.0 @ 320
* `vit_base_patch8_224` (85.8 top-1) & `in21k` variant weights added thanks [Martins Bruveris](https://github.com/martinsbruveris)
* Groundwork in for FX feature extraction thanks to [Alexander Soare](https://github.com/alexander-soare)
  * models updated for tracing compatibility (almost full support with some distlled transformer exceptions)

### Oct 19, 2021
* ResNet strikes back (https://arxiv.org/abs/2110.00476) weights added, plus any extra training components used. Model weights and some more details here (https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-rsb-weights)
* BCE loss and Repeated Augmentation support for RSB paper
* 4 series of ResNet based attention model experiments being added (implemented across byobnet.py/byoanet.py). These include all sorts of attention, from channel attn like SE, ECA to 2D QKV self-attention layers such as Halo, Bottlneck, Lambda. Details here (https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-attn-weights)
* Working implementations of the following 2D self-attention modules (likely to be differences from paper or eventual official impl):
  * Halo (https://arxiv.org/abs/2103.12731)
  * Bottleneck Transformer (https://arxiv.org/abs/2101.11605)
  * LambdaNetworks (https://arxiv.org/abs/2102.08602)
* A RegNetZ series of models with some attention experiments (being added to). These do not follow the paper (https://arxiv.org/abs/2103.06877) in any way other than block architecture, details of official models are not available. See more here (https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-attn-weights)
* ConvMixer (https://openreview.net/forum?id=TVHS5Y4dNvM), CrossVit (https://arxiv.org/abs/2103.14899), and BeiT (https://arxiv.org/abs/2106.08254) architectures + weights added
* freeze/unfreeze helpers by [Alexander Soare](https://github.com/alexander-soare)

### Aug 18, 2021
* Optimizer bonanza!
  * Add LAMB and LARS optimizers, incl trust ratio clipping options. Tweaked to work properly in PyTorch XLA (tested on TPUs w/ `timm bits` [branch](https://github.com/rwightman/pytorch-image-models/tree/bits_and_tpu/timm/bits))
  * Add MADGRAD from FB research w/ a few tweaks (decoupled decay option, step handling that works with PyTorch XLA)
  * Some cleanup on all optimizers and factory. No more `.data`, a bit more consistency, unit tests for all!
  * SGDP and AdamP still won't work with PyTorch XLA but others should (have yet to test Adabelief, Adafactor, Adahessian myself).
* EfficientNet-V2 XL TF ported weights added, but they don't validate well in PyTorch (L is better). The pre-processing for the V2 TF training is a bit diff and the fine-tuned 21k -> 1k weights are very sensitive and less robust than the 1k weights.
* Added PyTorch trained EfficientNet-V2 'Tiny' w/ GlobalContext attn weights. Only .1-.2 top-1 better than the SE so more of a curiosity for those interested.
