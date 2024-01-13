python main.py --model deit_small_MCTformerV1_patch16_224 \
                --data-set VOC12MS \
                --scales 1.0 \
                --img-list /data/c425/tjf/mct_mod33/voc12/ \
                --data-path /data/c425/tjf/datasets/VOC2012/ \
                --output_dir /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_1_sparsity_10 \
                --resume /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_1_sparsity_10/checkpoint_best.pth \
                --gen_attention_maps \
                --attention-type fused \
                --layer-index 3 \
                --visualize-cls-attn \
                --patch-attn-refine \
                --sparsity 0.1 \
                --cam-npy-dir /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_1_sparsity_10/attn-patchrefine-cpb-npy


python main.py --model deit_small_MCTformerV1_patch16_224 \
                --data-set VOC12MS \
                --scales 1.0 \
                --img-list /data/c425/tjf/mct_mod33/voc12/ \
                --data-path /data/c425/tjf/datasets/VOC2012/ \
                --output_dir /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_2_sparsity_10 \
                --resume /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_2_sparsity_10/checkpoint_best.pth \
                --gen_attention_maps \
                --attention-type fused \
                --layer-index 3 \
                --visualize-cls-attn \
                --patch-attn-refine \
                --sparsity 0.1 \
                --cam-npy-dir /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_2_sparsity_10/attn-patchrefine-cpb-npy

python main.py --model deit_small_MCTformerV1_patch16_224 \
                --data-set VOC12MS \
                --scales 1.0 \
                --img-list /data/c425/tjf/mct_mod33/voc12/ \
                --data-path /data/c425/tjf/datasets/VOC2012/ \
                --output_dir /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_3_sparsity_10 \
                --resume /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_3_sparsity_10/checkpoint_best.pth \
                --gen_attention_maps \
                --attention-type fused \
                --layer-index 3 \
                --visualize-cls-attn \
                --patch-attn-refine \
                --sparsity 0.1 \
                --cam-npy-dir /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_3_sparsity_10/attn-patchrefine-cpb-npy


python main.py --model deit_small_MCTformerV1_patch16_224 \
                --data-set VOC12MS \
                --scales 1.0 \
                --img-list /data/c425/tjf/mct_mod33/voc12/ \
                --data-path /data/c425/tjf/datasets/VOC2012/ \
                --output_dir /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_4_sparsity_10 \
                --resume /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_4_sparsity_10/checkpoint_best.pth \
                --gen_attention_maps \
                --attention-type fused \
                --layer-index 3 \
                --visualize-cls-attn \
                --patch-attn-refine \
                --sparsity 0.1 \
                --cam-npy-dir /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_4_sparsity_10/attn-patchrefine-cpb-npy

python main.py --model deit_small_MCTformerV1_patch16_224 \
                --data-set VOC12MS \
                --scales 1.0 \
                --img-list /data/c425/tjf/mct_mod33/voc12/ \
                --data-path /data/c425/tjf/datasets/VOC2012/ \
                --output_dir /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_5_sparsity_10 \
                --resume /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_5_sparsity_10/checkpoint_best.pth \
                --gen_attention_maps \
                --attention-type fused \
                --layer-index 3 \
                --visualize-cls-attn \
                --patch-attn-refine \
                --sparsity 0.1 \
                --cam-npy-dir /data/c425/tjf/mct_mod33/new_results/block_10_alpha_0_5_sparsity_10/attn-patchrefine-cpb-npy