#!/bin/bash
# 一键运行所有 ResMLP 消融实验
# 用法: bash run_ablation_experiments.sh

# CONFIG="config/cifar10_resmlp_ablation.yaml"
CONFIG="config/sst2_resmlp_ablation.yaml"

echo "=========================================="
echo "ResMLP 消融实验 - 批量运行"
echo "=========================================="
echo ""

# 定义所有实验变体
variants=("baseline" "attn" "no_affine" "no_layerscale" "no_cross_patch" "full")

# 变体描述
declare -A descriptions
descriptions["baseline"]="原始 ResMLP"
descriptions["attn"]="ResMLP + Attention"
descriptions["no_affine"]="ResMLP + LayerNorm"
descriptions["no_layerscale"]="ResMLP - LayerScale"
descriptions["no_cross_patch"]="ResMLP - CrossPatch"
descriptions["full"]="ResMLP Full (类 ViT)"

# 运行每个实验
for variant in "${variants[@]}"
do
    echo "=========================================="
    echo "🧪 实验: ${descriptions[$variant]}"
    echo "=========================================="
    echo ""
    
    # 运行训练
    python train_resmlp_ablation_sst2.py --config $CONFIG --variant $variant
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 实验 $variant 完成"
        echo ""
    else
        echo ""
        echo "❌ 实验 $variant 失败"
        echo ""
        exit 1
    fi
done

echo "=========================================="
echo "🎉 所有实验完成！"
echo "=========================================="
# echo ""
# echo "运行以下命令查看结果对比："
# echo "python compare_ablation_results.py"