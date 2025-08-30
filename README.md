## 数据准备(Data Preparation)

你需要准备以下数据文件并放入`Data`文件夹：

- 轨迹数据文件（如：`trajectory.csv`）
- POI 数据文件（如：`poi.csv`）

## 使用步骤

### 1. 数据检查

在运行模型前，请确保您的数据包含以下必要特征列：

- **轨迹数据必需字段**：轨迹 ID、时间戳、经纬度坐标
- **POI 数据必需字段**：POI 名称、类别、经纬度坐标

### 2. 运行半监督学习脚本

进入 SemiSupervised 目录，按照脚本内的示例命令运行：

```bash
python .\SemiSupervised\SemiModulesTrain.py \
    --tsv ./SemiSupervised/SemiData.tsv \
    --out_model ./SemiSupervised/traj_anomaly_rf.joblib \
    --metrics_json ./SemiSupervised/traj_anomaly_metrics.json \
    --featimp_csv ./SemiSupervised/traj_anomaly_feature_importance.csv
```

### 3. 数据预处理

返回 Data 目录，运行预处理脚本：

```bash
python Data/DataProcessPipline.py \
  --poi ./Data/CombinedPOI.csv \
  --geolife ./Data/Geolife.csv \
  --model ./SemiSupervised/traj_anomaly_rf.joblib \
  --out-pkl ./Data/beijing_geolife_incident_strict.pkl \
  --out-labels ./Data/beijing_geolife_labels_strict.csv \
  --resolution 10 --max-kmh 110 --max-idle-min 60 --min-points 10 \
  --ratio 0.3 --alpha 4.0 --beta 0.15 --base-scale 0.9 --poisson
```

### 4. 运行主模型

使用 unified_pipeline 中的演示方法运行完整模型：

```bash
python unified_pipeline.py \
  --models patchlstm \
  --data ./Data/beijing_geolife_incident_strict.pkl \
  --batch 32 --max_len 120 --max_seq 16 \
  --epochs 100 --patience 8 \
  --loss focal \
  --outdir ./checkpoints --results_csv ./batch_results.csv
```

## 文件结构

```
├── Data/
│   ├── trajectory.csv    # 轨迹数据（需用户提供）
│   ├── poi.csv           # POI数据（需用户提供）
│   └── DataProcessPipline.py
├── modules/
│   ├── Componet.py
│   └── PatchLSTM.py
├── SemiSupervised/
│   ├── SemiData.tsv
│   └── semi_supervised_processing.py
├── utils/
│   ├── DataLoader.py
│   └── utils.py
├── unified_pipeline.py
└── README.md

```

## 获取帮助

如遇到问题，请检查：

1. 数据格式是否符合要求
2. 所有依赖包是否已正确安装
3. 控制台输出的错误信息

---
