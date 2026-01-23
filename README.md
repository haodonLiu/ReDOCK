# 蛋白质对接软件 - Scan Engine

一个用于蛋白质-蛋白质对接的专业构象搜索软件，以`scan_engine.py`为核心，支持GPU加速计算，专为研究蛋白质相互作用设计。

## 功能特性

1. **高效构象搜索**：基于两步策略（粗扫描+精扫描）的构象生成，提高搜索效率和精度
2. **六自由度扫描**：同时支持xyz方向平移和旋转，生成全面的构象空间
3. **专业IO模块**：增强的PDB文件解析和写入功能，支持多种输出格式
4. **GPU加速计算**：支持CUDA GPU加速，提高计算效率
5. **构象聚类**：基于六自由度特征的K-means聚类，减少冗余构象
6. **能量优化**：随机抖动优化以寻找最低能量状态
7. **距离验证**：基于残基组距离的构象验证，确保构象质量
8. **HETATM记录支持**：确保HETATM记录能随着蛋白一起移动
9. **安全机制**：添加错误处理和参数验证，提高程序鲁棒性

## 项目结构

```
protein_dock/
├── data/
│   ├── force_field/          # 力场参数文件
│   │   └── ff14SB.xml
│   └── structure/            # 示例PDB文件
│       ├── PP5_CD.pdb
│       └── triP-KD_AKT1.pdb
├── src/
│   ├── core/                 # 核心功能模块
│   │   ├── __init__.py
│   │   ├── scan_engine.py    # 主要扫描引擎（核心文件）
│   │   ├── coordinate_manager.py  # 坐标管理
│   │   └── energy_calculator.py   # 能量计算
│   ├── io/                   # 专业IO模块
│   │   ├── __init__.py
│   │   ├── parser.py         # PDB文件解析
│   │   └── writer.py         # PDB文件写入
│   ├── models/               # 数据模型
│   │   ├── __init__.py
│   │   ├── structure.py      # 蛋白质结构
│   │   ├── force_field.py     # 力场参数
│   │   ├── atom.py            # 原子模型
│   │   ├── chain.py           # 链模型
│   │   └── residue.py         # 残基模型
│   └── utils/                # 工具模块
│       ├── __init__.py
│       ├── logger.py          # 日志系统
│       └── structure_utils.py # 结构工具函数
├── run_scan_engine.py        # 主入口文件
├── input_config.json         # 配置文件
├── requirements.txt          # 依赖包
└── README.md                 # 文档
```

## 安装要求

- Python 3.8+
- PyTorch 1.10+（支持CUDA GPU）
- numpy
- scikit-learn（用于聚类）
- tqdm（用于进度显示）

## 快速开始

### 安装依赖

```bash
pip install torch numpy scikit-learn tqdm
```

### 基本使用

```bash
# 使用配置文件运行扫描引擎
python run_scan_engine.py
```

## 使用方法

### 配置文件说明

通过`input_config.json`文件配置扫描参数：

```json
{
  "debug": false,                  # 调试模式
  "receptor_file": "data\\structure\\triP-KD_AKT1.pdb",  # 受体PDB文件
  "ligand_file": "data\\structure\\PP5_CD.pdb",         # 配体PDB文件
  "receptor_residues": [           # 受体残基组
    "A:473"
  ],
  "ligand_residues": [             # 配体残基组
    "B:275",
    "B:427",
    "B:451",
    "C:501",
    "C:502"
  ],
  "min_dist": 6.0,                 # 最小残基组距离
  "max_dist": 8.0,                 # 最大残基组距离
  "num_rotations": 10,             # 旋转次数
  "step_size": 2.0                 # 平移步长
}
```

### 运行扫描引擎

```bash
# 使用默认配置文件
python run_scan_engine.py

# 或使用自定义配置文件
python run_scan_engine.py --config custom_config.json
```

## 算法说明

### 扫描引擎工作流程

1. **初始化**：加载受体和配体结构，计算残基组中心
2. **粗扫描**：使用较大步长进行全面的平移和旋转搜索
3. **构象聚类**：对粗扫描结果进行K-means聚类，选择每个聚类的代表构象
4. **精扫描**：在每个聚类代表构象周围进行小步长精细搜索
5. **构象优化**：对精扫描结果应用随机抖动，寻找最低能量状态
6. **结果输出**：保存最优构象到PDB文件

### 能量计算与验证

- **碰撞检测**：使用自定义碰撞分数，避免原子重叠
- **距离验证**：确保残基组距离在合理范围内
- **构象筛选**：基于距离标准和能量值筛选高质量构象

## 示例数据

项目中包含示例PDB文件，位于`data/structure/`目录下：
- `PP5_CD.pdb`：受体蛋白示例
- `triP-KD_AKT1.pdb`：配体蛋白示例

力场文件位于`data/force_field/`目录下：
- `ff14SB.xml`：力场参数文件

## 输出结果

扫描完成后，程序会：
1. 在控制台输出详细的扫描过程和结果统计
2. 生成`scan_results.pdb`文件，包含最优构象
3. 生成`coarse_cluster_*.pdb`文件，包含聚类后的构象

## 性能优化

- **GPU加速**：使用PyTorch张量运算和GPU加速，提高计算速度
- **批处理**：高效的批处理算法，减少内存使用
- **并行计算**：使用现代Python特性，优化计算流程
- **缓存机制**：避免重复计算，提高性能

## 许可证

MIT License