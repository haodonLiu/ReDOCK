# 蛋白质对接软件

一个用于蛋白质-蛋白质对接的预对齐和构象搜索软件，支持GPU加速计算，专为研究蛋白质相互作用设计。

## 功能特性

1. **PDB文件基本处理**：解析PDB文件，显示统计信息，支持文件转换
2. **蛋白质坐标直接平动**：支持X、Y、Z三个方向的精确坐标平动
3. **蛋白质预对齐**：将受体和配体的关键中心对齐到Z轴，保留原有几何关系
4. **对接构象搜索**：基于旋转搜索的对接构象生成，支持第二轮旋转搜索
5. **GPU加速计算**：支持CUDA GPU加速，提高计算效率
6. **结果可视化**：输出预对齐结果和所有构象，便于后续分析
7. **项目化管理**：为每个任务创建独立文件夹，保存所有结果文件
8. **多构象评分**：使用范德华力、静电力、溶剂惩罚和距离惩罚计算构象得分
9. **HETATM记录支持**：确保HETATM记录能随着蛋白一起移动
10. **输出构象数量控制**：支持指定输出的构象数量
11. **安全机制**：添加安全计数器防止无限循环

## 安装要求

- Python 3.8+
- PyTorch 1.10+（支持CUDA GPU）
- numpy
- tqdm（用于显示进度条）
- XML解析（内置）

## 快速开始

### 安装依赖

```bash
pip install torch numpy tqdm
```

### 基本使用

```bash
# 解析PDB文件
python -m src.main data/structure/PP5_CD.pdb

# 预对齐蛋白质
python -m src.main --prealign --receptor data/structure/PP5_CD.pdb --ligand data/structure/triP-KD_AKT1.pdb \
              --receptor-residues B:184,B:185 --ligand-residues A:144,A:145 --max-dist 5.0

# 对接构象搜索
python -m src.main --dock --receptor data/structure/PP5_CD.pdb --ligand data/structure/triP-KD_AKT1.pdb \
              --receptor-residues B:184,B:185 --ligand-residues A:144,A:145 --max-dist 5.0 \
              --num-rotations 10 --force-field data/force_field/ff14SB.xml \
              --solvent-penalty 0.1 --distance-penalty 0.5 --num-output-confs 10
```

## 使用方法

### 1. 基本PDB文件处理

```bash
python -m src.main <pdb_file_path> [output_file_path] [--debug/-d]
```

**功能**：解析PDB文件并显示统计信息，可选择将解析后的数据写入新文件

### 2. 蛋白质坐标直接平动

```bash
python -m src.main --translate <pdb_file_path> [--dx <value>] [--dy <value>] [--dz <value>] [output_file_path] [--debug/-d]
```

**参数说明**：
- `--dx <value>`：X轴平动距离（Å），默认：0.0
- `--dy <value>`：Y轴平动距离（Å），默认：0.0
- `--dz <value>`：Z轴平动距离（Å），默认：0.0

**功能**：直接平动蛋白质坐标，支持三个方向的平动

### 3. 蛋白质预对齐

```bash
python -m src.main --prealign --receptor <receptor_file> --ligand <ligand_file> \
              --receptor-residues <residues> --ligand-residues <residues> [--max-dist <distance>] \
              [--force-field <file>] [-o <output_file>] [--gpu] [--debug/-d]
```

**参数说明**：
- `--receptor <file>`：受体蛋白PDB文件路径
- `--ligand <file>`：配体蛋白PDB文件路径
- `--receptor-residues <list>`：受体残基组，格式：链名:残基号,链名:残基号（如：B:184,B:185）
- `--ligand-residues <list>`：配体残基组，格式同上
- `--max-dist <distance>`：最大搜索距离，默认：5.0 Å
- `--force-field <file>`：力场XML文件路径，用于能量计算
- `-o <file>`：输出文件路径
- `--gpu`：启用GPU加速

**功能**：预对齐受体和配体蛋白质，将四个中心（受体蛋白中心、受体基团组中心、配体蛋白中心、配体组中心）对齐到Z轴

### 4. 蛋白质对接构象搜索

```bash
python -m src.main --dock --receptor <receptor_file> --ligand <ligand_file> \
              --receptor-residues <residues> --ligand-residues <residues> [--max-dist <distance>] \
              [--num-rotations <number>] [--force-field <file>] [--step-size <value>] \
              [--solvent-penalty <value>] [--distance-penalty <value>] [--optimal-distance <value>] \
              [--num-output-confs <number>] [-o <output_file>] [--save-all] [--gpu] [--debug/-d]
```

**参数说明**：
- `--num-rotations <number>`：旋转次数，默认：36次（每次10度）
- `--force-field <file>`：力场XML文件路径，用于能量计算
- `--step-size <value>`：中间构象生成的步长，默认：1.0 Å
- `--solvent-penalty <value>`：溶剂惩罚系数，默认：0.1 kcal/mol per contact
- `--distance-penalty <value>`：距离惩罚系数，默认：0.5 kcal/mol per Å
- `--optimal-distance <value>`：最优中心距离，默认：10.0 Å
- `--num-output-confs <number>`：输出构象数量，默认：10
- `--save-all`：保存所有生成的构象，默认：False

**功能**：搜索蛋白质对接构象，生成并评分多个构象，输出最优结果

## 项目结构

```
protein_dock/
├── src/                 # 源代码目录
│   ├── core/            # 核心功能模块
│   │   ├── alignment.py      # 蛋白质对齐简化算法
│   │   ├── coordinate_manager.py  # 坐标管理和变换（平移+旋转）
│   │   ├── docking.py       # 对接和预对齐算法
│   │   ├── energy_calculator.py  # 能量计算
│   │   └── statistics.py    # 统计信息计算
│   ├── io/              # 输入输出模块
│   │   ├── parser.py        # PDB文件解析
│   │   └── writer.py        # PDB文件写入
│   ├── models/          # 数据模型模块
│   │   ├── atom.py          # 原子模型
│   │   ├── chain.py         # 链模型
│   │   ├── force_field.py    # 力场模型
│   │   ├── residue.py        # 残基模型
│   │   └── structure.py     # 蛋白质结构模型
│   ├── utils/           # 工具模块
│   │   ├── common.py        # 通用工具函数
│   │   └── logger.py        # 日志管理
│   ├── cli.py            # 命令行界面
│   ├── main.py           # 主入口文件
│   └── __init__.py       # 包初始化文件
├── data/                # 数据目录
│   ├── force_field/      # 力场文件
│   │   ├── ff14SB.xml      # 力场参数文件
│   │   └── tip3p_standard.xml  # 溶剂参数文件
│   └── structure/        # 示例PDB文件
│       ├── PP5_CD.pdb       # 受体蛋白示例
│       └── triP-KD_AKT1.pdb  # 配体蛋白示例
├── dock_config.json     # 对接配置文件
├── README.md            # 项目说明文档
└── requirements.txt     # 依赖文件
```

## 算法说明

### 预对齐算法

预对齐算法确保：
1. 受体组中心、受体中心、配体中心、配体组中心共线
2. 受体中心→受体组中心方向与配体中心→配体组中心方向相反
3. 受体组中心和配体组中心间距为指定max_dist值
4. 所有四个中心精确位于Z轴上
5. 保留蛋白质原有几何关系

算法步骤：
1. 平移受体，将受体组中心移至原点
2. 旋转受体，使受体中心与受体组中心连线沿Z轴负方向
3. 平移配体，将配体组中心移至指定位置
4. 旋转配体，使配体中心与配体组中心连线沿Z轴正方向
5. 微调确保所有中心精确位于Z轴

### 蛋白质对齐简化算法

新的蛋白质对齐简化算法（位于src/core/alignment.py）提供了更高效、更精确的对齐功能：

1. **坐标系建立**：
   - 将受体蛋白质中心设为坐标系统原点(0,0,0)
   - 定义从坐标原点到受体目标组中心的向量为正z轴方向

2. **配体蛋白质对齐**：
   - 计算配体蛋白质中心点和其目标组中心点
   - 调整配体蛋白质，使其中心到目标组中心的连线与z轴精确对齐
   - 确保配体目标组中心沿z轴方向朝向受体目标组

3. **实现特点**：
   - 使用PyTorch进行高效的张量运算
   - 提供向量计算工具，用于确定关键组之间的空间关系
   - 实现旋转和平移变换，实现指定的对齐
   - 通过验证距离和轴对齐来确保对齐准确性

### 构象搜索算法

1. **中间构象生成**：通过逐渐减小距离生成多个中间构象，避免死循环
2. **旋转搜索**：绕Z轴旋转配体，生成指定数量的构象
3. **系统抽样**：对最佳构象进行系统抽样，找到更优构象
4. **第二轮旋转搜索**：对系统抽样找到的更优构象进行第二轮旋转搜索
5. **能量计算**：使用PyTorch计算每个构象的评分（范德华力+静电力+溶剂惩罚+距离惩罚）
6. **结果筛选**：按评分排序，输出指定数量的最优构象

### 能量计算算法

能量计算包括以下组件：
1. **范德华力**：使用Lennard-Jones势能计算
2. **静电力**：使用库仑势能计算
3. **溶剂惩罚**：基于原子接触表面面积的溶剂可及性惩罚
4. **距离惩罚**：基于残基组中心距离的惩罚

### HETATM支持

确保HETATM记录能随着蛋白一起移动，即使它们的chain_id与蛋白链不同。

## 开发流程

### 代码结构

1. **命令行界面**：`src/cli.py` 负责处理命令行参数和执行相应功能
2. **核心算法**：
   - `src/core/coordinate_manager.py`：坐标管理和变换（平移+旋转）
   - `src/core/docking.py`：对接和预对齐算法，支持第二轮旋转搜索
   - `src/core/energy_calculator.py`：能量计算，包括范德华力、静电力、溶剂惩罚和距离惩罚
   - `src/core/statistics.py`：统计信息计算
3. **输入输出**：`src/io/` 模块负责PDB文件的解析和写入
4. **数据模型**：`src/models/` 定义了原子、链、残基、力场和结构的数据模型
5. **工具函数**：`src/utils/` 提供通用工具函数和日志管理

### 贡献规范

1. 遵循现有代码风格和命名规范
2. 提交前运行代码检查和测试
3. 提交信息清晰，说明变更内容
4. 新功能需要添加相应的测试用例
5. 更新文档以反映变更
6. 确保所有HETATM记录能正确处理
7. 避免死循环，添加安全计数器
8. 确保所有参数能正确传递和使用

## 故障排除

### 常见问题

1. **GPU加速未生效**：
   - 检查是否安装了CUDA版本的PyTorch
   - 确保GPU驱动程序已正确安装
   - 使用`nvidia-smi`命令检查GPU状态

2. **构象生成速度慢**：
   - 减少旋转次数（--num-rotations）
   - 启用GPU加速（--gpu）
   - 增加max-dist距离，减少近距离构象

3. **预对齐失败**：
   - 检查残基组是否正确指定
   - 确保残基组存在于PDB文件中
   - 检查PDB文件格式是否正确

4. **HETATM记录未移动**：
   - 确保使用了正确的命令格式
   - 检查HETATM记录是否存在于PDB文件中

5. **评分参数无效**：
   - 确保使用了--solvent-penalty和--distance-penalty参数
   - 检查参数值是否合理

6. **死循环**：
   - 程序已添加安全计数器，会自动停止
   - 检查max-dist和min-distance参数设置

## 示例数据

项目中包含示例PDB文件，位于`data/structure/`目录下：
- `PP5_CD.pdb`：受体蛋白示例
- `triP-KD_AKT1.pdb`：配体蛋白示例

力场文件位于`data/force_field/`目录下：
- `ff14SB.xml`：力场参数文件

可以使用这些示例文件测试软件功能：

```bash
python -m src.main --prealign --receptor data/structure/PP5_CD.pdb --ligand data/structure/triP-KD_AKT1.pdb \
              --receptor-residues B:184,B:185 --ligand-residues A:144,A:145 --max-dist 5.0 -o prealigned.pdb
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系我们：

- 项目地址：https://github.com/yourusername/protein_dock
- 邮箱：your.email@example.com

## 致谢

感谢所有为项目做出贡献的开发者和用户！