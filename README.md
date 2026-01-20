# 蛋白质对接软件

一个用于蛋白质-蛋白质对接的预对齐和构象搜索软件，支持GPU加速计算，专为研究蛋白质相互作用设计。

## 功能特性

1. **PDB文件基本处理**：解析PDB文件，显示统计信息，支持文件转换
2. **蛋白质坐标系统标准化**：自动计算几何中心并平移至原点，旋转使蛋白向量与z轴对齐
3. **蛋白质预对齐**：将受体和配体的关键中心对齐到Z轴，确保蛋白向量方向相反
4. **预对接流程**：实现两个蛋白质中心的重叠对齐，沿z轴调整距离至合适位置
5. **对接构象搜索**：基于沿z轴旋转的对接构象生成，找到能量最低的最优构象
6. **GPU加速计算**：支持CUDA GPU加速，提高计算效率
7. **结果可视化**：输出预对齐结果和最优构象，便于后续分析
8. **多构象评分**：使用范德华力、静电力计算构象得分
9. **HETATM记录支持**：确保HETATM记录能随着蛋白一起移动
10. **安全机制**：添加错误处理和参数验证，提高程序鲁棒性

## 安装要求

- Python 3.8+
- PyTorch 1.10+（支持CUDA GPU）
- numpy
- XML解析（内置）

## 快速开始

### 安装依赖

```bash
pip install torch numpy
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
              --receptor-residues <residues> --ligand-residues <residues> \
              [--num-rotations <number>] [--force-field <file>] \
              [-o <output_file>] [--gpu] [--debug/-d]
```

**参数说明**：
- `--num-rotations <number>`：旋转次数，默认：36次（每次10度）
- `--force-field <file>`：力场XML文件路径，用于能量计算

**功能**：搜索蛋白质对接构象，生成并评分多个构象，输出能量最低的最优构象

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

### 蛋白质坐标系统标准化

坐标系统标准化算法确保：
1. 蛋白质几何中心平移至原点
2. 蛋白向量（从几何中心指向指定残基组中心）与z轴对齐
3. 保留蛋白质原有几何关系

算法步骤：
1. 计算蛋白质几何中心
2. 平移蛋白质使几何中心至原点
3. 计算蛋白向量（从几何中心指向指定残基组中心）
4. 旋转蛋白质使蛋白向量与z轴对齐

### 预对接流程

预对接流程确保：
1. 受体和配体的几何中心重叠对齐
2. 受体蛋白向量沿正z轴方向
3. 配体蛋白向量沿负z轴方向（与受体方向相反）
4. 两个蛋白质中心距离逐步增大至合适位置

算法步骤：
1. 标准化受体坐标（对齐至正z轴）
2. 标准化配体坐标并旋转180度（对齐至负z轴）
3. 计算两蛋白质向量长度
4. 沿z轴方向逐步增大距离，直到中心距离 > 两蛋白向量长度之和 + 5Å

### 对接构象优化

对接构象优化算法：
1. **构象生成**：沿z轴旋转配体，生成指定数量的构象（默认36个，每10度一个）
2. **能量计算**：使用范德华力和静电力计算每个构象的能量
3. **最优构象选择**：找到并返回能量最低的构象

实现特点：
- 使用PyTorch进行高效的张量运算
- 支持GPU加速计算
- 详细的日志输出，便于监控构象搜索和能量评估过程
- 确保找到能量最低的最优构象

### 能量计算算法

能量计算包括以下组件：
1. **范德华力**：使用Lennard-Jones势能计算
2. **静电力**：使用库仑势能计算

实现特点：
- 使用PyTorch进行高效的张量运算
- 支持GPU加速计算
- 跳过氢原子计算，提高性能
- 应用距离截断和能量值裁剪，确保计算稳定性

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