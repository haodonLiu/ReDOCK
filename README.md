# 蛋白质对接软件

一个用于蛋白质-蛋白质对接的构象搜索软件，支持GPU加速计算，专为研究蛋白质相互作用设计。

## 功能特性

1. **对接构象搜索**：基于沿z轴旋转的对接构象生成，找到能量最低的最优构象
2. **GPU加速计算**：支持CUDA GPU加速，提高计算效率
3. **多构象评分**：使用范德华力、静电力计算构象得分
4. **结果可视化**：输出最优构象，便于后续分析
5. **HETATM记录支持**：确保HETATM记录能随着蛋白一起移动
6. **安全机制**：添加错误处理和参数验证，提高程序鲁棒性

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
# 对接构象搜索
python -m src.main dock --receptor data/structure/PP5_CD.pdb --ligand data/structure/triP-KD_AKT1.pdb \
              --receptor-residues B:184,B:185 --ligand-residues A:144,A:145 --max-dist 5.0 \
              --num-rotations 10 --force-field data/force_field/ff14SB.xml \
              --solvent-penalty 0.1 --distance-penalty 0.5 --num-output-confs 10
```

## 使用方法

### 蛋白质对接构象搜索

```bash
python -m src.main dock --receptor <receptor_file> --ligand <ligand_file> \
              --receptor-residues <residues> --ligand-residues <residues> \
              [--max-dist <distance>] [--num-rotations <number>] [--step-size <step>] \
              [--solvent-penalty <coeff>] [--distance-penalty <coeff>] [--num-output-confs <num>] \
              [--force-field <file>] [-o <output_file>] [--save-all] [--gpu] [--debug/-d]
```

**参数说明**：
- `--receptor <file>`：受体蛋白PDB文件路径
- `--ligand <file>`：配体蛋白PDB文件路径
- `--receptor-residues <list>`：受体残基组，格式：链名:残基号,链名:残基号（如：B:184,B:185）
- `--ligand-residues <list>`：配体残基组，格式同上
- `--max-dist <distance>`：最大搜索距离，默认：5.0 Å
- `--num-rotations <number>`：旋转次数，默认：36次（每次10度）
- `--step-size <step>`：中间构象生成的步长（Å），默认：1.0
- `--solvent-penalty <coeff>`：溶剂惩罚系数，默认：0.1
- `--distance-penalty <coeff>`：距离惩罚系数，默认：0.5
- `--num-output-confs <num>`：输出构象数量，默认：10
- `--force-field <file>`：力场XML文件路径，用于能量计算
- `-o <file>`：输出文件路径
- `--save-all`：保存所有生成的构象
- `--gpu`：启用GPU加速

**功能**：搜索蛋白质对接构象，生成并评分多个构象，输出能量最低的最优构象

## 算法说明

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

## 示例数据

项目中包含示例PDB文件，位于`data/structure/`目录下：
- `PP5_CD.pdb`：受体蛋白示例
- `triP-KD_AKT1.pdb`：配体蛋白示例

力场文件位于`data/force_field/`目录下：
- `ff14SB.xml`：力场参数文件

可以使用这些示例文件测试软件功能：

```bash
# 使用JSON参数进行对接
python -m src.main dock --json '{"receptor_file":"data/structure/PP5_CD.pdb","ligand_file":"data/structure/triP-KD_AKT1.pdb","receptor_residues":["B:184","B:185"],"ligand_residues":["A:144","A:145"],"max_dist":5.0,"num_rotations":10,"gpu":true}'
```

## 许可证

MIT License