# 实现配体蛋白在三维空间的表面搜索优化流程

## 1. 问题分析
用户要求在旋转搜索之前添加一个流程，实现以下功能：
- 让蛋白可以在xy方向上随意移动
- 允许在z轴上运动
- 整体运动类似于沿着表面搜索
- 配体蛋白持续受到向受体蛋白目标基团的作用力
- 使得配体蛋白向量指向受体目标基团
- 不断搜索直到两个目标基团尽可能靠近
- 确保全程范德华力不大于1000
- 在运动过程中保持刚体

## 2. 解决方案

### 2.1 新增方法设计
在`Docking`类中新增`optimize_position`方法，实现以下功能：

#### 方法功能
- 计算受体目标基团的位置
- 计算配体蛋白向量
- 在三维空间中移动配体，使得配体向量指向受体目标基团
- 模拟配体受到向受体目标基团的作用力
- 沿着表面搜索最佳位置
- 确保全程范德华力不大于1000

#### 实现步骤
1. **初始化**：
   - 获取受体目标基团中心位置
   - 获取配体目标基团中心位置
   - 创建坐标管理器
   - 初始化搜索参数

2. **力导向优化**：
   - 计算从配体目标基团到受体目标基团的方向向量
   - 计算作用力大小（与距离相关）
   - 在作用力方向上移动配体
   - 每次移动后检查范德华力

3. **表面搜索**：
   - 如果范德华力超过阈值，调整运动方向
   - 沿着受体表面搜索可行路径
   - 保持配体向量始终指向受体目标基团

4. **精细优化**：
   - 在最佳位置附近进行精细搜索
   - 确保距离最小且范德华力不大于1000

### 2.2 流程集成
在`dock`方法中，在`align_proteins`之后、`search_conformations`之前添加对`optimize_position`方法的调用。

## 3. 代码修改

### 3.1 新增方法
在`src/core/docking.py`中添加`optimize_position`方法：

```python
def optimize_position(self) -> None:
    """
    Optimize ligand position in 3D space to align with receptor target group.
    Simulates ligand being pulled towards receptor target group while searching along the surface,
    ensuring van der Waals energy stays below 1000 and ligand remains rigid.
    """
    # 实现代码
```

### 3.2 修改dock方法
在`dock`方法中添加对`optimize_position`的调用：

```python
# Step 2: Align proteins
self.logger.section("Aligning Proteins")
self.align_proteins()

# Step 2.5: Optimize 3D position
self.logger.section("Optimizing 3D Position")
self.optimize_position()

# Step 3: Generate conformations through rotation
self.logger.section("Generating Conformations")
conformations = self.search_conformations(num_rotations)
```

## 4. 关键技术点

### 4.1 力导向算法
- 计算从配体到受体目标基团的方向向量
- 根据距离计算作用力大小（距离越远，作用力越大）
- 沿着作用力方向移动配体

### 4.2 表面搜索策略
- 当范德华力超过阈值时，调整运动方向
- 采用梯度下降法，在可行空间内搜索最佳路径
- 保持配体向量始终指向受体目标基团

### 4.3 范德华力约束
- 每次移动后计算范德华力
- 如果范德华力超过1000，调整运动方向或减小步长
- 确保整个搜索过程中范德华力不超过阈值

### 4.4 刚体保持
- 移动时保持配体的内部结构不变
- 只进行整体平移，不改变内部坐标关系

## 5. 预期效果

- 配体蛋白在三维空间中找到最佳位置
- 配体蛋白向量始终指向受体目标基团
- 两个目标基团尽可能靠近
- 全程范德华力不大于1000
- 运动过程中配体保持刚体特性
- 提高后续旋转搜索的效率和准确性