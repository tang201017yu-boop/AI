# 训练完成详情查看功能实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 用户可以查看已完成训练的详细信息，包括训练指标、曲线图表、验证结果图片，并支持导出模型、推理测试、继续训练。

**Architecture:** 后端新增实验详情 API，前端新增详情页，使用 experiments.json 作为数据源（解决系统重启后 training_tasks 为空的问题）。

**Tech Stack:** FastAPI (后端), React + Recharts (前端), YOLO (模型训练)

---

## Task 1: 修复状态同步 - 训练完成时更新 experiments.json

**Files:**
- Modify: `backend/core/yolo_engine.py:763-785`

**Step 1: 找到 finish_callback 函数**

在 yolo_engine.py 的 finish_callback 中添加同步更新 experiments.json 的代码。

**Step 2: 修改代码**

```python
# 找到第 763-785 行左右的 finish_callback 函数
# 在 status.status = "completed" 后添加:

# 同步更新 training_service.experiments
from backend.modules.training.training_service import training_service
if task_id in training_service.experiments:
    training_service.experiments[task_id]["status"] = "completed"
    training_service._save_experiments()
```

**Step 3: 验证**

```bash
# 检查后端日志，确认训练完成时状态同步
curl -s http://localhost:8000/api/v1/experiments | python3 -c "import sys,json; d=json.load(sys.stdin); print([e['status'] for e in d['experiments']])"
```

---

## Task 2: 新增后端 API - 获取单个实验详情

**Files:**
- Modify: `backend/modules/training/routes.py:408-425`

**Step 1: 在 get_experiments 后添加新 API**

```python
@router.get("/experiments/{task_id}")
async def get_experiment(task_id: str):
    """
    获取单个实验详情接口

    Args:
        task_id: 实验 ID

    Returns:
        实验详情
    """
    logger.debug(f"[训练] 查询实验详情: task_id={task_id}")

    experiment = training_service.experiments.get(task_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    # 获取图表数据
    from backend.core.yolo_engine import yolo_engine
    chart_data = None
    status = None

    if yolo_engine:
        status = yolo_engine.get_training_status(task_id)
        if status:
            chart_data = status.get_chart_data()

    # 构建返回数据
    result = {
        "success": True,
        "data": {
            **experiment,
            "chart_data": chart_data,
            "best_metrics": status.best_metrics if status else experiment.get("best_metrics"),
            "checkpoint_path": status.checkpoint_path if status else experiment.get("checkpoint_path")
        }
    }
    return result
```

**Step 2: 测试 API**

```bash
curl -s http://localhost:8000/api/v1/experiments/train_1773021304 | python3 -m json.tool
```

---

## Task 3: 新增后端 API - 获取验证结果图片

**Files:**
- Modify: `backend/modules/training/routes.py` (在上一个任务添加的 API 后面)

**Step 1: 添加获取验证结果的 API**

```python
@router.get("/experiments/{task_id}/results")
async def get_experiment_results(task_id: str):
    """
    获取实验验证结果图片接口

    Args:
        task_id: 实验 ID

    Returns:
        验证结果图片列表
    """
    logger.debug(f"[训练] 查询验证结果: task_id={task_id}")

    experiment = training_service.experiments.get(task_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    project_name = experiment.get("project_name")
    if not project_name:
        raise HTTPException(status_code=400, detail="实验缺少项目名称")

    train_dir = Path(settings.MODELS_DIR) / project_name / "train"
    if not train_dir.exists():
        return {"success": True, "data": {"images": [], "path": str(train_dir)}}

    # 查找验证结果图片
    result_images = []
    image_patterns = [
        "confusion_matrix*.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "labels.jpg",
        "results.png"
    ]

    for pattern in image_patterns:
        for img_path in train_dir.glob(pattern):
            result_images.append({
                "name": img_path.name,
                "url": f"/models/{project_name}/train/{img_path.name}"
            })

    return {"success": True, "data": {"images": result_images, "path": str(train_dir)}}
```

**Step 2: 注册静态文件路由（如果尚未注册）**

确保 app.py 中有 `/models` 目录的静态文件映射。

**Step 3: 测试**

```bash
curl -s http://localhost:8000/api/v1/experiments/train_1773021304/results | python3 -m json.tool
```

---

## Task 4: 新增后端 API - 导出模型

**Files:**
- Modify: `backend/modules/training/routes.py`

**Step 1: 添加导出 API**

```python
@router.post("/experiments/{task_id}/export")
async def export_experiment_model(
    task_id: str,
    format: str = Query("onnx", description="导出格式: onnx/torchscript/tflite/pytorch")
):
    """
    导出实验模型接口

    Args:
        task_id: 实验 ID
        format: 导出格式

    Returns:
        导出结果
    """
    logger.info(f"[训练] 导出模型: task_id={task_id}, format={format}")

    experiment = training_service.experiments.get(task_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    project_name = experiment.get("project_name")
    checkpoint_path = experiment.get("checkpoint_path")

    if not checkpoint_path:
        # 查找默认路径
        best_pt = Path(settings.MODELS_DIR) / project_name / "train" / "weights" / "best.pt"
        if best_pt.exists():
            checkpoint_path = str(best_pt)

    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise HTTPException(status_code=400, detail="模型文件不存在")

    # 调用导出
    from backend.core.yolo_engine import yolo_engine
    result = yolo_engine.export_model(checkpoint_path, format=format)

    return {"success": True, "data": result}
```

**Step 2: 测试**

```bash
curl -X POST "http://localhost:8000/api/v1/experiments/train_1773021304/export?format=onnx"
```

---

## Task 5: 新增后端 API - 推理测试

**Files:**
- Modify: `backend/modules/training/routes.py`

**Step 1: 添加推理 API**

```python
@router.post("/experiments/{task_id}/infer")
async def infer_with_experiment_model(
    task_id: str,
    image_url: str = Body(..., description="图片URL"),
    conf_threshold: float = Body(0.25, description="置信度阈值"),
    iou_threshold: float = Body(0.45, description="IOU阈值")
):
    """
    使用实验模型进行推理接口

    Args:
        task_id: 实验 ID
        image_url: 图片URL
        conf_threshold: 置信度阈值
        iou_threshold: IOU阈值

    Returns:
        推理结果
    """
    logger.info(f"[训练] 推理测试: task_id={task_id}")

    experiment = training_service.experiments.get(task_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    project_name = experiment.get("project_name")
    checkpoint_path = experiment.get("checkpoint_path")

    if not checkpoint_path:
        best_pt = Path(settings.MODELS_DIR) / project_name / "train" / "weights" / "best.pt"
        if best_pt.exists():
            checkpoint_path = str(best_pt)

    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise HTTPException(status_code=400, detail="模型文件不存在")

    # 调用推理
    from backend.core.yolo_engine import yolo_engine
    result = yolo_engine.predict(
        model_path=checkpoint_path,
        source=image_url,
        conf=conf_threshold,
        iou=iou_threshold
    )

    return {"success": True, "data": result}
```

---

## Task 6: 新增后端 API - 继续训练

**Files:**
- Modify: `backend/modules/training/routes.py`

**Step 1: 添加继续训练 API**

```python
@router.post("/experiments/{task_id}/resume")
async def resume_training(
    task_id: str,
    epochs: int = Body(100, description="继续训练的轮数"),
    batch_size: int = Body(None, description="批量大小"),
    resume_from_best: bool = Body(True, description="是否从最佳权重继续")
):
    """
    继续训练接口

    Args:
        task_id: 实验 ID
        epochs: 继续训练的轮数
        batch_size: 批量大小
        resume_from_best: 是否从最佳权重继续

    Returns:
        新的训练任务信息
    """
    logger.info(f"[训练] 继续训练: task_id={task_id}, epochs={epochs}")

    experiment = training_service.experiments.get(task_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    project_name = experiment.get("project_name")
    model_type = experiment.get("model_type")

    # 查找模型路径
    if resume_from_best:
        model_path = Path(settings.MODELS_DIR) / project_name / "train" / "weights" / "best.pt"
    else:
        model_path = Path(settings.MODELS_DIR) / project_name / "train" / "weights" / "last.pt"

    if not model_path.exists():
        raise HTTPException(status_code=400, detail="模型权重文件不存在")

    # 调用训练
    from backend.core.yolo_engine import yolo_engine
    result = yolo_engine.train(
        model=str(model_path),
        data=experiment.get("dataset_path"),
        epochs=epochs,
        batch_size=batch_size or experiment.get("batch_size"),
        project=project_name + "_continue",
        name=datetime.now().strftime("train_%H%M%S"),
        exist_ok=True,
        resume=True
    )

    return {"success": True, "data": result}
```

---

## Task 7: 前端 - 修改训练监控页使用正确的 API

**Files:**
- Modify: `frontend-react/src/pages/TrainingMonitor/TrainingMonitor.tsx:178-214`
- Modify: `frontend-react/src/services/api.ts`

**Step 1: 添加 API 函数**

在 api.ts 中添加:

```typescript
getExperiment: (taskId: string) => api.get(`/experiments/${taskId}`),
getExperimentResults: (taskId: string) => api.get(`/experiments/${taskId}/results`),
exportExperimentModel: (taskId: string, format: string) =>
  api.post(`/experiments/${taskId}/export?format=${format}`),
inferWithExperiment: (taskId: string, data: any) =>
  api.post(`/experiments/${taskId}/infer`, data),
resumeTraining: (taskId: string, data: any) =>
  api.post(`/experiments/${taskId}/resume`, data),
```

**Step 2: 修改 loadTasks 函数**

将 `/training/tasks` 改为 `/experiments`:

```typescript
const loadTasks = async () => {
  try {
    const res = await trainingApi.getExperiments(); // 改为 /experiments
    const tasksData = res.data?.experiments || [];
    // ... 后续代码保持不变
  } catch (e) {
    console.error(e);
  }
};
```

---

## Task 8: 前端 - 新增训练详情详情页

**Files:**
- Create: `frontend-react/src/pages/TrainingDetails/TrainingDetails.tsx`

**Step 1: 创建详情页组件**

创建 `TrainingDetails.tsx` 组件，包含:
- 使用 useParams 获取 taskId
- 加载实验详情 API
- 显示指标卡片（mAP50, mAP50-95, Precision, Recall）
- 显示训练曲线图表
- 显示验证结果图片网格
- 操作按钮（导出、推理、继续训练）

**Step 2: 添加路由**

在 `App.tsx` 或路由配置中添加:

```typescript
<Route path="/training/:taskId/details" element={<TrainingDetails />} />
```

**Step 3: 在列表页添加入口**

在 TrainingMonitor 列表中添加点击事件，跳转到详情页。

---

## Task 9: 前端 - 添加验证结果图片展示

**Files:**
- Modify: `frontend-react/src/pages/TrainingDetails/TrainingDetails.tsx`

**Step 1: 实现图片展示**

```typescript
// 获取验证结果
const resultsRes = await trainingApi.getExperimentResults(taskId);
const results = resultsRes.data?.data?.images || [];

// 渲染
<div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 'var(--space-4)' }}>
  {results.map((img: any) => (
    <Card key={img.name}>
      <img
        src={img.url}
        alt={img.name}
        style={{ width: '100%', borderRadius: 'var(--radius-md)' }}
      />
      <p style={{ marginTop: 'var(--space-2)', fontSize: '0.875rem' }}>{img.name}</p>
    </Card>
  ))}
</div>
```

---

## Task 10: 前端 - 添加操作按钮功能

**Files:**
- Modify: `frontend-react/src/pages/TrainingDetails/TrainingDetails.tsx`

**Step 1: 导出按钮**

```typescript
const handleExport = async (format: string) => {
  try {
    const res = await trainingApi.exportExperimentModel(taskId, format);
    message.success('导出成功');
  } catch (e) {
    message.error('导出失败');
  }
};
```

**Step 2: 推理按钮**

```typescript
const handleInfer = () => {
  navigate(`/inference?model=${taskId}`);
};
```

**Step 3: 继续训练按钮**

```typescript
const handleResume = () => {
  navigate(`/training?resume=${taskId}`);
};
```

---

## Task 11: 测试完整流程

**Step 1: 测试后端 API**

```bash
# 获取实验列表
curl -s http://localhost:8000/api/v1/experiments | python3 -c "import sys,json; d=json.load(sys.stdin); print('Experiments:', len(d['experiments']))"

# 获取单个实验详情
curl -s http://localhost:8000/api/v1/experiments/train_1773021304 | python3 -m json.tool | head -30

# 获取验证结果
curl -s http://localhost:8000/api/v1/experiments/train_1773021304/results | python3 -m json.tool
```

**Step 2: 测试前端页面**

访问 http://localhost:3000/training-monitor 确认列表显示正确。

---

## Task 12: 提交代码

```bash
git add -A
git commit -m "feat: 添加训练完成详情查看功能

- 修复训练完成时状态同步到 experiments.json
- 新增 /experiments/{task_id} API 获取实验详情
- 新增 /experiments/{task_id}/results API 获取验证结果图片
- 新增 /experiments/{task_id}/export API 导出模型
- 新增 /experiments/{task_id}/infer API 推理测试
- 新增 /experiments/{task_id}/resume API 继续训练
- 前端新增详情页显示完整训练信息
- 前端添加导出、推理、继续训练操作按钮

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
