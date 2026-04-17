# 智能方案（Solutions）API 返回字段说明

本文档说明 `backend/modules/solutions` 模块下各接口响应 JSON 中字段含义。  
**路径前缀**：`/api/v1`（与 `app.py` 中路由挂载一致）。

---

## 1. 列出方案 — `GET /solutions/list`

| 字段 | 类型 | 含义 |
|------|------|------|
| `success` | bool | 是否成功（路由内为 `true`）。 |
| `solutions` | array | 可用方案列表。 |
| `solutions[].name` | string | 内部标识，如 `object-counting`、`heatmap`。 |
| `solutions[].title` | string | 展示用中文名称。 |
| `solutions[].description` | string | 功能简述。 |
| `solutions[].input_types` | string[] | 支持的输入：`image`、`video`。 |
| `solutions[].features` | string[] | 能力标签（展示用）。 |

---

## 2. 对象计数 — `POST /solutions/object-counting`

| 字段 | 类型 | 含义 |
|------|------|------|
| `success` | bool | 处理是否成功。 |
| `message` | string | 说明；**单张图片**会提示进出计数依赖视频连续帧。 |
| `results` | object | 统计与推理摘要，见下表。 |
| `output_path` | string | 结果图/视频相对 URL，如 `/uploads/counted_xxx.ext`。 |

### `results` 对象

| 字段 | 类型 | 含义 |
|------|------|------|
| `in_count` | int | 进入计数区域的累计次数（**视频**有意义；静态图多为 0）。 |
| `out_count` | int | 离开计数区域的累计次数（同上）。 |
| `total_frames` | int | 已处理帧数（图片为 1）。 |
| `detected_objects` | int | 当前/最后一帧跟踪到的目标数量。 |
| `objects_by_class` | object | `{ "类名": 数量 }`，当前帧各类数量。 |
| `recent_inference_logs` | string[] | 每帧摘要日志，格式类似：`帧号: 高x宽 solution_ms, 类摘要 \| track_ms`。 |

失败时常见：`{ "success": false, "message": "..." }`。

---

## 3. 热图 — `POST /solutions/heatmap`（异步）与 `GET /solutions/heatmap/status/{task_id}`

### 提交任务 — `POST /solutions/heatmap`

| 字段 | 类型 | 含义 |
|------|------|------|
| `success` | bool | 是否成功入队。 |
| `task_id` | string | 短 ID，用于轮询状态。 |
| `message` | string | 如「任务已提交，请轮询获取进度」。 |

### 轮询状态 — `GET /solutions/heatmap/status/{task_id}`

任务状态保存在内存字典中，典型字段：

| 字段 | 类型 | 含义 |
|------|------|------|
| `status` | string | `processing` / `completed` / `failed`；无任务时为 `not_found`。 |
| `progress` | int | 0–100。 |
| `message` | string | 进度说明或错误信息；完成时为服务层返回的 `message`。 |
| `output_path` | string \| null | 完成后为 `/uploads/...`；处理中为 `null`。 |

若 `task_id` 不存在：`{ "status": "not_found", "message": "任务不存在" }`。

---

## 4. 速度估算 — `POST /solutions/speed-estimation`

| 字段 | 类型 | 含义 |
|------|------|------|
| `success` | bool | 是否成功。 |
| `message` | string | 如「速度估算完成」。 |
| `results` | object | 见下表。 |
| `output_path` | string | 输出视频 `/uploads/speed_...`（若有）。 |

### `results` 对象

| 字段 | 类型 | 含义 |
|------|------|------|
| `speeds` | array | 按帧收集的 `SpeedEstimator` 的 `speed`；结构随 Ultralytics 版本可能为每帧字典等。 |
| `frame_count` | int | 实际处理帧数。 |

**请求参数提示**：`pixel_to_meter` 为表单中的像素–米标定相关参数；是否与当前 Ultralytics 内部实现完全对应，以运行环境版本为准。

---

## 5. 距离计算 — `POST /solutions/distance-calculation`

| 字段 | 类型 | 含义 |
|------|------|------|
| `success` | bool | 是否成功（少于 2 个检测框会失败）。 |
| `message` | string | 说明或错误原因。 |
| `distances` | array | 所有目标**两两**中心点距离；失败且无可算距离时可能为 `[]`。 |
| `output_path` | string | 路由将服务层 `output_image` 转为 `/uploads/...`。 |

### `distances[]` 元素

| 字段 | 类型 | 含义 |
|------|------|------|
| `object1_index` | int | 检测框序号（从 1 起，与图上一致）。 |
| `object2_index` | int | 另一检测框序号。 |
| `pixel_distance` | float | 两框中心点欧氏距离，**单位：像素**。 |
| `p1` | [int, int] | 第一个中心点 `(x, y)`。 |
| `p2` | [int, int] | 第二个中心点 `(x, y)`。 |

可视化仅绘制距离最近的若干对（服务内 `max_connections` 限制），返回的 `distances` 仍为**全量两两组合**。

---

## 6. 对象模糊 — `POST /solutions/object-blur`

| 字段 | 类型 | 含义 |
|------|------|------|
| `success` | bool | 是否成功。 |
| `message` | string | 如「对象模糊完成」。 |
| `total_frames` | int | 处理帧数。 |
| `output_path` | string | `/uploads/blurred_...`。 |

---

## 7. 对象裁剪 — `POST /solutions/object-crop`

存在两套挂载（均在 `/api/v1` 下），**实际以先注册的路由为准**（当前主应用中 `backend/api/routes.py` 中的实现优先）。字段结构略有不同。

### 7.1 模块路由 `backend/modules/solutions/routes.py`（扁平 JSON）

| 字段 | 类型 | 含义 |
|------|------|------|
| `success` | bool | 是否成功。 |
| `message` | string | 如「对象裁剪完成」。 |
| `total_crops` | int | 裁剪块数量（无检测时为 0）。 |
| `cropped_images` | array | 每个检测一条记录。 |
| `output_dir` | string | 服务端保存目录（默认与上传根目录 `uploads` 相同，裁剪文件以 `crop_` 前缀命名）。 |

### 7.2 聚合响应 `SolutionResponse`（`backend/api/routes.py`）

| 字段 | 类型 | 含义 |
|------|------|------|
| `success` | bool | 是否成功。 |
| `message` | string | 如「对象裁剪完成」。 |
| `output_path` | string \| null | 首张裁剪图的浏览 URL，便于预览；无检测时为 `null`。 |
| `results` | object | 见下表。 |

#### `results` 对象

| 字段 | 类型 | 含义 |
|------|------|------|
| `total_crops` | int | 裁剪块数量。 |
| `cropped_images` | array | 与下表相同结构的数组。 |

### `cropped_images[]` 元素（两套接口一致）

| 字段 | 类型 | 含义 |
|------|------|------|
| `class_name` | string | 类别名称。 |
| `class_id` | int | 类别 ID。 |
| `crop_path` | string | 对外 URL：`/uploads/` + **相对上传根目录的路径**（一般为根目录下文件），例如 `/uploads/crop_person_0_原图名.jpg`，与 `StaticFiles` 挂载一致。 |
| `crop_size` | int[] | `[高度, 宽度]` 像素。 |

---


## 8. 队列管理 — `POST /solutions/queue-management`

| 字段 | 类型 | 含义 |
|------|------|------|
| `success` | bool | 是否成功。 |
| `message` | string | 如「队列管理完成」。 |
| `results` | object | 见下表。 |
| `output_path` | string | `/uploads/queue_...`。 |

### `results` 对象

| 字段 | 类型 | 含义 |
|------|------|------|
| `max_queue_count` | int | 全视频中队列区域内目标数的**峰值**。 |
| `frame_counts` | int[] | 每一帧的队列计数（时间顺序）。 |
| `total_frames` | int | 总帧数。 |
| `avg_queue_count` | float | 各帧队列计数的**平均值**（有数据时存在）。 |

---

## 9. 停车管理 — `POST /solutions/parking-management`

| 字段 | 类型 | 含义 |
|------|------|------|
| `success` | bool | 是否成功。 |
| `message` | string | 如「停车管理分析完成」。 |
| `results` | object | 见下表。 |
| `output_path` | string | 标注图或视频 `/uploads/parking_...`。 |

### `results` 对象

| 字段 | 类型 | 含义 |
|------|------|------|
| `total_slots` | int | 车位（多边形）数量。 |
| `current_occupied` | int | **最后一帧**判定为占用的车位数。 |
| `current_free` | int | `total_slots - current_occupied`。 |
| `avg_occupied` | float | 各帧占用数的平均值。 |
| `max_occupied` | int | 各帧占用数的最大值。 |
| `total_frames` | int | 处理帧数（单图为 1）。 |

占用判定：检测框**中心点**落在某车位多边形内则该车为占用（画面上常见 `OCC` / `FREE`）。

---

## 通用失败响应

多数接口在异常时返回：

```json
{ "success": false, "message": "错误说明" }
```

具体文案以服务端实现为准。

---

## 代码参考

- 路由：`backend/modules/solutions/routes.py`
- 业务与返回结构：`backend/modules/solutions/solutions_service.py`
