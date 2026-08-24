# 治理域（规则库 / 仲裁 / 标注增强）整合设计

> 适用范围：`/governance/rules`、`/governance/arbitration`、标注模块的合法性/分歧增强能力。
> 本文档对照设计稿与现有代码，给出**完整类型定义、页面布局、API 路由、架构差距**。
> 编写日期：2026-04-29

---

## 1. 整合架构

### 1.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      AI Vision Platform                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│  │   标注模块   │◄─►│  仲裁模块    │◄─►│  规则库模块  │         │
│  │ /annotation  │   │/arbitration  │   │   /rules     │         │
│  │              │   │              │   │              │         │
│  │ ✅ Agent建议 │   │ ✅ 中心列表  │   │ ⚠️ 通用文本  │         │
│  │ ✅ 合法性Badge│  │ ✅ 详情页    │   │   规则        │        │
│  │ ✅ 分歧提示  │   │ ✅ 评估接口  │   │ ❌ 物理规则  │         │
│  │ ✅ 物理判定  │   │ ❌ 四场景图  │   │ ❌ 后端路由  │         │
│  │   Hook       │   │ ❌ 专家队列  │   │ ❌ 审计/导入 │         │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘         │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│  │ 人类标注接口 │   │ 分歧检测引擎 │   │ 规则存储服务 │         │
│  │ Agent标注接口│   │ (IoU 阈值)   │   │ (JSON文件)   │         │
│  │ ✅ SAM/YOLO  │   │ ✅ 已实现    │   │ ❌ 待建      │         │
│  └──────────────┘   ├──────────────┤   ├──────────────┤         │
│                     │ 合法性判定器 │   │ 版本管理服务 │         │
│                     │ ✅ 硬编码规则│   │ ❌ 待建      │         │
│                     │   (要从规则  │   │              │         │
│                     │    库读取)   │   │              │         │
│                     └──────┬───────┘   └──────────────┘         │
│                            │                                    │
│                            ▼                                    │
│                     ┌──────────────┐                            │
│                     │ 反馈联动模块 │  ❌ 完全缺失              │
│                     │ (裁决→更新   │                            │
│                     │  规则阈值)   │                            │
│                     └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 关键链路差距

设计稿期望的核心闭环：

```
仲裁裁决 → 反馈联动模块 → 规则库阈值微调建议 → 规则版本升级
   ✅         ❌              ❌                  ❌
```

只完成了第一步，后续三步都缺失。

---

## 2. 现状对照表

| 模块/对象 | 当前状态 | 设计稿要求 | 差距 |
|---|---|---|---|
| `RuleDetail` / `RuleSummary` 类型 | 通用文本规则（`content: string`） | 结构化物理规则 | ❌ 完全是两个东西 |
| 后端 `_evaluate_legality()` | 已使用结构化物理规则 `{id, label, metric, operator, threshold, value}` | 要的就是这个 | ✅ 形态对，但**硬编码在代码里**没入库 |
| 后端 `/governance/rules` 路由 | **不存在** | CRUD/版本/审计/导入导出 | ❌ 没实现 |
| 仲裁中心 `/governance/arbitration` | 列表 + 统计卡 + 详情页 | + 四场景统计图 + 专家介入队列 | ⚠️ 缺图表与队列 |
| 仲裁详情 | 整页 `DisputeDetailPage` | 弹窗形态 | ⚠️ 形态不一样，但功能基本齐 |
| 反馈联动模块（裁决回流） | 完全没有 | 触发规则阈值调整建议 | ❌ 全缺 |

**核心问题**：现在的"规则库"是**通用文本规则**，但仲裁实际跑的是**结构化物理规则**——两套系统没打通。本期目标即让**规则库存的就是仲裁用的那套规则**。

---

## 3. 模块详细设计

### 3.1 模块1：规则库管理

**路由**：`/governance/rules`

**功能**：
- 规则 CRUD 操作
- 规则版本管理
- 规则审计日志
- 规则导入/导出

**支持的目标物（与后端 `_evaluate_legality` 对齐 + 业务扩展）**：
裂缝、渗水、盐析、岩层剥落、钢筋外露、空鼓、通用。

#### 3.1.1 数据结构

```typescript
/** 缺陷/目标物类型 */
export type DefectType =
  | 'crack'         // 裂缝
  | 'seepage'       // 渗水
  | 'efflorescence' // 盐析
  | 'spalling'      // 岩层剥落
  | 'rebar'         // 钢筋外露
  | 'hollow'        // 空鼓
  | 'common';       // 通用

/** 物理量度指标 */
export type PhysicalMetric =
  | 'width'                // 宽度（px/mm）
  | 'height'               // 高度
  | 'area'                 // 像素面积
  | 'width_length_ratio'   // 宽长比
  | 'wetness_index'        // 湿润指数（渗水类）
  | 'connected_components' // 连通域数量
  | 'aspect_ratio'         // 长宽比
  | 'circularity'          // 圆度
  | 'solidity';            // 致密度

/** 比较运算符 */
export type RuleOperator = '<=' | '>=' | '<' | '>' | '==' | '!=';

/** 严重等级 */
export type RuleSeverity = 'low' | 'medium' | 'high' | 'critical';

/** 规则状态（沿用现有） */
export type RuleStatus = 'draft' | 'published' | 'archived';

/** 物理规则单条（核心） */
export interface PhysicalRule {
  id: string;                         // 规则ID，如 R_crack_001
  targetType: DefectType;             // 目标物类型
  name: string;                       // 规则名（"裂缝宽长比上限"）
  description?: string;               // 说明文字
  metric: PhysicalMetric;             // 检测指标
  operator: RuleOperator;             // 运算符
  threshold: number;                  // 阈值
  unit?: 'px' | 'mm' | '%' | 'count' | 'ratio'; // 单位
  severity: RuleSeverity;             // 严重等级
  enabled: boolean;                   // 是否启用（false 仍存在但不参与仲裁）
  version: string;                    // 当前版本号 SemVer
  status: RuleStatus;                 // 草稿/已发布/已归档
  tags?: string[];                    // 标签（如 "L2-自动判定"、"水下工况"）
  createdAt: string;                  // ISO 时间
  updatedAt: string;
  createdBy?: string;                 // 创建人
  updatedBy?: string;                 // 最近修改人
  publishedAt?: string;               // 发布时间
  publishedBy?: string;
}

/** 规则列表项（瘦身版，列表/卡片用） */
export interface PhysicalRuleSummary
  extends Pick<PhysicalRule,
    'id' | 'targetType' | 'name' | 'metric' | 'operator' | 'threshold'
    | 'unit' | 'severity' | 'enabled' | 'version' | 'status' | 'updatedAt'> {}

/** 版本历史 */
export interface PhysicalRuleVersion {
  id: string;
  version: string;
  ruleId: string;
  summary?: string;                   // 改动摘要
  diff?: {                            // 关键字段 diff
    before?: Partial<PhysicalRule>;
    after?: Partial<PhysicalRule>;
  };
  createdAt: string;
  createdBy?: string;
}

/** 审计日志 */
export interface RuleAuditLog {
  id: string;
  ruleId: string;
  action: 'create' | 'update' | 'publish' | 'archive' | 'delete' | 'enable' | 'disable';
  operator: string;
  timestamp: string;
  comment?: string;
  beforeSnapshot?: Partial<PhysicalRule>;
  afterSnapshot?: Partial<PhysicalRule>;
}

/** 规则导入导出包 */
export interface RuleBundle {
  schemaVersion: '1.0';
  exportedAt: string;
  rules: PhysicalRule[];
}

/** 表单值（编辑器用） */
export interface PhysicalRuleFormValues {
  name: string;
  targetType: DefectType;
  metric: PhysicalMetric;
  operator: RuleOperator;
  threshold: string;        // 表单输入是字符串，提交时转 number
  unit: string;
  severity: RuleSeverity;
  description: string;
  tags: string;             // 逗号分隔
  enabled: boolean;
}
```

#### 3.1.2 页面布局：规则库管理

```
┌────────────────────────────────────────────────────────────────────────┐
│  规则库管理                                       [导入] [导出] [+ 新建规则] │
├────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│  │ 总数 24  │ │ 已发布 18│ │ 草稿 4   │ │ 已归档 2 │                   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘                   │
├────────────────────────────────────────────────────────────────────────┤
│  目标物筛选: [全部▾] [裂缝] [渗水] [盐析] [剥落] [钢筋外露]            │
│  严重等级:   [全部▾] [Low] [Medium] [High] [Critical]                  │
│  状态:       [全部▾]  搜索: [_____________]                            │
├────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────┐ ┌──────────────────────────────┐ │
│  │ R_crack_001  v1.2.0  [已发布]    │ │ R_seep_001  v1.0.0 [草稿]    │ │
│  │ 裂缝宽长比上限                   │ │ 渗水湿润指数下限             │ │
│  │ width_length_ratio  <=  0.08     │ │ wetness_index  >=  0.45      │ │
│  │ 严重: High         启用 ●        │ │ 严重: Medium    启用 ○       │ │
│  │ 标签: L2-自动 / 水下             │ │ 更新于 2 小时前              │ │
│  └──────────────────────────────────┘ └──────────────────────────────┘ │
│  ...                                                                   │
└────────────────────────────────────────────────────────────────────────┘
```

#### 3.1.3 页面布局：规则编辑器

```
┌─────────────────────────────────────────────────────────────────┐
│ ← 返回                              [保存为草稿] [发布]         │
├─────────────────────────────────────────────────────────────────┤
│ 基本信息                                                        │
│   规则ID:    [R_crack_001        ]  (自动)                      │
│   规则名称:  [裂缝宽长比上限     ]                              │
│   目标物:    [裂缝 ▾]                                           │
│   严重等级:  [● Low  ○ Medium  ○ High  ○ Critical]              │
│   说明:      [_______________________________________________]  │
├─────────────────────────────────────────────────────────────────┤
│ 物理判定                                                        │
│   检测指标:  [width_length_ratio ▾]  单位: [ratio ▾]            │
│   运算符:    [<= ▾]   阈值: [0.08]                              │
│   预览:      "当 width_length_ratio <= 0.08 时合法"             │
├─────────────────────────────────────────────────────────────────┤
│ 元数据                                                          │
│   标签:      [L2-自动判定, 水下工况]                            │
│   启用:      [● 是  ○ 否]                                       │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.1.4 页面布局：规则详情

```
┌─────────────────────────────────────────────────────────────────┐
│ ← 规则列表          [编辑] [复制] [归档] [删除]                 │
├─────────────────────────────────────────────────────────────────┤
│ R_crack_001                                  [v1.2.0 已发布]    │
│ 裂缝宽长比上限                                                  │
│ ─────────────────────────────────────────────────────────────── │
│ 目标物: 裂缝   严重: High   启用: ●                             │
│ 判定:   width_length_ratio <= 0.08                              │
│ 说明:   该指标用于过滤毛细裂缝...                               │
│ 标签:   [L2-自动判定] [水下工况]                                │
├─────────────────────────────────────────────────────────────────┤
│ 版本历史 (RuleHistory)                                          │
│  • v1.2.0 - 2024-12-01 调整阈值 0.10 → 0.08  by Alice           │
│  • v1.1.0 - 2024-11-15 增加水下工况标签       by Alice          │
│  • v1.0.0 - 2024-10-01 初始版本                by Bob           │
├─────────────────────────────────────────────────────────────────┤
│ 审计日志                                                        │
│  [发布] 2024-12-01 14:30  Alice  "针对水下场景收紧阈值"         │
│  [更新] 2024-11-15 10:15  Alice                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 模块2：分歧仲裁

**路由**：`/governance/arbitration`

**功能**：
- 分歧样本队列管理
- 实时仲裁结果展示
- 四种场景统计
- 专家介入队列

#### 3.2.1 物理四场景编码

| 编码 | 后端字符串 | 含义 | 配色（CSS 变量） |
|---|---|---|---|
| 1 | `ILLEGAL-B` | 人类违规，Agent 合法 | `var(--warning)` 橙 |
| 2 | `ILLEGAL-A` | Agent 违规，人类合法 | `var(--info)` 蓝 |
| 3 | `LEGAL` | 双方均合法（真实分歧） | `var(--gray-400)` 灰 |
| 4 | `ILLEGAL-AB` | 双方均违规 | `var(--error)` 红 |

类型定义见 `frontend-react/src/types/arbitration.ts` 中的 `ScenarioCode` 与 `mapScenarioToCode()`（已实现）。

#### 3.2.2 页面布局：仲裁中心

```
┌─────────────────────────────────────────────────────────────────────┐
│ 分歧仲裁中心                                                        │
├─────────────────────────────────────────────────────────────────────┤
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                         │
│ │待处理 5│ │审核中 3│ │已解决61│ │总数 69 │                         │
│ └────────┘ └────────┘ └────────┘ └────────┘                         │
│ 自动覆盖率 ████████░░ 88%   预计专家介入 ▼ 88%                      │
├─────────────────────────────────────────────────────────────────────┤
│ 物理四场景统计 (饼图/柱状图)                                        │
│ ▰▰▰▰▰▰  场景1 人类违规    18 (26%)  橙                              │
│ ▰▰▰▰    场景2 Agent违规   12 (17%)  蓝                              │
│ ▰▰▰▰▰▰▰ 场景3 真实分歧    32 (46%)  灰                              │
│ ▰▰      场景4 双方违规     7 (10%)  红                              │
├─────────────────────────────────────────────────────────────────────┤
│ 仲裁案件队列 (DisputeCard 已实现)                                   │
│ [筛选: 状态▾ 缺陷类型▾ 场景▾]                                       │
│ ┌──────────────────────┐ ┌──────────────────────┐                   │
│ │ ... DisputeCard ...  │ │ ... DisputeCard ...  │                   │
│ └──────────────────────┘ └──────────────────────┘                   │
├─────────────────────────────────────────────────────────────────────┤
│ 专家介入队列                                                        │
│  • Case #042 双方违规 - 等待专家分配         [分配 →]               │
│  • Case #051 双方违规 - 已分配给 Bob 12h 前  [查看]                 │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3.2.3 页面布局：仲裁详情弹窗

```
┌─────────────────────────────────────────────────────────────────────┐
│ 分歧样本 #042 详情                                          [×]     │
├─────────────────────────────────────────────────────────────────────┤
│ 缺陷类型: 裂缝     IoU: 0.42 (T=0.30)     场景: 双方均合法 (灰)     │
│ 触发时间: 2024-12-01 14:30                                          │
├─────────────────────────────────────────────────────────────────────┤
│ ┌──── 人类标注 ────┐    ┌──── Agent标注 ────┐                       │
│ │  [图像预览]      │    │  [图像预览]      │                        │
│ │  bbox 42,82,...  │    │  bbox 40,80,...  │                        │
│ │  [合法] 绿       │    │  [合法] 绿       │                        │
│ └──────────────────┘    └──────────────────┘                        │
├─────────────────────────────────────────────────────────────────────┤
│ 规则命中明细                                                        │
│ ┌──── 人类 ────┐  ┌──── Agent ────┐                                 │
│ │ ✓ R_crack_001│  │ ✓ R_crack_001 │                                 │
│ │   0.06 <= 0.08│  │   0.05 <= 0.08│                                │
│ │ ✓ R_crack_002│  │ ✓ R_crack_002 │                                 │
│ │   5000 >= 180│  │   2500 >= 180 │                                 │
│ └──────────────┘  └───────────────┘                                 │
├─────────────────────────────────────────────────────────────────────┤
│ 处置建议: 接受任一标注并建议取平均位置                              │
│                                                                     │
│ 裁定结果: ○ 通过(采信 Agent)  ○ 通过(采信人类)  ○ 推送专家          │
│ 备注: [____________________________________]                        │
│                                                  [取消] [确认裁决] │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 模块3：标注模块增强（已实现）

| 组件/Hook | 文件 | 状态 |
|---|---|---|
| Agent 建议 | `modules/annotation/components/AgentSuggestion.tsx` | ✅ |
| 属性面板 | `modules/annotation/components/AttributePanel.tsx` | ✅ |
| 合法性徽章 | `modules/annotation/components/LegalBadge.tsx` | ✅ |
| 分歧提示 | `modules/annotation/components/DisputeAlert.tsx` | ✅ |
| 分歧检测 Hook | `modules/annotation/hooks/useDisputeDetection.ts` | ✅ |
| 物理判定 Hook | `modules/annotation/hooks/usePhysicalJudge.ts` | ✅ |
| 物理属性计算 | `shared/utils/physicalAttributes.ts` | ✅ |

---

## 4. API 路由完整规划

### 4.1 规则库（**后端缺失，需新建**）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/governance/rules` | 列表 + 筛选（targetType / severity / status / q） |
| GET | `/governance/rules/:id` | 单条详情 |
| POST | `/governance/rules` | 创建（默认 draft） |
| PUT | `/governance/rules/:id` | 更新 |
| DELETE | `/governance/rules/:id` | 删除 |
| POST | `/governance/rules/:id/publish` | 发布（draft→published） |
| POST | `/governance/rules/:id/archive` | 归档 |
| GET | `/governance/rules/:id/versions` | 版本历史 |
| GET | `/governance/rules/:id/audit` | 审计日志 |
| GET | `/governance/rules/export?ids=...` | 导出 RuleBundle JSON |
| POST | `/governance/rules/import` | 导入 RuleBundle |

### 4.2 仲裁（**已存在 5 个，缺 3 个**）

| 状态 | 方法 | 路径 |
|---|---|---|
| ✅ | GET | `/governance/arbitration/disputes` |
| ✅ | GET | `/governance/arbitration/disputes/:id` |
| ✅ | GET | `/governance/arbitration/stats` |
| ✅ | POST | `/governance/arbitration/evaluate` |
| ✅ | POST | `/governance/arbitration/disputes/:id/judge` |
| ❌ | GET | `/governance/arbitration/scenarios` （四场景分布统计，柱状图用） |
| ❌ | GET | `/governance/arbitration/expert-queue` （专家介入队列） |
| ❌ | POST | `/governance/arbitration/expert-queue/:id/assign` |

---

## 5. 前端组件结构

### 5.1 当前结构

```
src/
├── modules/
│   ├── rules/
│   │   ├── pages/
│   │   │   ├── RulesList.tsx          ✅
│   │   │   ├── RuleDetail.tsx         ✅
│   │   │   └── RuleEditor.tsx         ✅
│   │   ├── components/
│   │   │   ├── RuleCard.tsx           ✅（基于通用文本规则，需重构）
│   │   │   ├── RuleForm.tsx           ✅（同上）
│   │   │   └── RuleHistory.tsx        ✅
│   │   └── api/
│   │       └── rulesApi.ts            ✅（基于通用文本规则）
│   │
│   ├── arbitration/
│   │   ├── pages/
│   │   │   ├── ArbitrationCenter.tsx  ✅（缺四场景图、专家队列）
│   │   │   └── DisputeDetail.tsx      ✅（整页形态，待加弹窗版）
│   │   ├── components/
│   │   │   ├── DisputeCard.tsx        ✅（已按设计稿改造）
│   │   │   ├── AnnotationCompare.tsx  ✅
│   │   │   ├── JudgeResult.tsx        ✅
│   │   │   └── StatsPanel.tsx         ✅（四个统计卡）
│   │   └── api/
│   │       └── arbitrationApi.ts      ✅（含 normalizeDispute）
│   │
│   └── annotation/                    ✅（全部存在，见 3.3）
│
├── shared/
│   ├── components/
│   │   ├── ImageBox/                  ✅
│   │   └── Badge/                     ✅
│   └── utils/
│       └── physicalAttributes.ts      ✅
│
└── types/
    ├── rule.ts                        ⚠️ 通用文本规则，需扩展为 PhysicalRule
    ├── annotation.ts                  ✅
    └── arbitration.ts                 ✅（含 ScenarioCode、mapScenarioToCode）
```

### 5.2 待新增/重构

```
modules/rules/
├── pages/
│   ├── RulesList.tsx                  🔧 加目标物/严重等级/状态筛选 + 顶部统计卡
│   ├── RuleDetail.tsx                 🔧 显示结构化字段 + 审计日志
│   └── RuleEditor.tsx                 🔧 改用 PhysicalRuleFormValues
├── components/
│   ├── RuleCard.tsx                   🔧 显示 metric/operator/threshold/severity/启用开关
│   ├── RuleForm.tsx                   🔧 改为结构化表单
│   ├── RuleHistory.tsx                ✅
│   ├── RuleAuditList.tsx              ➕ 新增：审计日志列表
│   ├── RuleFilterBar.tsx              ➕ 新增：筛选条
│   └── RuleStats.tsx                  ➕ 新增：顶部统计卡
└── api/
    └── rulesApi.ts                    🔧 加 publish/archive/audit/export/import 方法

modules/arbitration/
├── pages/
│   └── ArbitrationCenter.tsx          🔧 加四场景统计图 + 专家介入队列
├── components/
│   ├── ScenarioChart.tsx              ➕ 新增：四场景柱状/饼图
│   ├── ExpertQueue.tsx                ➕ 新增：专家介入队列
│   └── DisputeDetailModal.tsx         ➕ 新增：弹窗版详情
└── api/
    └── arbitrationApi.ts              🔧 加 scenarios/expertQueue/assign 方法
```

---

## 6. 后端待办

### 6.1 新建 `backend/services/rules_service.py`

参考 `arbitration_service.py` 的写法（JSON 文件存储 + Service 单例）：

```python
class RulesService:
    def __init__(self) -> None:
        self.base_dir = settings.DATA_DIR / "governance"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.rules_file = self.base_dir / "rules.json"
        self.versions_file = self.base_dir / "rule_versions.json"
        self.audit_file = self.base_dir / "rule_audit.json"
        self._bootstrap()

    def _bootstrap(self):
        # 把现在 _evaluate_legality() 里硬编码的规则迁移成种子数据
        # crack: R_crack_001, R_crack_002
        # seepage: R_seep_001, R_seep_002
        # spalling: R_spall_001, R_spall_002
        # hollow: R_hollow_001, R_hollow_002
        # common: R_common_001, R_common_002
        ...

    def list_rules(self, target_type=None, severity=None, status=None, q=None, page=1, page_size=20): ...
    def get_rule(self, rule_id: str): ...
    def create_rule(self, payload: dict, operator: str): ...
    def update_rule(self, rule_id: str, payload: dict, operator: str): ...
    def publish_rule(self, rule_id: str, operator: str, comment: str = None): ...
    def archive_rule(self, rule_id: str, operator: str): ...
    def delete_rule(self, rule_id: str, operator: str): ...
    def list_versions(self, rule_id: str): ...
    def list_audit(self, rule_id: str): ...
    def export_bundle(self, rule_ids: list[str] = None): ...
    def import_bundle(self, bundle: dict, operator: str): ...
```

### 6.2 改造 `arbitration_service._evaluate_legality()`

**目标**：把硬编码规则改为从 `rules_service` 读取**已发布且启用**的规则。

```python
def _evaluate_legality(self, defect_type: str, annotation: Dict[str, Any]) -> Dict[str, Any]:
    metrics = self._metrics_from_annotation(annotation)

    rules = rules_service.list_rules(
        target_type=defect_type,
        status="published",
    )
    enabled_rules = [r for r in rules["items"] if r.get("enabled")]

    rule_hits = []
    passed = True
    for rule in enabled_rules:
        value = metrics.get(rule["metric"])
        if value is None:
            continue
        op = rule["operator"]
        threshold = rule["threshold"]
        if op == "<=":
            hit = value <= threshold
        elif op == ">=":
            hit = value >= threshold
        # ... 其他运算符
        rule_hits.append({
            "id": rule["id"],
            "label": rule["name"],
            "metric": rule["metric"],
            "operator": op,
            "threshold": threshold,
            "value": value,
            "passed": hit,
        })
        passed = passed and hit
    ...
```

### 6.3 反馈联动模块（P2，先占位）

新建 `backend/services/feedback_service.py`：
- 监听 `submit_judgement` 事件
- 当某规则连续 N 次"判定结果与最终裁决不一致"时，生成阈值微调建议
- 建议写入 `data/governance/rule_suggestions.json`，规则库 UI 展示"待审建议"

---

## 7. 实施分期

| 阶段 | 范围 | 说明 |
|---|---|---|
| **P0** | 前端 `types/rule.ts` 重构、后端 `rules_service` + 路由、前端 `RuleForm`/`RuleCard` 改造 | 打通规则库 |
| **P1** | 仲裁服务接通规则库（`_evaluate_legality` 改造）、`RulesList` 加筛选/统计、仲裁中心加四场景图 | 形成闭环 |
| **P2** | 审计日志、规则导入导出、专家介入队列、仲裁详情弹窗 | 完善体验 |
| **P3** | 反馈联动模块（裁决回流到规则微调） | 闭环升级 |

---

## 8. 兼容性策略

### 8.1 旧 `RuleSummary` / `RuleDetail` 类型

考虑现有代码已在用，**保留**这两个类型作为"通用文本规则"，并在 `types/rule.ts` 新增 `PhysicalRule` 系列类型。

如果未来确认通用文本规则不再使用，再 deprecate 旧类型。

### 8.2 旧 `rulesApi`

现有 `rulesApi.list/get/create/update/history/delete` 保留。新增方法挂在同一个对象下：
- `rulesApi.publish(id, comment?)`
- `rulesApi.archive(id)`
- `rulesApi.audit(id)`
- `rulesApi.export(ids?)`
- `rulesApi.import(bundle)`

### 8.3 后端兼容

后端路由全为新增（`/governance/rules` 当前不存在），无兼容问题。

`arbitration_service` 改造时建议：
- 优先用规则库；
- 规则库为空时回退到当前硬编码逻辑（保险）。

---

## 9. 已实现成果（截至本文档撰写日）

- ✅ `types/arbitration.ts`：增加 `ScenarioCode` / `JudgeResultLabel` / `SideJudge` 等设计稿对齐字段，导出 `mapScenarioToCode()` 工具函数。
- ✅ `arbitrationApi.ts`：新增 `normalizeDispute()` + `listNormalized()` / `getNormalized()` 便捷方法。
- ✅ `DisputeCard.tsx`：按设计稿重写——双预览图、IoU、状态/合法性 Badge、场景彩色横条、`onViewDetail` 回调。
- ✅ `DisputeCard.module.css`：新样式（Grid 双列、4:3 预览框、占位 SVG、响应式）。
- ✅ `ArbitrationCenter.tsx`：调用 `listNormalized()`。

下一步建议从 **P0** 开始：先做 `types/rule.ts` 重构与后端 `rules_service`。
