# 治理域（规则库 / 仲裁联动）实现计划

> **配套设计文档**：`docs/plans/2026-04-29-governance-design.md`
> **Goal**：把"规则库"从"通用文本规则"升级为"结构化物理规则"，并接通到仲裁服务，形成 `规则库 → 仲裁判定 → 裁决回流` 的完整闭环。
> **Architecture**：FastAPI（后端） + React + TypeScript（前端） + JSON 文件存储（参考 `arbitration_service` 模式）
> **Tech Stack**：Python 3 / FastAPI / Pydantic / React 18 / Vite / 无数据库
> **撰写日期**：2026-04-29

---

## 总览

按 P0 → P3 四个阶段交付，每个阶段独立可用、可验证：

| 阶段 | 范围 | Tasks | 交付物 |
|---|---|---|---|
| **P0** | 规则库后端 + 类型迁移 + 前端表单/卡片改造 | T1–T7 | 规则库可独立 CRUD 使用 |
| **P1** | 仲裁服务接通规则库 + 列表筛选 + 四场景图 | T8–T11 | 完整治理闭环（首次跑通） |
| **P2** | 审计日志 + 规则导入导出 + 专家介入队列 + 详情弹窗 | T12–T15 | 工程化完善 |
| **P3** | 反馈联动模块（裁决回流到规则微调） | T16 | 闭环升级版 |

**当前状态**：仲裁前端 ✅、仲裁后端 ✅、规则库前端壳子 ✅、规则库后端 ❌、闭环 ❌。

---

# 阶段 P0：规则库后端 + 类型迁移

## Task 1: 新增 Pydantic 请求模型

**Files:**
- Modify: `backend/api/routes.py`（在文件顶部 `class ArbitrationJudgeRequest` 附近）

**Step 1: 在 routes.py 添加规则库请求模型**

```python
# ===== 规则库（Physical Rules）请求模型 =====
class RuleCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    targetType: str = Field(..., min_length=1)        # crack/seepage/...
    metric: str = Field(..., min_length=1)             # area/width_length_ratio/...
    operator: str = Field(..., pattern=r'^(<=|>=|<|>|==|!=)$')
    threshold: float
    unit: Optional[str] = None                          # px/mm/%/ratio/count
    severity: str = Field(default="medium", pattern=r'^(low|medium|high|critical)$')
    description: Optional[str] = ""
    tags: List[str] = Field(default_factory=list)
    enabled: bool = True


class RuleUpdateRequest(BaseModel):
    name: Optional[str] = None
    targetType: Optional[str] = None
    metric: Optional[str] = None
    operator: Optional[str] = None
    threshold: Optional[float] = None
    unit: Optional[str] = None
    severity: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    enabled: Optional[bool] = None
    comment: Optional[str] = None                       # 改动摘要（写入版本）


class RulePublishRequest(BaseModel):
    comment: Optional[str] = ""
    operator: Optional[str] = "system"


class RuleBundleRequest(BaseModel):
    schemaVersion: str = "1.0"
    rules: List[Dict[str, Any]]
```

**Step 2: 验证语法**

```bash
python -c "from backend.api import routes" 
```

---

## Task 2: 实现 RulesService（数据层）

**Files:**
- Create: `backend/services/rules_service.py`
- Modify: `backend/services/__init__.py`

**Step 1: 创建 rules_service.py**

参考 `backend/services/arbitration_service.py` 的写法，使用 JSON 文件存储：

```python
import json
import re
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.config import settings


class RulesService:
    def __init__(self) -> None:
        self.base_dir = settings.DATA_DIR / "governance"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.rules_file = self.base_dir / "rules.json"
        self.versions_file = self.base_dir / "rule_versions.json"
        self.audit_file = self.base_dir / "rule_audit.json"
        self._bootstrap()

    # ---- IO ----
    def _read(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, path: Path, data: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _now(self) -> str:
        return datetime.now().isoformat()

    # ---- 种子数据：把 arbitration_service._evaluate_legality 的硬编码规则迁移过来 ----
    def _bootstrap(self) -> None:
        if self.rules_file.exists():
            return
        seeds = self._build_seeds()
        self._write(self.rules_file, seeds)
        self._write(self.versions_file, [])
        self._write(self.audit_file, [])

    def _build_seeds(self) -> List[Dict[str, Any]]:
        now = self._now()
        base = lambda **kw: {
            "version": "1.0.0", "status": "published",
            "enabled": True, "createdAt": now, "updatedAt": now,
            "publishedAt": now, "createdBy": "system", "publishedBy": "system",
            **kw,
        }
        return [
            base(id="R_crack_001", targetType="crack", name="裂缝宽长比上限",
                 metric="width_length_ratio", operator="<=", threshold=0.08,
                 unit="ratio", severity="high",
                 description="过滤毛细裂缝，宽长比超过阈值视为违规",
                 tags=["裂缝", "L2-自动判定"]),
            base(id="R_crack_002", targetType="crack", name="裂缝最小像素面积",
                 metric="area", operator=">=", threshold=180.0,
                 unit="px", severity="medium",
                 description="过滤过小检测目标",
                 tags=["裂缝"]),
            base(id="R_seep_001", targetType="seepage", name="渗水湿润指数下限",
                 metric="wetness_index", operator=">=", threshold=0.45,
                 unit="ratio", severity="high",
                 description="湿润度低于阈值不构成渗水",
                 tags=["渗水"]),
            base(id="R_seep_002", targetType="seepage", name="渗水连通域上限",
                 metric="connected_components", operator="<=", threshold=5.0,
                 unit="count", severity="medium",
                 tags=["渗水"]),
            base(id="R_spall_001", targetType="spalling", name="脱落区域面积下限",
                 metric="area", operator=">=", threshold=200.0,
                 unit="px", severity="high",
                 tags=["岩层剥落"]),
            base(id="R_spall_002", targetType="spalling", name="脱落宽长比上限",
                 metric="width_length_ratio", operator="<=", threshold=0.85,
                 unit="ratio", severity="medium",
                 tags=["岩层剥落"]),
            base(id="R_hollow_001", targetType="hollow", name="空鼓区域面积下限",
                 metric="area", operator=">=", threshold=150.0,
                 unit="px", severity="high",
                 tags=["空鼓"]),
            base(id="R_hollow_002", targetType="hollow", name="空鼓形状紧致度下限",
                 metric="width_length_ratio", operator=">=", threshold=0.12,
                 unit="ratio", severity="medium",
                 tags=["空鼓"]),
            base(id="R_common_001", targetType="common", name="目标面积下限",
                 metric="area", operator=">=", threshold=120.0,
                 unit="px", severity="low", tags=["通用"]),
            base(id="R_common_002", targetType="common", name="目标形状稳定性",
                 metric="width_length_ratio", operator=">=", threshold=0.12,
                 unit="ratio", severity="low", tags=["通用"]),
        ]

    # ---- 工具：版本号自增（patch +1） ----
    def _bump_version(self, ver: str) -> str:
        m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", ver or "1.0.0")
        if not m:
            return "1.0.0"
        major, minor, patch = m.groups()
        return f"{major}.{minor}.{int(patch) + 1}"

    # ---- 审计日志 ----
    def _log_audit(self, rule_id: str, action: str, operator: str,
                   comment: Optional[str] = None,
                   before: Optional[Dict] = None, after: Optional[Dict] = None) -> None:
        logs = self._read(self.audit_file)
        logs.append({
            "id": str(uuid.uuid4()),
            "ruleId": rule_id,
            "action": action,
            "operator": operator,
            "timestamp": self._now(),
            "comment": comment,
            "beforeSnapshot": before,
            "afterSnapshot": after,
        })
        self._write(self.audit_file, logs)

    # ---- 版本快照 ----
    def _snapshot_version(self, rule: Dict[str, Any], summary: Optional[str],
                          operator: str, before: Optional[Dict] = None) -> None:
        versions = self._read(self.versions_file)
        versions.append({
            "id": str(uuid.uuid4()),
            "ruleId": rule["id"],
            "version": rule["version"],
            "summary": summary,
            "diff": {"before": before, "after": deepcopy(rule)} if before else None,
            "createdAt": self._now(),
            "createdBy": operator,
        })
        self._write(self.versions_file, versions)

    # ===== 公开接口 =====
    def list_rules(self, target_type: Optional[str] = None,
                   severity: Optional[str] = None,
                   status: Optional[str] = None,
                   q: Optional[str] = None,
                   page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        rules = self._read(self.rules_file)
        if target_type:
            rules = [r for r in rules if r.get("targetType") == target_type]
        if severity:
            rules = [r for r in rules if r.get("severity") == severity]
        if status:
            rules = [r for r in rules if r.get("status") == status]
        if q:
            kw = q.lower()
            rules = [r for r in rules if kw in (r.get("name", "") + r.get("description", "") + r.get("id", "")).lower()]
        rules = sorted(rules, key=lambda r: r.get("updatedAt", ""), reverse=True)
        start = max(page - 1, 0) * page_size
        end = start + page_size
        return {"items": rules[start:end], "total": len(rules), "page": page, "page_size": page_size}

    def get_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        for r in self._read(self.rules_file):
            if r.get("id") == rule_id:
                return r
        return None

    def create_rule(self, payload: Dict[str, Any], operator: str = "system") -> Dict[str, Any]:
        now = self._now()
        rule = {
            "id": payload.get("id") or f"R_{payload['targetType'][:5]}_{uuid.uuid4().hex[:6]}",
            "targetType": payload["targetType"],
            "name": payload["name"],
            "description": payload.get("description", ""),
            "metric": payload["metric"],
            "operator": payload["operator"],
            "threshold": float(payload["threshold"]),
            "unit": payload.get("unit"),
            "severity": payload.get("severity", "medium"),
            "enabled": bool(payload.get("enabled", True)),
            "version": "1.0.0",
            "status": "draft",
            "tags": payload.get("tags", []),
            "createdAt": now, "updatedAt": now, "createdBy": operator,
        }
        rules = self._read(self.rules_file)
        rules.append(rule)
        self._write(self.rules_file, rules)
        self._snapshot_version(rule, "初始版本", operator)
        self._log_audit(rule["id"], "create", operator, after=rule)
        return rule

    def update_rule(self, rule_id: str, payload: Dict[str, Any], operator: str = "system") -> Optional[Dict[str, Any]]:
        rules = self._read(self.rules_file)
        for i, r in enumerate(rules):
            if r.get("id") != rule_id:
                continue
            before = deepcopy(r)
            for k, v in payload.items():
                if k == "comment" or v is None:
                    continue
                r[k] = v
            r["updatedAt"] = self._now()
            r["updatedBy"] = operator
            r["version"] = self._bump_version(r.get("version"))
            rules[i] = r
            self._write(self.rules_file, rules)
            self._snapshot_version(r, payload.get("comment"), operator, before=before)
            self._log_audit(rule_id, "update", operator, comment=payload.get("comment"), before=before, after=r)
            return r
        return None

    def publish_rule(self, rule_id: str, operator: str = "system", comment: Optional[str] = "") -> Optional[Dict[str, Any]]:
        return self._set_status(rule_id, "published", operator, "publish", comment)

    def archive_rule(self, rule_id: str, operator: str = "system", comment: Optional[str] = "") -> Optional[Dict[str, Any]]:
        return self._set_status(rule_id, "archived", operator, "archive", comment)

    def _set_status(self, rule_id, new_status, operator, action, comment) -> Optional[Dict[str, Any]]:
        rules = self._read(self.rules_file)
        for i, r in enumerate(rules):
            if r.get("id") != rule_id:
                continue
            before = deepcopy(r)
            r["status"] = new_status
            r["updatedAt"] = self._now()
            if new_status == "published":
                r["publishedAt"] = r["updatedAt"]
                r["publishedBy"] = operator
            rules[i] = r
            self._write(self.rules_file, rules)
            self._log_audit(rule_id, action, operator, comment=comment, before=before, after=r)
            return r
        return None

    def delete_rule(self, rule_id: str, operator: str = "system") -> bool:
        rules = self._read(self.rules_file)
        new_rules = [r for r in rules if r.get("id") != rule_id]
        if len(new_rules) == len(rules):
            return False
        before = next(r for r in rules if r.get("id") == rule_id)
        self._write(self.rules_file, new_rules)
        self._log_audit(rule_id, "delete", operator, before=before)
        return True

    def list_versions(self, rule_id: str) -> List[Dict[str, Any]]:
        return [v for v in self._read(self.versions_file) if v.get("ruleId") == rule_id]

    def list_audit(self, rule_id: str) -> List[Dict[str, Any]]:
        return [a for a in self._read(self.audit_file) if a.get("ruleId") == rule_id]

    def export_bundle(self, ids: Optional[List[str]] = None) -> Dict[str, Any]:
        rules = self._read(self.rules_file)
        if ids:
            rules = [r for r in rules if r.get("id") in ids]
        return {"schemaVersion": "1.0", "exportedAt": self._now(), "rules": rules}

    def import_bundle(self, bundle: Dict[str, Any], operator: str = "system") -> Dict[str, Any]:
        if bundle.get("schemaVersion") != "1.0":
            raise ValueError("Unsupported schemaVersion")
        imported, skipped = [], []
        for r in bundle.get("rules", []):
            existing = self.get_rule(r.get("id", ""))
            if existing:
                skipped.append(r.get("id"))
                continue
            self.create_rule(r, operator=operator)
            imported.append(r.get("id"))
        return {"imported": imported, "skipped": skipped}

    def list_published_for_target(self, target_type: str) -> List[Dict[str, Any]]:
        """供 arbitration_service 调用：取已发布且启用的规则。"""
        return [
            r for r in self._read(self.rules_file)
            if r.get("targetType") == target_type
            and r.get("status") == "published"
            and r.get("enabled")
        ]


rules_service = RulesService()
```

**Step 2: 在 services/__init__.py 导出**

```python
# backend/services/__init__.py 顶部 imports 区追加
from backend.services.rules_service import rules_service
# ...
__all__ = [
    'dataset_service',
    'annotation_service',
    'arbitration_service',
    'rules_service',         # ← 追加
    'solutions_service',
    'supervision_service',
    'yolo_service'
]
```

**Step 3: 验证种子数据**

```bash
python -c "from backend.services.rules_service import rules_service; import json; print(json.dumps(rules_service.list_rules(), ensure_ascii=False, indent=2))"
```

应该看到 10 条预置规则。

---

## Task 3: 在 routes.py 注册 11 个路由

**Files:**
- Modify: `backend/api/routes.py`（在 `submit_arbitration_judgement` 之后追加）

**Step 1: import service**

```python
from backend.services.rules_service import rules_service
```

**Step 2: 追加路由**

```python
# ============ 规则库（Physical Rules）============

@router.get("/governance/rules")
async def list_governance_rules(
    target_type: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    q: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    try:
        return {"success": True, "data": rules_service.list_rules(
            target_type=target_type, severity=severity, status=status,
            q=q, page=page, page_size=page_size,
        )}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/governance/rules/{rule_id}")
async def get_governance_rule(rule_id: str):
    rule = rules_service.get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"success": True, "data": rule}


@router.post("/governance/rules")
async def create_governance_rule(payload: RuleCreateRequest):
    try:
        rule = rules_service.create_rule(payload.model_dump())
        return {"success": True, "data": rule}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/governance/rules/{rule_id}")
async def update_governance_rule(rule_id: str, payload: RuleUpdateRequest):
    rule = rules_service.update_rule(rule_id, payload.model_dump(exclude_none=True))
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"success": True, "data": rule}


@router.delete("/governance/rules/{rule_id}")
async def delete_governance_rule(rule_id: str):
    ok = rules_service.delete_rule(rule_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"success": True, "message": "Rule deleted"}


@router.post("/governance/rules/{rule_id}/publish")
async def publish_governance_rule(rule_id: str, payload: RulePublishRequest):
    rule = rules_service.publish_rule(rule_id, operator=payload.operator or "system", comment=payload.comment)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"success": True, "data": rule}


@router.post("/governance/rules/{rule_id}/archive")
async def archive_governance_rule(rule_id: str, payload: RulePublishRequest):
    rule = rules_service.archive_rule(rule_id, operator=payload.operator or "system", comment=payload.comment)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"success": True, "data": rule}


@router.get("/governance/rules/{rule_id}/versions")
async def list_governance_rule_versions(rule_id: str):
    return {"success": True, "data": {"versions": rules_service.list_versions(rule_id)}}


@router.get("/governance/rules/{rule_id}/audit")
async def list_governance_rule_audit(rule_id: str):
    return {"success": True, "data": {"logs": rules_service.list_audit(rule_id)}}


@router.get("/governance/rules-export")
async def export_governance_rules(ids: Optional[str] = None):
    id_list = ids.split(",") if ids else None
    return {"success": True, "data": rules_service.export_bundle(id_list)}


@router.post("/governance/rules-import")
async def import_governance_rules(payload: RuleBundleRequest):
    try:
        result = rules_service.import_bundle(payload.model_dump())
        return {"success": True, "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

> **路径设计说明**：导出/导入用 `rules-export` / `rules-import`（不带斜杠），避免与 `/rules/{rule_id}` 冲突。

**Step 3: 启动后验证**

```bash
# 列表
curl -s http://localhost:8000/api/v1/governance/rules | python -m json.tool

# 创建
curl -s -X POST http://localhost:8000/api/v1/governance/rules \
  -H "Content-Type: application/json" \
  -d '{"name":"测试规则","targetType":"crack","metric":"area","operator":">=","threshold":100}'

# 详情
curl -s http://localhost:8000/api/v1/governance/rules/R_crack_001 | python -m json.tool

# 发布
curl -s -X POST http://localhost:8000/api/v1/governance/rules/{id}/publish \
  -H "Content-Type: application/json" -d '{"comment":"发布"}'
```

---

## Task 4: 前端类型迁移 `types/rule.ts`

**Files:**
- Modify: `frontend-react/src/types/rule.ts`

**Step 1: 追加 PhysicalRule 系列类型，旧类型保留**

把设计稿 §3.1.1 的所有类型粘贴到 `rule.ts` 末尾。**保留** `RuleSummary` / `RuleDetail` / `RuleVersionEntry` / `RuleFormValues` 四个旧类型不变，避免破坏现有页面。

**Step 2: 验证编译**

```bash
cd frontend-react
npm run build
# 或在 dev 模式下查看 vite 是否报错
```

---

## Task 5: 前端 `rulesApi.ts` 扩展

**Files:**
- Modify: `frontend-react/src/modules/rules/api/rulesApi.ts`

**Step 1: 加 publish/archive/audit/export/import 方法**

```typescript
export const rulesApi = {
  // ... 既有方法不变 ...

  publish: (id: string, body: { comment?: string; operator?: string } = {}) =>
    api.post<ApiResponse<unknown>>(`/governance/rules/${id}/publish`, body),

  archive: (id: string, body: { comment?: string; operator?: string } = {}) =>
    api.post<ApiResponse<unknown>>(`/governance/rules/${id}/archive`, body),

  audit: (id: string) =>
    api.get<ApiResponse<{ logs: unknown[] }>>(`/governance/rules/${id}/audit`),

  exportBundle: (ids?: string[]) =>
    api.get<ApiResponse<unknown>>('/governance/rules-export', {
      params: ids?.length ? { ids: ids.join(',') } : undefined,
    }),

  importBundle: (bundle: unknown) =>
    api.post<ApiResponse<{ imported: string[]; skipped: string[] }>>('/governance/rules-import', bundle),
};
```

---

## Task 6: 重写 `RuleForm.tsx`（结构化表单）

**Files:**
- Modify: `frontend-react/src/modules/rules/components/RuleForm.tsx`
- Modify: `frontend-react/src/modules/rules/components/RuleForm.module.css`

**Step 1: 改用 PhysicalRuleFormValues**

字段：`name / targetType / metric / operator / threshold / unit / severity / description / tags / enabled`

布局参考设计稿 §3.1.3：
- 基本信息（name / targetType / severity / description）
- 物理判定（metric / operator / threshold / unit + 实时预览句子）
- 元数据（tags / enabled）

**Step 2: 实时预览**

```tsx
<p className={styles.preview}>
  当 <code>{metric}</code> {operator} <code>{threshold}{unit ? ` ${unit}` : ''}</code> 时合法
</p>
```

---

## Task 7: 重写 `RuleCard.tsx`

**Files:**
- Modify: `frontend-react/src/modules/rules/components/RuleCard.tsx`
- Modify: `frontend-react/src/modules/rules/components/RuleCard.module.css`

**Step 1: 显示结构化字段**

参考设计稿 §3.1.2 的卡片样式：
- 顶部：`R_crack_001  v1.2.0  [状态徽章]`
- 标题：`裂缝宽长比上限`
- 主体：`width_length_ratio  <=  0.08  (ratio)` (代码体)
- 严重等级 Badge + 启用开关
- 标签 chips
- `更新于 X 时间`

**Step 2: severity 配色**

```typescript
const severityColor = {
  low: 'var(--info)',
  medium: 'var(--warning)',
  high: 'var(--accent-500)',
  critical: 'var(--error)',
};
```

---

## P0 验收

- [ ] `curl /api/v1/governance/rules` 返回 10 条种子规则
- [ ] 前端 `/governance/rules` 列表显示 10 条卡片，结构化展示
- [ ] 新建规则 → 编辑 → 发布 → 归档 全流程可走
- [ ] 仲裁页面/标注页面**仍正常**（无回归）

---

# 阶段 P1：仲裁服务接通规则库

## Task 8: 改造 `arbitration_service._evaluate_legality`

**Files:**
- Modify: `backend/services/arbitration_service.py:104-223`

**Step 1: 替换硬编码为读规则库**

```python
def _evaluate_legality(self, defect_type: str, annotation: Dict[str, Any]) -> Dict[str, Any]:
    metrics = self._metrics_from_annotation(annotation)

    # === 优先从规则库读取 ===
    from backend.services.rules_service import rules_service
    rules = rules_service.list_published_for_target(defect_type) or \
            rules_service.list_published_for_target("common")

    rule_hits: List[Dict[str, Any]] = []
    passed = True
    for rule in rules:
        metric_key = rule["metric"]
        value = metrics.get(metric_key)
        if value is None:
            continue
        op = rule["operator"]
        threshold = float(rule["threshold"])
        if op == "<=": hit = value <= threshold
        elif op == ">=": hit = value >= threshold
        elif op == "<":  hit = value <  threshold
        elif op == ">":  hit = value >  threshold
        elif op == "==": hit = value == threshold
        elif op == "!=": hit = value != threshold
        else: hit = True
        rule_hits.append({
            "id": rule["id"],
            "label": rule["name"],
            "metric": metric_key,
            "operator": op,
            "threshold": threshold,
            "value": value,
            "passed": hit,
        })
        passed = passed and hit

    # === 兜底：规则库为空时回退到旧硬编码逻辑 ===
    if not rule_hits:
        return self._evaluate_legality_legacy(defect_type, annotation)

    legality = "compliant" if passed else "violation"
    notes = [f"{h['label']} {'通过' if h['passed'] else '未通过'}" for h in rule_hits]
    return {"legality": legality, "metrics": metrics, "rule_hits": rule_hits, "notes": notes}
```

**Step 2: 把原 `_evaluate_legality` 方法体重命名为 `_evaluate_legality_legacy`**（兜底用）

**Step 3: 联动测试**

```bash
# 触发一次仲裁，确认 ruleHits.id 是 R_crack_001 这样的规则库 ID
curl -s -X POST http://localhost:8000/api/v1/governance/arbitration/evaluate \
  -H "Content-Type: application/json" \
  -d '{"defect_type":"crack","agent_annotation":{"label":"crack","bbox":[40,80,200,120]},"human_annotation":{"label":"crack","bbox":[42,82,198,118]}}' \
  | python -m json.tool | grep -A1 '"id"'
```

---

## Task 9: 前端 `RulesList` 加筛选 / 统计

**Files:**
- Modify: `frontend-react/src/modules/rules/pages/RulesList.tsx`
- Create: `frontend-react/src/modules/rules/components/RuleFilterBar.tsx`
- Create: `frontend-react/src/modules/rules/components/RuleStats.tsx`

参考设计稿 §3.1.2：
- 顶部 4 个统计卡（总数 / 已发布 / 草稿 / 已归档）
- 筛选条（targetType / severity / status / 搜索）
- 列表网格

**关键改造点**：把 `rulesApi.list({ target_type, severity, status, q })` 当成受控查询，state 变化触发重新拉取。

---

## Task 10: 仲裁中心加 ScenarioChart

**Files:**
- Create: `frontend-react/src/modules/arbitration/components/ScenarioChart.tsx`
- Modify: `frontend-react/src/modules/arbitration/pages/ArbitrationCenter.tsx`

**Step 1: 后端补 `/governance/arbitration/scenarios` 接口**

在 `arbitration_service.py` 加：

```python
def get_scenario_breakdown(self) -> List[Dict[str, Any]]:
    cases = self._read_cases()
    counts = {"ILLEGAL-B": 0, "ILLEGAL-A": 0, "LEGAL": 0, "ILLEGAL-AB": 0}
    for c in cases:
        s = c.get("scenario")
        if s in counts:
            counts[s] += 1
    label = {"ILLEGAL-B":"人类违规","ILLEGAL-A":"Agent违规","LEGAL":"双方合法","ILLEGAL-AB":"双方违规"}
    code = {"ILLEGAL-B":1,"ILLEGAL-A":2,"LEGAL":3,"ILLEGAL-AB":4}
    return [{"code": code[k], "scenario": k, "label": label[k], "count": v} for k, v in counts.items()]
```

`routes.py`：

```python
@router.get("/governance/arbitration/scenarios")
async def get_arbitration_scenarios():
    return {"success": True, "data": arbitration_service.get_scenario_breakdown()}
```

**Step 2: 前端 ScenarioChart**

简单柱状图（不引外部图表库，用 div + width%）：

```tsx
<div className={styles.bars}>
  {data.map(d => (
    <div key={d.code} className={styles.row}>
      <span className={styles.lbl}>场景{d.code} {d.label}</span>
      <div className={styles.bar} style={{ width: `${(d.count/total)*100}%`, background: colorMap[d.code] }} />
      <span className={styles.cnt}>{d.count}</span>
    </div>
  ))}
</div>
```

---

## Task 11: P1 验收

- [ ] 仲裁返回的 `ruleHits[*].id` 是规则库 ID（如 `R_crack_001`）
- [ ] 修改规则库阈值 → 重新触发仲裁 → 结果跟着变
- [ ] `RulesList` 筛选/搜索/统计可用
- [ ] 仲裁中心显示四场景分布图

---

# 阶段 P2：完善

## Task 12: 审计日志 UI

**Files:**
- Create: `frontend-react/src/modules/rules/components/RuleAuditList.tsx`
- Modify: `frontend-react/src/modules/rules/pages/RuleDetail.tsx`

调用 `rulesApi.audit(id)` 显示。

---

## Task 13: 规则导入导出

**Files:**
- Modify: `frontend-react/src/modules/rules/pages/RulesList.tsx`

UI：列表头部 `[导入] [导出]` 两个按钮。
- 导出：`rulesApi.exportBundle()` → 下载 JSON 文件
- 导入：上传 JSON → `rulesApi.importBundle(bundle)`

---

## Task 14: 专家介入队列

**Files:**
- Modify: `backend/services/arbitration_service.py`（加 `expert_queue` 数据结构）
- Modify: `backend/api/routes.py`（加 2 个路由）
- Create: `frontend-react/src/modules/arbitration/components/ExpertQueue.tsx`

```python
@router.get("/governance/arbitration/expert-queue")
@router.post("/governance/arbitration/expert-queue/{case_id}/assign")
```

逻辑：场景 4（双方违规）的案件自动进入专家队列，可分配给 operator。

---

## Task 15: 仲裁详情弹窗

**Files:**
- Create: `frontend-react/src/modules/arbitration/components/DisputeDetailModal.tsx`

复用 `DisputeDetailPage` 的内容，外层套 `Modal`。`ArbitrationCenter` 列表点击 `DisputeCard` 时改为打开弹窗（保留整页路由作为深链接）。

---

# 阶段 P3：反馈联动模块

## Task 16: 反馈联动

**Files:**
- Create: `backend/services/feedback_service.py`
- Modify: `backend/services/arbitration_service.py:submit_judgement`

**逻辑**：
- `submit_judgement` 触发 `feedback_service.record(case, judgement)`
- 当某规则连续 N 次（例如 5 次）"判定结果与最终裁决不一致"时，生成阈值微调建议
- 建议写入 `data/governance/rule_suggestions.json`
- 规则库前端加"待审建议"入口

详细设计**先放着不做**，等 P2 完成再启动。

---

# 风险与回退

| 风险 | 缓解 |
|---|---|
| 改造 `_evaluate_legality` 后仲裁结果不一致 | Task 8 保留 `_evaluate_legality_legacy` 兜底 + 灰度（rules.json 删了就自动回退） |
| 规则 JSON 文件并发写入丢数据 | 项目目前单实例无并发，暂不处理；如需可加 fcntl/portalocker |
| 前端 `RuleSummary` 旧类型与新 `PhysicalRule` 混用 | T4 双类型并存，按页面逐步迁移 |
| 种子数据已写入但用户想重置 | 删除 `data/governance/rules.json` 即可，重启自动重建 |

---

# 提交节点（建议）

```bash
# P0 完成后
git add -A
git commit -m "feat(governance): 规则库后端 + 物理规则类型迁移

- 新增 RulesService（JSON 存储 + 版本/审计）
- 新增 11 个 /governance/rules API
- 前端类型扩展 PhysicalRule 系列
- RuleForm/RuleCard 改造为结构化形态
- 种子 10 条预置规则（裂缝/渗水/剥落/空鼓/通用）"

# P1 完成后
git commit -m "feat(governance): 仲裁服务接通规则库 + 列表筛选 + 四场景图

- arbitration_service._evaluate_legality 改读规则库（保留 legacy 兜底）
- 新增 /governance/arbitration/scenarios API
- 前端 RulesList 加筛选/统计、仲裁中心加 ScenarioChart"

# P2 完成后
git commit -m "feat(governance): 审计日志 + 导入导出 + 专家队列 + 详情弹窗"

# P3 完成后
git commit -m "feat(governance): 反馈联动模块（裁决回流到规则微调建议）"
```

---

# 下一步

如果你确认这份计划，**从 Task 1 开始直接动手**，每个 Task 完成后我会在终端验证一遍再进入下一个。

或者你也可以指定从某个 Task 开始（例如直接跳到 Task 8 联调）。
