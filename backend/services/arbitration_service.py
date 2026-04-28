import json
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.config import settings


class ArbitrationService:
    def __init__(self) -> None:
        self.base_dir = settings.DATA_DIR / "governance"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cases_file = self.base_dir / "arbitration_cases.json"
        self._bootstrap()

    def _bootstrap(self) -> None:
        if self.cases_file.exists():
            return

        seed = self._build_seed_case()
        self._write_cases([seed])

    def _build_seed_case(self) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        payload = {
            "annotation_id": "demo-crack-001",
            "defect_type": "crack",
            "source": {
                "project": "demo",
                "image_name": "sample-crack.jpg",
            },
            "agent_annotation": {
                "label": "crack",
                "bbox": [40, 80, 200, 120],
                "attributes": {"area": 2500, "width_length_ratio": 0.06},
            },
            "human_annotation": {
                "label": "crack",
                "bbox": [42, 82, 198, 118],
                "attributes": {"area": 5000, "width_length_ratio": 0.12},
            },
        }
        result = self.evaluate(payload, persist=False)
        detail = result["case"]
        detail["id"] = "seed-demo-case"
        detail["status"] = "open"
        detail["openedAt"] = now
        detail["updatedAt"] = now
        return detail

    def _read_cases(self) -> List[Dict[str, Any]]:
        if not self.cases_file.exists():
            return []
        with open(self.cases_file, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_cases(self, cases: List[Dict[str, Any]]) -> None:
        with open(self.cases_file, "w", encoding="utf-8") as handle:
            json.dump(cases, handle, ensure_ascii=False, indent=2)

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _compute_iou(self, left: List[float], right: List[float]) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in left]
        bx1, by1, bx2, by2 = [float(v) for v in right]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _metrics_from_annotation(self, annotation: Dict[str, Any]) -> Dict[str, float]:
        bbox = annotation.get("bbox") or [0, 0, 0, 0]
        attrs = annotation.get("attributes") or {}
        x1, y1, x2, y2 = [float(v) for v in bbox]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        long_side = max(width, height)
        short_side = min(width, height)
        area = float(attrs.get("area", width * height))
        width_length_ratio = float(
            attrs.get("width_length_ratio", short_side / long_side if long_side else 0.0)
        )
        wetness_index = float(attrs.get("wetness_index", 0.5))
        connected_components = float(attrs.get("connected_components", 1))
        return {
            "width": width,
            "height": height,
            "area": area,
            "width_length_ratio": width_length_ratio,
            "wetness_index": wetness_index,
            "connected_components": connected_components,
        }

    def _evaluate_legality(self, defect_type: str, annotation: Dict[str, Any]) -> Dict[str, Any]:
        metrics = self._metrics_from_annotation(annotation)
        rule_hits: List[Dict[str, Any]] = []
        passed = True

        if defect_type == "crack":
            checks = [
                {
                    "id": "R_crack_001",
                    "label": "裂缝宽长比",
                    "metric": "width_length_ratio",
                    "operator": "<=",
                    "threshold": 0.08,
                    "value": metrics["width_length_ratio"],
                },
                {
                    "id": "R_crack_002",
                    "label": "最小像素面积",
                    "metric": "area",
                    "operator": ">=",
                    "threshold": 180.0,
                    "value": metrics["area"],
                },
            ]
        elif defect_type == "seepage":
            checks = [
                {
                    "id": "R_seep_001",
                    "label": "湿润指数下限",
                    "metric": "wetness_index",
                    "operator": ">=",
                    "threshold": 0.45,
                    "value": metrics["wetness_index"],
                },
                {
                    "id": "R_seep_002",
                    "label": "连通域上限",
                    "metric": "connected_components",
                    "operator": "<=",
                    "threshold": 5.0,
                    "value": metrics["connected_components"],
                },
            ]
        elif defect_type in ("spalling", "delamination", "脱落"):
            checks = [
                {
                    "id": "R_spall_001",
                    "label": "脱落区域面积下限",
                    "metric": "area",
                    "operator": ">=",
                    "threshold": 200.0,
                    "value": metrics["area"],
                },
                {
                    "id": "R_spall_002",
                    "label": "宽长比上限",
                    "metric": "width_length_ratio",
                    "operator": "<=",
                    "threshold": 0.85,
                    "value": metrics["width_length_ratio"],
                },
            ]
        elif defect_type in ("hollow", "void", "空鼓"):
            checks = [
                {
                    "id": "R_hollow_001",
                    "label": "空鼓区域面积下限",
                    "metric": "area",
                    "operator": ">=",
                    "threshold": 150.0,
                    "value": metrics["area"],
                },
                {
                    "id": "R_hollow_002",
                    "label": "形状紧致度下限",
                    "metric": "width_length_ratio",
                    "operator": ">=",
                    "threshold": 0.12,
                    "value": metrics["width_length_ratio"],
                },
            ]
        else:
            checks = [
                {
                    "id": "R_common_001",
                    "label": "目标面积下限",
                    "metric": "area",
                    "operator": ">=",
                    "threshold": 120.0,
                    "value": metrics["area"],
                },
                {
                    "id": "R_common_002",
                    "label": "形状稳定性",
                    "metric": "width_length_ratio",
                    "operator": ">=",
                    "threshold": 0.12,
                    "value": metrics["width_length_ratio"],
                },
            ]

        for check in checks:
            value = check["value"]
            threshold = check["threshold"]
            operator = check["operator"]
            if operator == "<=":
                hit = value <= threshold
            else:
                hit = value >= threshold
            rule_hits.append({**check, "passed": hit})
            passed = passed and hit

        legality = "compliant" if passed else "violation"
        notes = [f"{item['label']} {'通过' if item['passed'] else '未通过'}" for item in rule_hits]
        return {
            "legality": legality,
            "metrics": metrics,
            "rule_hits": rule_hits,
            "notes": notes,
        }

    def _scenario(self, agent_ok: bool, human_ok: bool) -> Dict[str, str]:
        if agent_ok and human_ok:
            return {
                "code": "LEGAL",
                "label": "双方合法",
                "winner": "either",
                "action": "接受任一标注并建议取平均位置",
            }
        if agent_ok and not human_ok:
            return {
                "code": "ILLEGAL-B",
                "label": "人类违规",
                "winner": "agent",
                "action": "采信 Agent 标注",
            }
        if not agent_ok and human_ok:
            return {
                "code": "ILLEGAL-A",
                "label": "Agent 违规",
                "winner": "human",
                "action": "采信人类标注",
            }
        return {
            "code": "ILLEGAL-AB",
            "label": "双方违规",
            "winner": "expert",
            "action": "推送 L4 专家重标",
        }

    def _build_compare_annotation(self, title: str, annotation: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
        bbox = annotation.get("bbox") or [0, 0, 0, 0]
        metrics = review["metrics"]
        json_summary = {
            "label": annotation.get("label"),
            "bbox": bbox,
            "legality": review["legality"],
            "metrics": metrics,
        }
        return {
            "id": title.lower().replace(" ", "-"),
            "label": title,
            "bbox": bbox,
            "legality": review["legality"],
            "notes": review["notes"],
            "jsonSummary": json.dumps(json_summary, ensure_ascii=False, indent=2),
            "attributes": {
                "area": round(metrics["area"], 3),
                "width_length_ratio": round(metrics["width_length_ratio"], 4),
                "wetness_index": round(metrics["wetness_index"], 4),
                "connected_components": round(metrics["connected_components"], 3),
            },
        }

    def _upsert_case(self, detail: Dict[str, Any]) -> Dict[str, Any]:
        cases = self._read_cases()
        existing_index: Optional[int] = None
        for index, case in enumerate(cases):
            if detail["annotationId"] and case.get("annotationId") == detail["annotationId"]:
                existing_index = index
                break
            if case.get("id") == detail.get("id"):
                existing_index = index
                break

        if existing_index is None:
            detail["id"] = detail.get("id") or str(uuid.uuid4())
            detail["openedAt"] = detail.get("openedAt") or self._now()
            detail["updatedAt"] = self._now()
            cases.append(detail)
        else:
            previous = cases[existing_index]
            detail["id"] = previous["id"]
            detail["openedAt"] = previous.get("openedAt", self._now())
            detail["updatedAt"] = self._now()
            detail["status"] = previous.get("status", detail.get("status", "open"))
            detail["judgement"] = previous.get("judgement")
            cases[existing_index] = detail

        self._write_cases(cases)
        return detail

    def evaluate(self, payload: Dict[str, Any], persist: bool = True) -> Dict[str, Any]:
        defect_type = str(payload.get("defect_type") or "crack").strip().lower()
        annotation_id = payload.get("annotation_id")
        source = payload.get("source") or {}
        agent_annotation = deepcopy(payload.get("agent_annotation") or {})
        human_annotation = deepcopy(payload.get("human_annotation") or {})
        iou_threshold = float(payload.get("iou_threshold", 0.3))

        iou_score = self._compute_iou(agent_annotation.get("bbox") or [0, 0, 0, 0], human_annotation.get("bbox") or [0, 0, 0, 0])
        # 与专利流程图一致：IoU 达到阈值及以上时进入 IoU 分歧仲裁（两框足够重叠才裁决）
        triggered = iou_score >= iou_threshold
        agent_review = self._evaluate_legality(defect_type, agent_annotation)
        human_review = self._evaluate_legality(defect_type, human_annotation)
        scenario = self._scenario(
            agent_review["legality"] == "compliant",
            human_review["legality"] == "compliant",
        )

        detail = {
            "id": str(uuid.uuid4()),
            "title": f"{defect_type.capitalize()} 物理约束仲裁",
            "annotationId": annotation_id,
            "status": "open" if triggered else "review",
            "openedAt": self._now(),
            "updatedAt": self._now(),
            "defectType": defect_type,
            "scenario": scenario["code"],
            "scenarioLabel": scenario["label"],
            "decisionAction": scenario["action"],
            "decisionWinner": scenario["winner"],
            "description": (
                f"IoU={iou_score:.3f}，T_iou={iou_threshold:.2f}；仲裁入口"
                f"{'已开启' if triggered else '未开启（两框重叠不足）'}。"
                f"物理四场景：{scenario['label']}。"
            ),
            "claimantNote": "Agent 标注候选",
            "respondentNote": "人类标注候选",
            "triggered": triggered,
            "triggerIou": round(iou_score, 4),
            "iouThreshold": iou_threshold,
            "source": source,
            "leftAnnotation": self._build_compare_annotation("Agent 标注", agent_annotation, agent_review),
            "rightAnnotation": self._build_compare_annotation("人类标注", human_annotation, human_review),
            "metrics": {
                "agent": agent_review["metrics"],
                "human": human_review["metrics"],
            },
            "ruleHits": {
                "agent": agent_review["rule_hits"],
                "human": human_review["rule_hits"],
            },
            "judgement": None,
        }

        if persist and triggered:
            detail = self._upsert_case(detail)

        alerts: List[Dict[str, Any]] = []
        if triggered:
            alerts.append(
                {
                    "id": detail["id"],
                    "level": "warning",
                    "message": (
                        f"IoU {iou_score:.3f} ≥ T_iou {iou_threshold:.2f}，已触发 {scenario['label']} 仲裁流程。"
                    ),
                    "disputeId": detail["id"],
                }
            )
        else:
            alerts.append(
                {
                    "id": f"pass-{annotation_id or 'local'}",
                    "level": "info",
                    "message": (
                        f"IoU {iou_score:.3f} < T_iou {iou_threshold:.2f}，两框重叠不足，暂不进入仲裁。"
                    ),
                }
            )

        return {
            "triggered": triggered,
            "iou": round(iou_score, 4),
            "scenario": scenario["code"],
            "scenarioLabel": scenario["label"],
            "decisionAction": scenario["action"],
            "decisionWinner": scenario["winner"],
            "agentReview": agent_review,
            "humanReview": human_review,
            "alerts": alerts,
            "case": detail,
        }

    def list_cases(self, status: Optional[str] = None, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        cases = sorted(self._read_cases(), key=lambda item: item.get("updatedAt", ""), reverse=True)
        if status:
            cases = [case for case in cases if case.get("status") == status]
        start = max(page - 1, 0) * page_size
        end = start + page_size
        items = cases[start:end]
        return {
            "items": items,
            "total": len(cases),
            "page": page,
            "page_size": page_size,
        }

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        for case in self._read_cases():
            if case.get("id") == case_id:
                return case
        return None

    def get_stats(self) -> Dict[str, Any]:
        cases = self._read_cases()
        open_count = sum(1 for case in cases if case.get("status") == "open")
        review_count = sum(1 for case in cases if case.get("status") == "review")
        resolved_count = sum(1 for case in cases if case.get("status") == "resolved")
        triggered_cases = [case for case in cases if case.get("triggered")]
        auto_resolved = [case for case in triggered_cases if case.get("decisionWinner") != "expert"]
        coverage = (len(auto_resolved) / len(triggered_cases) * 100.0) if triggered_cases else 0.0
        expert_reduction = coverage
        return {
            "open": open_count,
            "inReview": review_count,
            "resolved": resolved_count,
            "total": len(cases),
            "coverage": round(coverage, 1),
            "expertReduction": round(expert_reduction, 1),
        }

    def submit_judgement(self, case_id: str, valid: bool, reason: Optional[str] = None) -> bool:
        cases = self._read_cases()
        for case in cases:
            if case.get("id") != case_id:
                continue
            case["status"] = "resolved" if valid else "rejected"
            case["updatedAt"] = self._now()
            case["judgement"] = {
                "valid": valid,
                "reason": reason or case.get("decisionAction"),
                "decidedAt": case["updatedAt"],
                "decidedBy": "operator",
            }
            self._write_cases(cases)
            return True
        return False


arbitration_service = ArbitrationService()
