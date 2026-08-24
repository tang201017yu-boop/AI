import json
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.config import settings


class RuleService:
    """File-backed governance rule library.

    The frontend currently models rules as editable text rules. This service keeps
    that contract while storing versions so the rule detail/history pages work.
    """

    def __init__(self) -> None:
        self.base_dir = settings.DATA_DIR / "governance"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.rules_file = self.base_dir / "rules.json"
        self.versions_file = self.base_dir / "rule_versions.json"
        self._bootstrap()

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _read_json(self, path: Path, fallback: Any) -> Any:
        if not path.exists():
            return deepcopy(fallback)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return deepcopy(fallback)

    def _write_json(self, path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    def _read_rules(self) -> List[Dict[str, Any]]:
        return self._read_json(self.rules_file, [])

    def _write_rules(self, rules: List[Dict[str, Any]]) -> None:
        self._write_json(self.rules_file, rules)

    def _read_versions(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._read_json(self.versions_file, {})

    def _write_versions(self, versions: Dict[str, List[Dict[str, Any]]]) -> None:
        self._write_json(self.versions_file, versions)

    def _bootstrap(self) -> None:
        if self.rules_file.exists():
            return

        now = self._now()
        seed_rules = [
            {
                "id": "R_crack_001",
                "name": "裂缝宽长比",
                "description": "裂缝目标应满足细长形态约束，用于物理合法性判定。",
                "content": "defect_type=crack; metric=width_length_ratio; operator=<=; threshold=0.08",
                "tags": ["crack", "physical", "arbitration"],
                "version": "1.0.0",
                "status": "published",
                "createdBy": "system",
                "createdAt": now,
                "updatedAt": now,
            },
            {
                "id": "R_crack_002",
                "name": "裂缝最小像素面积",
                "description": "过滤过小疑似噪声目标。",
                "content": "defect_type=crack; metric=area; operator=>=; threshold=180",
                "tags": ["crack", "area", "arbitration"],
                "version": "1.0.0",
                "status": "published",
                "createdBy": "system",
                "createdAt": now,
                "updatedAt": now,
            },
            {
                "id": "R_seep_001",
                "name": "渗水湿润指数下限",
                "description": "渗水区域需要达到湿润指数阈值。",
                "content": "defect_type=seepage; metric=wetness_index; operator=>=; threshold=0.45",
                "tags": ["seepage", "wetness", "arbitration"],
                "version": "1.0.0",
                "status": "published",
                "createdBy": "system",
                "createdAt": now,
                "updatedAt": now,
            },
        ]
        self._write_rules(seed_rules)
        self._write_versions(
            {
                rule["id"]: [
                    {
                        "id": f"{rule['id']}-v1",
                        "version": rule["version"],
                        "summary": "系统初始化规则",
                        "createdAt": now,
                        "createdBy": "system",
                    }
                ]
                for rule in seed_rules
            }
        )

    def _summary(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": rule["id"],
            "name": rule["name"],
            "description": rule.get("description", ""),
            "version": rule.get("version", "1.0.0"),
            "status": rule.get("status", "draft"),
            "updatedAt": rule.get("updatedAt", rule.get("createdAt", self._now())),
            "createdBy": rule.get("createdBy"),
        }

    def _next_version(self, current: str) -> str:
        parts = current.split(".")
        try:
            major = int(parts[0]) if len(parts) > 0 else 1
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
        except ValueError:
            return "1.0.1"
        return f"{major}.{minor}.{patch + 1}"

    def _append_version(self, rule_id: str, rule: Dict[str, Any], summary: str) -> None:
        versions = self._read_versions()
        versions.setdefault(rule_id, []).insert(
            0,
            {
                "id": str(uuid.uuid4()),
                "version": rule.get("version", "1.0.0"),
                "summary": summary,
                "createdAt": rule.get("updatedAt", self._now()),
                "createdBy": rule.get("updatedBy", "operator"),
            },
        )
        self._write_versions(versions)

    def list_rules(self, page: int = 1, page_size: int = 20, q: Optional[str] = None) -> Dict[str, Any]:
        rules = sorted(self._read_rules(), key=lambda item: item.get("updatedAt", ""), reverse=True)
        if q:
            needle = q.strip().lower()
            rules = [
                rule
                for rule in rules
                if needle in rule.get("name", "").lower()
                or needle in rule.get("description", "").lower()
                or any(needle in str(tag).lower() for tag in rule.get("tags", []))
            ]
        start = max(page - 1, 0) * page_size
        end = start + page_size
        return {
            "items": [self._summary(rule) for rule in rules[start:end]],
            "total": len(rules),
            "page": page,
            "page_size": page_size,
        }

    def get_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        for rule in self._read_rules():
            if rule.get("id") == rule_id:
                return rule
        return None

    def create_rule(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = self._now()
        rule = {
            "id": payload.get("id") or str(uuid.uuid4()),
            "name": str(payload.get("name", "")).strip(),
            "description": payload.get("description") or "",
            "content": payload.get("content") or "",
            "tags": payload.get("tags") or [],
            "version": "1.0.0",
            "status": payload.get("status") or "draft",
            "createdBy": payload.get("createdBy") or "operator",
            "createdAt": now,
            "updatedAt": now,
        }
        if not rule["name"]:
            raise ValueError("Rule name is required")
        if not rule["content"]:
            raise ValueError("Rule content is required")

        rules = self._read_rules()
        rules.append(rule)
        self._write_rules(rules)
        self._append_version(rule["id"], rule, "创建规则")
        return rule

    def update_rule(self, rule_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rules = self._read_rules()
        for index, rule in enumerate(rules):
            if rule.get("id") != rule_id:
                continue
            updated = {**rule}
            for key in ("name", "description", "content", "tags", "status"):
                if key in payload and payload[key] is not None:
                    updated[key] = payload[key]
            updated["version"] = self._next_version(str(rule.get("version", "1.0.0")))
            updated["updatedAt"] = self._now()
            updated["updatedBy"] = payload.get("updatedBy") or "operator"
            rules[index] = updated
            self._write_rules(rules)
            self._append_version(rule_id, updated, "更新规则")
            return updated
        return None

    def delete_rule(self, rule_id: str) -> bool:
        rules = self._read_rules()
        next_rules = [rule for rule in rules if rule.get("id") != rule_id]
        if len(next_rules) == len(rules):
            return False
        self._write_rules(next_rules)
        return True

    def list_versions(self, rule_id: str) -> List[Dict[str, Any]]:
        return self._read_versions().get(rule_id, [])

    def set_status(self, rule_id: str, status: str) -> Optional[Dict[str, Any]]:
        if status not in {"draft", "published", "archived"}:
            raise ValueError("Invalid rule status")
        return self.update_rule(rule_id, {"status": status})


rule_service = RuleService()
