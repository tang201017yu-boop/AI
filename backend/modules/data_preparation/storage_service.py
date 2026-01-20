"""
智能存储服务 - Intelligent Storage Service
提供数据去重、完整性校验、存储优化等功能
"""
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    import PIL.Image
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class StorageService:
    """
    智能存储服务

    功能:
    1. 重复数据删除 - 相同图像仅存储一次 (基于感知哈希/内容哈希)
    2. 完整性校验 - 校验和确保数据完整性 (MD5/SHA256)
    3. 存储优化 - 快速处理和索引
    """

    # 数据库文件名
    STORAGE_DB = "storage_index.db"
    # 存储根目录
    STORAGE_DIR = "data/storage"
    # 文件索引表名
    TABLE_FILES = "file_index"
    TABLE_DUPLICATES = "duplicates"
    TABLE_CHECKSUMS = "checksums"

    def __init__(self, storage_root: str = None):
        self.storage_root = Path(storage_root) if storage_root else Path(self.STORAGE_DIR)
        self.storage_root.mkdir(parents=True, exist_ok=True)

        # 图片存储目录
        self.images_dir = self.storage_root / "images"
        self.images_dir.mkdir(exist_ok=True)

        # 缩略图目录
        self.thumbnails_dir = self.storage_root / "thumbnails"
        self.thumbnails_dir.mkdir(exist_ok=True)

        # 数据库
        self.db_path = self.storage_root / self.STORAGE_DB
        self._init_database()

        # 索引锁
        self.index_lock = threading.Lock()

        # 线程池用于并行处理
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 感知哈希相似度阈值 (越低越严格)
        self.phash_threshold = 5

    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # 文件索引表
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_FILES} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE NOT NULL,      # SHA256 内容哈希
                phash TEXT,                           # 感知哈希
                file_path TEXT NOT NULL,              # 相对路径
                file_size INTEGER,                    # 文件大小
                width INTEGER,                        # 图片宽度
                height INTEGER,                       # 图片高度
                file_type TEXT,                       # 文件类型
                created_at TEXT,                      # 创建时间
                reference_count INTEGER DEFAULT 1,    # 引用计数
                is_deleted INTEGER DEFAULT 0          # 软删除标记
            )
        """)

        # 重复文件映射表
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_DUPLICATES} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_file_id INTEGER,             # 原始文件ID
                duplicate_file_id INTEGER,            # 重复文件ID
                similarity REAL,                      # 相似度
                created_at TEXT,
                FOREIGN KEY (original_file_id) REFERENCES {self.TABLE_FILES}(id),
                FOREIGN KEY (duplicate_file_id) REFERENCES {self.TABLE_FILES}(id)
            )
        """)

        # 校验和记录表
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_CHECKSUMS} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER,
                checksum_type TEXT,                   # md5, sha256
                checksum_value TEXT,
                verified_at TEXT,
                is_valid INTEGER DEFAULT 1,
                FOREIGN KEY (file_id) REFERENCES {self.TABLE_FILES}(id)
            )
        """)

        # 创建索引
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_file_hash ON {self.TABLE_FILES}(file_hash)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_phash ON {self.TABLE_FILES}(phash)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_reference ON {self.TABLE_FILES}(reference_count)")

        conn.commit()
        conn.close()

    # ==================== 核心功能 ====================

    def add_file(self, file_path: str) -> Dict[str, Any]:
        """
        添加文件到存储

        Args:
            file_path: 源文件路径

        Returns:
            {
                "success": bool,
                "action": "stored" | "deduplicated" | "skipped",
                "file_id": int,
                "storage_path": str,
                "duplicate_of": int,  # 如果是重复文件
                "message": str
            }
        """
        source_path = Path(file_path)
        if not source_path.exists():
            return {"success": False, "message": "文件不存在"}

        try:
            # 计算文件哈希
            file_hash = self._calculate_file_hash(file_path)
            file_size = source_path.stat().st_size

            # 获取图片信息
            img_info = self._get_image_info(file_path)

            # 检查是否已存在
            existing = self._find_by_hash(file_hash)
            if existing:
                # 增加引用计数
                self._increment_reference(existing["id"])
                return {
                    "success": True,
                    "action": "deduplicated",
                    "file_id": existing["id"],
                    "storage_path": existing["file_path"],
                    "duplicate_of": existing["id"],
                    "message": f"文件已存在，跳过存储 (引用计数: {existing['reference_count'] + 1})"
                }

            # 生成存储路径
            ext = source_path.suffix.lower()
            storage_name = f"{file_hash[:16]}{ext}"
            storage_path = self.images_dir / storage_name

            # 复制文件
            shutil.copy2(file_path, storage_path)

            # 计算感知哈希
            phash = self._calculate_phash(storage_path)

            # 保存到数据库
            file_id = self._insert_file_record(
                file_hash=file_hash,
                phash=phash,
                file_path=str(storage_path.relative_to(self.storage_root)),
                file_size=file_size,
                width=img_info.get("width"),
                height=img_info.get("height"),
                file_type=ext.lstrip(".")
            )

            # 计算并存储校验和
            self._calculate_and_store_checksum(file_id, storage_path)

            # 生成缩略图
            self._generate_thumbnail(storage_path, file_id)

            return {
                "success": True,
                "action": "stored",
                "file_id": file_id,
                "storage_path": str(storage_path.relative_to(self.storage_root)),
                "message": "文件存储成功"
            }

        except Exception as e:
            return {"success": False, "message": f"存储失败: {str(e)}"}

    def add_files_batch(self, file_paths: List[str], progress_callback=None) -> Dict[str, Any]:
        """
        批量添加文件

        Args:
            file_paths: 文件路径列表
            progress_callback: 进度回调函数 (current, total)

        Returns:
            {
                "success": True,
                "total": int,
                "stored": int,
                "deduplicated": int,
                "failed": int,
                "saved_mb": float
            }
        """
        results = {
            "total": len(file_paths),
            "stored": 0,
            "deduplicated": 0,
            "failed": 0,
            "saved_bytes": 0
        }

        for i, path in enumerate(file_paths):
            result = self.add_file(path)

            if result["success"]:
                if result["action"] == "stored":
                    results["stored"] += 1
                elif result["action"] == "deduplicated":
                    results["deduplicated"] += 1
                    # 估算节省的空间
                    results["saved_bytes"] += Path(path).stat().st_size
            else:
                results["failed"] += 1

            if progress_callback:
                progress_callback(i + 1, len(file_paths))

        results["saved_mb"] = round(results["saved_bytes"] / (1024 * 1024), 2)

        return {
            "success": results["failed"] == 0,
            **results
        }

    def verify_integrity(self, file_id: int = None) -> Dict[str, Any]:
        """
        验证数据完整性

        Args:
            file_id: 可选，验证特定文件

        Returns:
            {
                "success": bool,
                "total": int,
                "valid": int,
                "corrupted": int,
                "corrupted_files": List[dict]
            }
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        if file_id:
            cursor.execute(f"""
                SELECT id, file_path, checksum_value, checksum_type
                FROM {self.TABLE_CHECKSUMS}
                WHERE file_id = ?
                ORDER BY verified_at DESC
            """, (file_id,))
        else:
            cursor.execute(f"""
                SELECT cf.id, cf.file_path, cf.checksum_value, cf.checksum_type, cf.verified_at
                FROM {self.TABLE_FILES} f
                JOIN {self.TABLE_CHECKSUMS} cf ON f.id = cf.file_id
                WHERE f.is_deleted = 0
                AND cf.checksum_type = 'sha256'
            """)

        records = cursor.fetchall()
        conn.close()

        valid_count = 0
        corrupted = []

        for record in records:
            record_id, file_path, stored_checksum, checksum_type, verified_at = record
            full_path = self.storage_root / file_path

            if not full_path.exists():
                corrupted.append({
                    "file_id": record_id,
                    "file_path": file_path,
                    "error": "文件不存在"
                })
                self._mark_checksum_invalid(record_id)
            else:
                # 重新计算校验和
                current_checksum = self._calculate_file_checksum(full_path, "sha256")

                if current_checksum == stored_checksum:
                    valid_count += 1
                    # 更新验证时间
                    self._update_verification_time(record_id)
                else:
                    corrupted.append({
                        "file_id": record_id,
                        "file_path": file_path,
                        "error": "校验和不匹配"
                    })
                    self._mark_checksum_invalid(record_id)

        return {
            "success": len(corrupted) == 0,
            "total": len(records),
            "valid": valid_count,
            "corrupted": len(corrupted),
            "corrupted_files": corrupted
        }

    def find_similar(self, file_path: str, threshold: float = 0.9) -> List[Dict]:
        """
        查找相似图像

        Args:
            file_path: 查询文件路径
            threshold: 相似度阈值 (0-1)

        Returns:
            相似文件列表
        """
        # 计算查询文件的感知哈希
        query_phash = self._calculate_phash(file_path)
        if not query_phash:
            return []

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT id, file_path, phash, file_hash, reference_count
            FROM {self.TABLE_FILES}
            WHERE is_deleted = 0 AND phash IS NOT NULL
        """)

        records = cursor.fetchall()
        conn.close()

        similar = []
        for record in records:
            file_id, file_path, stored_phash, file_hash, ref_count = record

            # 计算汉明距离
            distance = self._phash_distance(query_phash, stored_phash)
            similarity = 1 - (distance / 64.0)

            if similarity >= threshold:
                similar.append({
                    "file_id": file_id,
                    "file_path": file_path,
                    "similarity": round(similarity, 4),
                    "hamming_distance": distance,
                    "reference_count": ref_count
                })

        return sorted(similar, key=lambda x: x["similarity"], reverse=True)

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # 总文件数
        cursor.execute(f"SELECT COUNT(*) FROM {self.TABLE_FILES} WHERE is_deleted = 0")
        total_files = cursor.fetchone()[0]

        # 总存储大小
        cursor.execute(f"SELECT COALESCE(SUM(file_size), 0) FROM {self.TABLE_FILES} WHERE is_deleted = 0")
        total_size = cursor.fetchone()[0]

        # 去重节省的空间
        cursor.execute(f"SELECT COALESCE(SUM(file_size), 0) FROM {self.TABLE_FILES} WHERE is_deleted = 0 AND reference_count > 1")
        duplicated_size = cursor.fetchone()[0]

        # 校验和状态
        cursor.execute(f"SELECT COUNT(*) FROM {self.TABLE_CHECKSUMS} WHERE is_valid = 1")
        valid_checksums = cursor.fetchone()[0]
        cursor.execute(f"SELECT COUNT(*) FROM {self.TABLE_CHECKSUMS} WHERE is_valid = 0")
        invalid_checksums = cursor.fetchone()[0]

        conn.close()

        return {
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "duplicated_files": duplicated_size,
            "duplicated_size_mb": round(duplicated_size / (1024 * 1024), 2),
            "storage_efficiency": round((1 - duplicated_size / max(total_size, 1)) * 100, 2) if total_size > 0 else 100,
            "checksums_valid": valid_checksums,
            "checksums_invalid": invalid_checksums,
            "integrity_status": "OK" if invalid_checksums == 0 else f"{invalid_checksums} 损坏"
        }

    def cleanup_unused(self, min_references: int = 1) -> Dict[str, Any]:
        """清理未使用的文件"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # 查找引用计数为0的文件
        cursor.execute(f"""
            SELECT id, file_path FROM {self.TABLE_FILES}
            WHERE reference_count <= ? AND is_deleted = 0
        """, (min_references,))

        unused = cursor.fetchall()
        deleted_count = 0
        freed_space = 0

        for file_id, file_path in unused:
            full_path = self.storage_root / file_path
            if full_path.exists():
                file_size = full_path.stat().st_size
                freed_space += file_size
                full_path.unlink()

            # 删除缩略图
            thumb_path = self.thumbnails_dir / f"{file_id}.jpg"
            if thumb_path.exists():
                thumb_path.unlink()

            # 软删除记录
            cursor.execute(f"UPDATE {self.TABLE_FILES} SET is_deleted = 1 WHERE id = ?", (file_id,))
            deleted_count += 1

        conn.commit()
        conn.close()

        return {
            "success": True,
            "deleted_count": deleted_count,
            "freed_space_mb": round(freed_space / (1024 * 1024), 2)
        }

    # ==================== 私有方法 ====================

    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件 SHA256 哈希"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _calculate_file_checksum(self, file_path: Path, algorithm: str = "sha256") -> str:
        """计算文件校验和"""
        hash_func = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def _calculate_phash(self, image_path: str) -> Optional[str]:
        """计算感知哈希 (简化版)"""
        if not PIL_AVAILABLE:
            return None

        try:
            img = Image.open(image_path)
            img = img.convert('L').resize((32, 32), Image.LANCZOS)

            # 计算均值
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)

            # 计算哈希
            hash_bits = 0
            for i, pixel in enumerate(pixels):
                if pixel >= avg:
                    hash_bits |= 1 << (63 - i)

            return format(hash_bits, '064b')
        except Exception:
            return None

    def _phash_distance(self, hash1: str, hash2: str) -> int:
        """计算两个感知哈希之间的汉明距离"""
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 64

        distance = 0
        for i in range(len(hash1)):
            if hash1[i] != hash2[i]:
                distance += 1
        return distance

    def _get_image_info(self, file_path: str) -> Dict:
        """获取图片信息"""
        if not PIL_AVAILABLE:
            return {}

        try:
            img = Image.open(file_path)
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode
            }
        except Exception:
            return {}

    def _find_by_hash(self, file_hash: str) -> Optional[Dict]:
        """根据哈希查找文件"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT id, file_path, reference_count, file_size, phash
            FROM {self.TABLE_FILES}
            WHERE file_hash = ? AND is_deleted = 0
        """, (file_hash,))

        record = cursor.fetchone()
        conn.close()

        if record:
            return {
                "id": record[0],
                "file_path": record[1],
                "reference_count": record[2],
                "file_size": record[3],
                "phash": record[4]
            }
        return None

    def _insert_file_record(self, **kwargs) -> int:
        """插入文件记录"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(f"""
            INSERT INTO {self.TABLE_FILES}
            (file_hash, phash, file_path, file_size, width, height, file_type, created_at, reference_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            kwargs.get("file_hash"),
            kwargs.get("phash"),
            kwargs.get("file_path"),
            kwargs.get("file_size"),
            kwargs.get("width"),
            kwargs.get("height"),
            kwargs.get("file_type"),
            datetime.now().isoformat(),
            kwargs.get("reference_count", 1)
        ))

        file_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return file_id

    def _increment_reference(self, file_id: int):
        """增加引用计数"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(f"""
            UPDATE {self.TABLE_FILES}
            SET reference_count = reference_count + 1
            WHERE id = ?
        """, (file_id,))

        conn.commit()
        conn.close()

    def _calculate_and_store_checksum(self, file_id: int, file_path: Path):
        """计算并存储校验和"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # SHA256 校验和
        checksum = self._calculate_file_checksum(file_path, "sha256")
        cursor.execute(f"""
            INSERT INTO {self.TABLE_CHECKSUMS}
            (file_id, checksum_type, checksum_value, verified_at, is_valid)
            VALUES (?, 'sha256', ?, ?, 1)
        """, (file_id, checksum, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def _mark_checksum_invalid(self, record_id: int):
        """标记校验和无效"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(f"""
            UPDATE {self.TABLE_CHECKSUMS}
            SET is_valid = 0
            WHERE id = ?
        """, (record_id,))

        conn.commit()
        conn.close()

    def _update_verification_time(self, record_id: int):
        """更新验证时间"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(f"""
            UPDATE {self.TABLE_CHECKSUMS}
            verified_at = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), record_id))

        conn.commit()
        conn.close()

    def _generate_thumbnail(self, file_path: Path, file_id: int):
        """生成缩略图"""
        if not PIL_AVAILABLE:
            return

        try:
            img = Image.open(file_path)
            img.thumbnail((200, 200), Image.LANCZOS)
            thumb_path = self.thumbnails_dir / f"{file_id}.jpg"
            img.save(thumb_path, "JPEG", quality=80)
        except Exception:
            pass

    def get_file_path(self, file_id: int) -> Optional[str]:
        """获取文件路径"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT file_path FROM {self.TABLE_FILES}
            WHERE id = ? AND is_deleted = 0
        """, (file_id,))

        record = cursor.fetchone()
        conn.close()

        if record:
            return str(self.storage_root / record[0])
        return None

    def get_thumbnail_path(self, file_id: int) -> Optional[str]:
        """获取缩略图路径"""
        thumb_path = self.thumbnails_dir / f"{file_id}.jpg"
        if thumb_path.exists():
            return str(thumb_path)
        return None


# 全局实例
storage_service = StorageService()
