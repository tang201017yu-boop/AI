"""
PostgreSQL 数据库服务 - 支持 pgvector 向量检索
"""
import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2 import sql
import numpy as np


@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DB", "opencv_platform")
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    pool_size: int = int(os.getenv("POSTGRES_POOL_SIZE", "10"))


@dataclass
class Experiment:
    """实验配置"""
    id: Optional[int] = None
    name: str = ""
    dataset_name: str = ""
    model_type: str = ""
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    learning_rate: float = 0.01
    config: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingLog:
    """训练日志"""
    id: Optional[int] = None
    experiment_id: int = 0
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    lr: float = 0.0
    created_at: Optional[datetime] = None


@dataclass
class User:
    """用户信息"""
    id: Optional[int] = None
    username: str = ""
    email: str = ""
    role: str = "user"
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None


class PostgresService:
    """PostgreSQL 数据库服务"""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._connection = None

    def get_connection(self):
        """获取数据库连接"""
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
        return self._connection

    @contextmanager
    def get_cursor(self, commit=True):
        """获取游标上下文"""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()

    def init_database(self):
        """初始化数据库表结构"""
        with self.get_cursor() as cursor:
            # 启用 pgvector 扩展
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # 实验配置表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    dataset_name VARCHAR(255) NOT NULL,
                    model_type VARCHAR(100) NOT NULL,
                    epochs INTEGER DEFAULT 100,
                    batch_size INTEGER DEFAULT 16,
                    img_size INTEGER DEFAULT 640,
                    learning_rate FLOAT DEFAULT 0.01,
                    config JSONB DEFAULT '{}',
                    status VARCHAR(50) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    metrics JSONB DEFAULT '{}'
                )
            """)

            # 训练日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_logs (
                    id SERIAL PRIMARY KEY,
                    experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
                    epoch INTEGER DEFAULT 0,
                    step INTEGER DEFAULT 0,
                    loss FLOAT DEFAULT 0.0,
                    metrics JSONB DEFAULT '{}',
                    lr FLOAT DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 用户表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE,
                    password_hash VARCHAR(255),
                    role VARCHAR(50) DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            """)

            # 模型向量存储表（用于语义搜索）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_embeddings (
                    id SERIAL PRIMARY KEY,
                    model_id VARCHAR(255) NOT NULL,
                    model_name VARCHAR(255),
                    embedding VECTOR(384),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 数据集向量存储表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_embeddings (
                    id SERIAL PRIMARY KEY,
                    dataset_name VARCHAR(255) NOT NULL,
                    embedding VECTOR(384),
                    sample_count INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_status
                ON experiments(status);
                CREATE INDEX IF NOT EXISTS idx_experiments_created
                ON experiments(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_training_logs_experiment
                ON training_logs(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_model_embeddings_model_id
                ON model_embeddings(model_id);
                CREATE INDEX IF NOT EXISTS idx_dataset_embeddings_name
                ON dataset_embeddings(dataset_name);
            """)

    # ==================== 实验管理 ====================

    def create_experiment(self, experiment: Experiment) -> int:
        """创建实验"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO experiments (
                    name, dataset_name, model_type, epochs, batch_size,
                    img_size, learning_rate, config, status
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """, (
                experiment.name, experiment.dataset_name, experiment.model_type,
                experiment.epochs, experiment.batch_size, experiment.img_size,
                experiment.learning_rate, Json(experiment.config), experiment.status
            ))
            return cursor.fetchone()['id']

    def get_experiment(self, experiment_id: int) -> Optional[Dict]:
        """获取实验详情"""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute("SELECT * FROM experiments WHERE id = %s", (experiment_id,))
            result = cursor.fetchone()
            return dict(result) if result else None

    def update_experiment_status(self, experiment_id: int, status: str,
                                  metrics: Optional[Dict] = None):
        """更新实验状态"""
        with self.get_cursor() as cursor:
            if status in ['completed', 'failed']:
                cursor.execute("""
                    UPDATE experiments
                    SET status = %s, completed_at = CURRENT_TIMESTAMP,
                        metrics = %s
                    WHERE id = %s
                """, (status, Json(metrics or {}), experiment_id))
            else:
                cursor.execute("""
                    UPDATE experiments SET status = %s WHERE id = %s
                """, (status, experiment_id))

    def list_experiments(self, status: Optional[str] = None,
                         limit: int = 50) -> List[Dict]:
        """列出实验"""
        with self.get_cursor(commit=False) as cursor:
            if status:
                cursor.execute("""
                    SELECT * FROM experiments
                    WHERE status = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (status, limit))
            else:
                cursor.execute("""
                    SELECT * FROM experiments
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    # ==================== 训练日志 ====================

    def add_training_log(self, log: TrainingLog):
        """添加训练日志"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO training_logs (
                    experiment_id, epoch, step, loss, metrics, lr
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                log.experiment_id, log.epoch, log.step, log.loss,
                Json(log.metrics), log.lr
            ))

    def get_training_logs(self, experiment_id: int) -> List[Dict]:
        """获取训练日志"""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute("""
                SELECT * FROM training_logs
                WHERE experiment_id = %s
                ORDER BY epoch, step
            """, (experiment_id,))
            return [dict(row) for row in cursor.fetchall()]

    # ==================== 用户管理 ====================

    def create_user(self, user: User) -> int:
        """创建用户"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO users (username, email, role)
                VALUES (%s, %s, %s) RETURNING id
            """, (user.username, user.email, user.role))
            return cursor.fetchone()['id']

    def get_user(self, user_id: Optional[int] = None,
                 username: Optional[str] = None) -> Optional[Dict]:
        """获取用户"""
        with self.get_cursor(commit=False) as cursor:
            if user_id:
                cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            elif username:
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            else:
                return None
            result = cursor.fetchone()
            return dict(result) if result else None

    def update_last_login(self, user_id: int):
        """更新最后登录时间"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
                (user_id,)
            )

    # ==================== 向量检索 (pgvector) ====================

    def add_model_embedding(self, model_id: str, model_name: str,
                            embedding: List[float],
                            metadata: Optional[Dict] = None):
        """添加模型向量"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO model_embeddings (model_id, model_name, embedding, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (model_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
            """, (model_id, model_name, embedding, Json(metadata or {})))

    def search_models(self, query_embedding: List[float],
                      limit: int = 5) -> List[Dict]:
        """语义搜索模型"""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute("""
                SELECT model_id, model_name, metadata,
                       1 - (embedding <=> %s) as similarity
                FROM model_embeddings
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_embedding, query_embedding, limit))
            return [dict(row) for row in cursor.fetchall()]

    def add_dataset_embedding(self, dataset_name: str,
                               embedding: List[float],
                               sample_count: int = 0,
                               metadata: Optional[Dict] = None):
        """添加数据集向量"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO dataset_embeddings (dataset_name, embedding, sample_count, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (dataset_name) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    sample_count = EXCLUDED.sample_count,
                    metadata = EXCLUDED.metadata
            """, (dataset_name, embedding, sample_count, Json(metadata or {})))

    def search_datasets(self, query_embedding: List[float],
                        limit: int = 5) -> List[Dict]:
        """语义搜索数据集"""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute("""
                SELECT dataset_name, sample_count, metadata,
                       1 - (embedding <=> %s) as similarity
                FROM dataset_embeddings
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_embedding, query_embedding, limit))
            return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """关闭连接"""
        if self._connection and not self._connection.closed:
            self._connection.close()


# 全局实例
postgres_service = PostgresService()
