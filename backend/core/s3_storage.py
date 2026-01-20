"""
S3 存储服务 - 支持 MinIO 和 AWS S3
用于存储大文件：数据集图片、模型权重文件、导出文件
"""
import os
import hashlib
from typing import Optional, BinaryIO, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


@dataclass
class S3Config:
    """S3 配置"""
    endpoint_url: str = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
    access_key: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
    secret_key: str = os.getenv("S3_SECRET_KEY", "minioadmin")
    bucket_name: str = os.getenv("S3_BUCKET", "opencv-platform")
    region: str = os.getenv("S3_REGION", "us-east-1")
    use_ssl: bool = os.getenv("S3_USE_SSL", "false").lower() == "true"


@dataclass
class S3FileInfo:
    """S3 文件信息"""
    key: str
    size: int
    last_modified: datetime
    etag: str
    url: str


class S3StorageService:
    """S3 存储服务"""

    # 存储桶名称
    BUCKET_DATASETS = "datasets"
    BUCKET_MODELS = "models"
    BUCKET_EXPORTS = "exports"
    BUCKET_UPLOADS = "uploads"

    def __init__(self, config: Optional[S3Config] = None):
        self.config = config or S3Config()
        self._client = None
        self._resource = None

    @property
    def client(self):
        """获取 S3 客户端"""
        if self._client is None:
            self._client = boto3.client(
                's3',
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                region_name=self.config.region,
                config=Config(
                    signature_version='s3v4',
                    s3={'addressing_style': 'path'}
                )
            )
        return self._client

    @property
    def resource(self):
        """获取 S3 资源"""
        if self._resource is None:
            self._resource = boto3.resource(
                's3',
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                region_name=self.config.region,
                config=Config(
                    signature_version='s3v4',
                    s3={'addressing_style': 'path'}
                )
            )
        return self._resource

    def _ensure_bucket_exists(self, bucket_name: str):
        """确保存储桶存在"""
        try:
            self.client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404':
                # 创建存储桶
                self.client.create_bucket(Bucket=bucket_name)
            else:
                raise

    def init_storage(self):
        """初始化存储"""
        for bucket in [self.BUCKET_DATASETS, self.BUCKET_MODELS,
                       self.BUCKET_EXPORTS, self.BUCKET_UPLOADS]:
            self._ensure_bucket_exists(bucket)

    def _calculate_etag(self, file_path: Path) -> str:
        """计算文件的 ETag（用于去重）"""
        md5_hash = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    # ==================== 文件上传 ====================

    def upload_file(self, file_path: str, bucket: str, key: str,
                    extra_args: Optional[Dict] = None) -> Dict:
        """
        上传文件

        Args:
            file_path: 本地文件路径
            bucket: 存储桶名称
            key: S3 中的键名
            extra_args: 额外参数（ContentType, Metadata 等）

        Returns:
            上传结果信息
        """
        file_path = Path(file_path)
        extra_args = extra_args or {}

        # 自动检测内容类型
        if 'ContentType' not in extra_args:
            content_type = self._get_content_type(file_path.suffix)
            if content_type:
                extra_args['ContentType'] = content_type

        # 计算 ETag 用于去重
        etag = self._calculate_etag(file_path)
        extra_args['Metadata'] = {
            'etag': etag,
            'original_name': file_path.name
        }

        try:
            self.client.upload_file(
                str(file_path),
                bucket,
                key,
                ExtraArgs=extra_args
            )

            return {
                'success': True,
                'bucket': bucket,
                'key': key,
                'size': file_path.stat().st_size,
                'etag': etag,
                'url': f"{self.config.endpoint_url}/{bucket}/{key}"
            }
        except ClientError as e:
            return {
                'success': False,
                'error': str(e)
            }

    def upload_bytes(self, data: bytes, bucket: str, key: str,
                     content_type: str = "application/octet-stream") -> Dict:
        """上传字节数据"""
        try:
            self.client.put_object(
                Body=data,
                Bucket=bucket,
                Key=key,
                ContentType=content_type
            )

            return {
                'success': True,
                'bucket': bucket,
                'key': key,
                'size': len(data),
                'url': f"{self.config.endpoint_url}/{bucket}/{key}"
            }
        except ClientError as e:
            return {
                'success': False,
                'error': str(e)
            }

    # ==================== 文件下载 ====================

    def download_file(self, bucket: str, key: str, output_path: str) -> bool:
        """下载文件到本地"""
        try:
            self.client.download_file(bucket, key, output_path)
            return True
        except ClientError as e:
            print(f"下载失败: {e}")
            return False

    def download_bytes(self, bucket: str, key: str) -> Optional[bytes]:
        """下载文件为字节数据"""
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except ClientError:
            return None

    def get_presigned_url(self, bucket: str, key: str,
                          expires: int = 3600) -> str:
        """获取预签名 URL"""
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': bucket,
                    'Key': key
                },
                ExpiresIn=expires
            )
            return url
        except ClientError:
            return ""

    # ==================== 文件管理 ====================

    def delete_file(self, bucket: str, key: str) -> bool:
        """删除文件"""
        try:
            self.client.delete_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    def file_exists(self, bucket: str, key: str) -> bool:
        """检查文件是否存在"""
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    def get_file_info(self, bucket: str, key: str) -> Optional[S3FileInfo]:
        """获取文件信息"""
        try:
            response = self.client.head_object(Bucket=bucket, Key=key)
            return S3FileInfo(
                key=key,
                size=response['ContentLength'],
                last_modified=response['LastModified'],
                etag=response['ETag'].strip('"'),
                url=f"{self.config.endpoint_url}/{bucket}/{key}"
            )
        except ClientError:
            return None

    def list_files(self, bucket: str, prefix: str = "",
                   max_keys: int = 1000) -> List[S3FileInfo]:
        """列出文件"""
        try:
            response = self.client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )

            files = []
            for obj in response.get('Contents', []):
                files.append(S3FileInfo(
                    key=obj['Key'],
                    size=obj['Size'],
                    last_modified=obj['LastModified'],
                    etag=obj['ETag'].strip('"'),
                    url=f"{self.config.endpoint_url}/{bucket}/{obj['Key']}"
                ))
            return files
        except ClientError:
            return []

    # ==================== 存储桶操作 ====================

    def get_bucket_stats(self, bucket: str) -> Dict:
        """获取存储桶统计信息"""
        try:
            response = self.client.list_objects_v2(Bucket=bucket)
            objects = response.get('Contents', [])

            total_size = sum(obj['Size'] for obj in objects)
            file_count = len(objects)

            return {
                'bucket': bucket,
                'file_count': file_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
        except ClientError:
            return {'bucket': bucket, 'file_count': 0, 'total_size_bytes': 0}

    # ==================== 工具方法 ====================

    def _get_content_type(self, suffix: str) -> str:
        """根据文件后缀获取内容类型"""
        content_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
            '.pt': 'application/octet-stream',
            '.pth': 'application/octet-stream',
            '.onnx': 'application/octet-stream',
            '.engine': 'application/octet-stream',
            '.zip': 'application/zip',
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.yaml': 'application/x-yaml',
            '.yml': 'application/x-yaml'
        }
        return content_types.get(suffix.lower())

    # ==================== 便捷方法 ====================

    def upload_dataset_file(self, file_path: str, dataset_name: str,
                            relative_path: str) -> Dict:
        """上传数据集文件"""
        key = f"{dataset_name}/{relative_path}"
        return self.upload_file(file_path, self.BUCKET_DATASETS, key)

    def upload_model_file(self, file_path: str, model_name: str,
                          relative_path: str) -> Dict:
        """上传模型文件"""
        key = f"{model_name}/{relative_path}"
        return self.upload_file(file_path, self.BUCKET_MODELS, key)

    def upload_export_file(self, file_path: str, model_name: str,
                           format_name: str) -> Dict:
        """上传导出文件"""
        key = f"{model_name}/{format_name}/{Path(file_path).name}"
        return self.upload_file(file_path, self.BUCKET_EXPORTS, key)

    def get_dataset_file(self, dataset_name: str, relative_path: str) -> Optional[bytes]:
        """获取数据集文件"""
        key = f"{dataset_name}/{relative_path}"
        return self.download_bytes(self.BUCKET_DATASETS, key)

    def get_model_file(self, model_name: str, relative_path: str) -> Optional[bytes]:
        """获取模型文件"""
        key = f"{model_name}/{relative_path}"
        return self.download_bytes(self.BUCKET_MODELS, key)

    def list_dataset_files(self, dataset_name: str) -> List[S3FileInfo]:
        """列出数据集文件"""
        return self.list_files(self.BUCKET_DATASETS, prefix=f"{dataset_name}/")

    def list_model_files(self, model_name: str) -> List[S3FileInfo]:
        """列出模型文件"""
        return self.list_files(self.BUCKET_MODELS, prefix=f"{model_name}/")


# 全局实例
s3_service = S3StorageService()
