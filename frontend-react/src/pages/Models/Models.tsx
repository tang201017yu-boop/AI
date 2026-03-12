import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardHeader, Button, StatCard } from '../../components/common';
import { modelApi } from '../../services/api';
import type { Model } from '../../types';

// 已加载模型类型
interface LoadedModel {
  cache_key: string;
  model_name: string;
  task: string;
  device: string;
  loaded: boolean;
  classes?: string[];
  num_classes?: number;
}

// 用户上传的模型类型
interface UserModel {
  id: string;
  name: string;
  path: string;
  source: string;
  project?: string;
  task?: string;
  classes?: string[];
  created_at?: string;
  model_type?: string;
}

export const Models: React.FC = () => {
  const navigate = useNavigate();
  const [models, setModels] = useState<UserModel[]>([]);
  const [loadedModels, setLoadedModels] = useState<LoadedModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingModels, setLoadingModels] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 默认模型列表
  const defaultModels = [
    { name: 'yolo11n.pt', task: '检测', desc: 'nano - 最小最快' },
    { name: 'yolo11s.pt', task: '检测', desc: 'small - 轻量快速' },
    { name: 'yolo11m.pt', task: '检测', desc: 'medium - 平衡推荐' },
    { name: 'yolo11l.pt', task: '检测', desc: 'large - 高精度' },
    { name: 'yolo11x.pt', task: '检测', desc: 'xlarge - 最大精度' },
    { name: 'yolo11n-seg.pt', task: '分割', desc: '实例分割' },
    { name: 'yolo11n-pose.pt', task: '姿态', desc: '人体姿态估计' },
    { name: 'yolo11n-obb.pt', task: '旋转', desc: '旋转框检测' },
    { name: 'yolo26n.pt', task: '检测', desc: 'YOLO26 nano' },
    { name: 'yolo26s.pt', task: '检测', desc: 'YOLO26 small' },
    { name: 'yolo26m.pt', task: '检测', desc: 'YOLO26 medium' },
  ];

  useEffect(() => {
    loadModels();
    loadLoadedModels();
  }, []);

  const loadModels = async () => {
    try {
      // 获取用户模型
      const res = await modelApi.getUserModels();
      const responseData = res.data as any;
      const allModels = responseData?.models || [];
      // 只显示上传的模型 (source === 'uploaded')
      const uploadedModels = allModels.filter((m: UserModel) => m.source === 'uploaded');
      setModels(uploadedModels);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const loadLoadedModels = async () => {
    try {
      const res = await modelApi.getLoadedModels();
      const responseData = res.data as any;
      const data = responseData?.models || [];
      setLoadedModels(data);
    } catch (error) {
      console.error('获取已加载模型失败:', error);
    }
  };

  // 上传模型
  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // 检查文件格式
    if (!file.name.endsWith('.pt')) {
      alert('仅支持 .pt 格式的 PyTorch 模型文件');
      return;
    }

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await modelApi.upload(formData);
      if (res.data?.success) {
        alert('模型上传成功');
        loadModels();
      } else {
        alert(res.data?.message || '上传失败');
      }
    } catch (error) {
      console.error('上传模型失败:', error);
      alert('上传模型失败');
    } finally {
      setUploading(false);
      // 清空文件输入
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  // 加载模型到内存（默认自动检测GPU）
  const handleLoadModel = async (modelName: string) => {
    setLoadingModels(true);
    try {
      const res = await modelApi.loadModel(modelName, 'auto');
      if (res.data?.success) {
        alert(`模型 ${modelName} 已加载到内存`);
        loadLoadedModels();
      } else {
        alert(res.data?.message || '加载失败');
      }
    } catch (error) {
      console.error('加载模型失败:', error);
      alert('加载模型失败');
    } finally {
      setLoadingModels(false);
    }
  };

  // 检查模型是否已加载
  const isModelLoaded = (modelName: string) => {
    return loadedModels.some(m => m.model_name.includes(modelName.replace('.pt', '')));
  };

  // 删除模型
  const handleDeleteModel = async (model: UserModel, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm(`确定要删除模型 "${model.name}" 吗？此操作不可恢复。`)) {
      return;
    }
    setDeletingId(model.id);
    try {
      await modelApi.delete(model.id);
      setModels(models.filter(m => m.id !== model.id));
      alert('模型删除成功');
    } catch (error) {
      console.error('删除模型失败:', error);
      alert('删除模型失败');
    } finally {
      setDeletingId(null);
    }
  };

  // 推理测试
  const handleInference = (model: UserModel, e: React.MouseEvent) => {
    e.stopPropagation();
    navigate(`/inference?model=${encodeURIComponent(model.path)}`);
  };

  return (
    <div>
      {/* 隐藏的文件输入 */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pt"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-6)' }}>
        <h1 style={{ fontFamily: 'DM Sans', fontWeight: 700 }}>模型管理</h1>
        <Button variant="primary" onClick={handleUploadClick} disabled={uploading}>
          {uploading ? '上传中...' : '上传模型'}
        </Button>
      </div>

      {/* 已加载模型状态 */}
      <Card style={{ marginBottom: 'var(--space-6)' }}>
        <CardHeader icon="⚡" title="已加载到内存的模型" />
        {loadedModels.length === 0 ? (
          <div style={{ textAlign: 'center', padding: 'var(--space-6)', color: 'var(--text-secondary)' }}>
            <p>暂无模型加载到内存</p>
            <p style={{ fontSize: '0.875rem' }}>点击下方模型进行加载，加载后推理速度更快</p>
          </div>
        ) : (
          <div style={{ display: 'flex', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
            {loadedModels.map((model, idx) => (
              <div
                key={idx}
                style={{
                  padding: '8px 16px',
                  background: '#10b981',
                  color: '#fff',
                  borderRadius: '20px',
                  fontSize: '13px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                }}
              >
                <span>✓</span>
                <span>{model.model_name}</span>
                <span style={{ opacity: 0.7, fontSize: '12px' }}>({model.device})</span>
              </div>
            ))}
          </div>
        )}
      </Card>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
        <StatCard value={models.length + defaultModels.length} label="可用模型" />
        <StatCard value={loadedModels.length} label="已加载" variant="accent" />
        <StatCard value="0" label="训练中" variant="accent" />
        <StatCard value="完成" label="训练完成" variant="success" />
      </div>

      {/* 默认模型列表 */}
      <Card style={{ marginBottom: 'var(--space-6)' }}>
        <CardHeader icon="🤖" title="快速加载模型（预训练）" />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 'var(--space-3)' }}>
          {defaultModels.map((model, idx) => {
            const isLoaded = isModelLoaded(model.name);
            return (
              <div
                key={idx}
                style={{
                  padding: '12px',
                  border: isLoaded ? '2px solid #10b981' : '1px solid #e2e8f0',
                  borderRadius: '8px',
                  background: isLoaded ? '#ecfdf5' : '#fff',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span style={{ fontWeight: 600, fontSize: '14px' }}>{model.name}</span>
                  {isLoaded && <span style={{ color: '#10b981', fontSize: '12px' }}>✓ 已加载</span>}
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                  <p>{model.task} - {model.desc}</p>
                </div>
                <Button
                  variant={isLoaded ? "secondary" : "primary"}
                  size="sm"
                  disabled={loadingModels || isLoaded}
                  onClick={() => handleLoadModel(model.name)}
                  style={{ width: '100%' }}
                >
                  {isLoaded ? '已加载' : loadingModels ? '加载中...' : '加载'}
                </Button>
              </div>
            );
          })}
        </div>
      </Card>

      {/* 用户上传的模型列表 */}
      {loading ? (
        <Card>
          <p style={{ color: 'var(--text-secondary)' }}>加载中...</p>
        </Card>
      ) : models.length === 0 ? (
        <Card>
          <CardHeader icon="📁" title="我的模型" />
          <div style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-4)' }}>
              暂无上传的模型，请上传 YOLO 模型文件 (.pt)
            </p>
            <Button variant="primary" onClick={handleUploadClick} disabled={uploading}>
              {uploading ? '上传中...' : '上传模型'}
            </Button>
          </div>
        </Card>
      ) : (
        <Card>
          <CardHeader icon="📁" title={`我的模型 (${models.length})`} />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 'var(--space-4)' }}>
            {models.map((model, idx) => (
              <div
                key={idx}
                style={{
                  padding: '16px',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-3)' }}>
                  <span style={{ fontSize: '1.5rem' }}>📦</span>
                  <div>
                    <h3 style={{ fontWeight: 600, margin: 0 }}>{model.name}</h3>
                    <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>{model.model_type || 'YOLO'}</span>
                  </div>
                </div>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: 'var(--space-3)' }}>
                  <p>路径: {model.path}</p>
                  <p>任务: {model.task || '检测'}</p>
                  {model.classes && <p>类别数: {model.classes.length}</p>}
                  {model.created_at && <p>上传时间: {new Date(model.created_at).toLocaleString()}</p>}
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                  <Button variant="secondary" size="sm" onClick={(e) => handleInference(model, e)}>推理</Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => handleDeleteModel(model, e)}
                    disabled={deletingId === model.id}
                  >
                    {deletingId === model.id ? '删除中...' : '删除'}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};
