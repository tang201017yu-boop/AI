import React, { useState, useEffect } from 'react';
import { Card, CardHeader, Button, Input, Select } from '../../components/common';
import { annotationApi, inferenceApi } from '../../services/api';
import type { AnnotationProject } from '../../types';

export const Annotation: React.FC = () => {
  const [projects, setProjects] = useState<AnnotationProject[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDesc, setNewProjectDesc] = useState('');

  // 智能标注状态
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [autoLabelLoading, setAutoLabelLoading] = useState(false);
  const [autoLabelResult, setAutoLabelResult] = useState<any>(null);
  const [selectedModel, setSelectedModel] = useState('yolo11n.pt');
  const [confidence, setConfidence] = useState(0.25);
  const [showAutoLabel, setShowAutoLabel] = useState(false);

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    try {
      const res = await annotationApi.listProjects();
      setProjects(res.data?.data || res.data || []);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateProject = async () => {
    if (!newProjectName) return;
    try {
      await annotationApi.createProject({
        name: newProjectName,
        description: newProjectDesc,
      });
      setNewProjectName('');
      setNewProjectDesc('');
      setShowCreate(false);
      loadProjects();
    } catch (error) {
      console.error(error);
    }
  };

  // 处理文件选择并自动标注
  const handleFileSelect = async (files: File[]) => {
    if (files.length === 0) return;

    const file = files[0];
    const imageUrl = URL.createObjectURL(file);
    setSelectedImage(imageUrl);
    setAutoLabelResult(null);
    setShowAutoLabel(true);

    // 自动触发智能标注
    await handleAutoLabel(file);
  };

  // 智能标注
  const handleAutoLabel = async (file?: File) => {
    if (!file) return;

    setAutoLabelLoading(true);
    try {
      // 调用推理接口获取检测结果
      const res = await inferenceApi.image(
        file,
        selectedModel,
        confidence
      );

      const result = res.data?.data || res.data;
      if (result && result.detections) {
        setAutoLabelResult({
          success: true,
          detections: result.detections,
          annotated_image: result.annotated_image,
          inference_time: result.inference_time
        });
      }
    } catch (error) {
      console.error('智能标注失败:', error);
      setAutoLabelResult({
        success: false,
        message: '标注失败'
      });
    } finally {
      setAutoLabelLoading(false);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-6)' }}>
        <h1 style={{ fontFamily: 'DM Sans', fontWeight: 700 }}>智能标注</h1>
        <Button variant="primary" onClick={() => setShowCreate(!showCreate)}>
          {showCreate ? '取消' : '创建项目'}
        </Button>
      </div>

      {/* 创建项目 */}
      {showCreate && (
        <Card style={{ marginBottom: 'var(--space-6)' }}>
          <CardHeader icon="➕" title="创建标注项目" />
          <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
            <Input
              label="项目名称"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="输入项目名称"
            />
            <Input
              label="项目描述"
              value={newProjectDesc}
              onChange={(e) => setNewProjectDesc(e.target.value)}
              placeholder="输入项目描述（可选）"
            />
            <Button variant="primary" onClick={handleCreateProject}>
              创建
            </Button>
          </div>
        </Card>
      )}

      {/* 智能标注工具 */}
      <Card style={{ marginBottom: 'var(--space-6)' }}>
        <CardHeader icon="🤖" title="YOLO 智能预标注" />
        <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
          {/* 配置选项 */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 'var(--space-4)' }}>
            <div>
              <label style={{ display: 'block', marginBottom: 'var(--space-1)', fontWeight: 500 }}>
                检测模型
              </label>
              <select
                className="form-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                style={{ width: '100%', padding: 'var(--space-2)', borderRadius: 'var(--radius-md)' }}
              >
                <optgroup label="YOLO26 (最新)">
                  <option value="yolo26n.pt">YOLO26n - 速度最快</option>
                  <option value="yolo26s.pt">YOLO26s - 轻量快速</option>
                  <option value="yolo26m.pt">YOLO26m - 平衡推荐</option>
                </optgroup>
                <optgroup label="YOLO11 (经典)">
                  <option value="yolo11n.pt">YOLO11n</option>
                  <option value="yolo11s.pt">YOLO11s</option>
                  <option value="yolo11m.pt">YOLO11m</option>
                </optgroup>
              </select>
            </div>
            <Input
              type="number"
              label="置信度阈值"
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
              min={0}
              max={1}
              step={0.05}
            />
            <div style={{ display: 'flex', alignItems: 'flex-end' }}>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 'var(--space-2)',
                  cursor: 'pointer',
                  padding: 'var(--space-2) var(--space-3)',
                  background: 'var(--primary-50)',
                  border: '1px dashed var(--primary-300)',
                  borderRadius: 'var(--radius-md)',
                  width: '100%',
                  justifyContent: 'center'
                }}
              >
                <span style={{ fontSize: '1.5rem' }}>📁</span>
                <span>选择图片或拖拽</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleFileSelect(e.target.files ? Array.from(e.target.files) : [])}
                  style={{ display: 'none' }}
                />
              </label>
            </div>
          </div>

          {/* 标注结果显示 */}
          {selectedImage && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)', marginTop: 'var(--space-4)' }}>
              {/* 原图 */}
              <div>
                <h4 style={{ marginBottom: 'var(--space-2)' }}>📷 原图</h4>
                <div style={{
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-lg)',
                  overflow: 'hidden',
                  background: 'var(--gray-50)'
                }}>
                  <img
                    src={selectedImage}
                    alt="Original"
                    style={{ width: '100%', display: 'block' }}
                  />
                </div>
              </div>

              {/* 标注结果 */}
              <div>
                <h4 style={{ marginBottom: 'var(--space-2)' }}>🎯 智能标注结果</h4>
                <div style={{
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-lg)',
                  overflow: 'hidden',
                  background: 'var(--gray-50)',
                  minHeight: '300px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  {autoLabelLoading ? (
                    <div style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
                      <div style={{ fontSize: '2rem', marginBottom: 'var(--space-2)' }}>⏳</div>
                      <p>正在智能标注中...</p>
                    </div>
                  ) : autoLabelResult?.annotated_image ? (
                    <img
                      src={autoLabelResult.annotated_image}
                      alt="Annotated"
                      style={{ width: '100%', display: 'block' }}
                    />
                  ) : autoLabelResult?.detections ? (
                    <img
                      src={selectedImage}
                      alt="Original"
                      style={{ width: '100%', display: 'block' }}
                    />
                  ) : (
                    <div style={{ textAlign: 'center', padding: 'var(--space-8)', color: 'var(--text-secondary)' }}>
                      <div style={{ fontSize: '2rem', marginBottom: 'var(--space-2)' }}>📷</div>
                      <p>上传图片自动检测</p>
                    </div>
                  )}
                </div>

                {/* 检测结果统计 */}
                {autoLabelResult?.detections && (
                  <div style={{ marginTop: 'var(--space-3)' }}>
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      padding: 'var(--space-2) var(--space-3)',
                      background: 'var(--primary-50)',
                      borderRadius: 'var(--radius-md)',
                      marginBottom: 'var(--space-2)'
                    }}>
                      <span>检测到 <strong>{autoLabelResult.detections.length}</strong> 个对象</span>
                      <span>⏱️ {autoLabelResult.inference_time?.toFixed(2)}s</span>
                    </div>

                    {/* 类别统计 */}
                    <div style={{ maxHeight: '150px', overflowY: 'auto' }}>
                      {Object.entries(
                        autoLabelResult.detections.reduce((acc: any, det: any) => {
                          acc[det.class_name] = (acc[det.class_name] || 0) + 1;
                          return acc;
                        }, {})
                      ).map(([className, count]: [string, any]) => (
                        <div key={className} style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          padding: 'var(--space-1) var(--space-2)',
                          borderBottom: '1px solid var(--gray-100)'
                        }}>
                          <span style={{ textTransform: 'capitalize' }}>{className}</span>
                          <span style={{ color: 'var(--primary-600)', fontWeight: 600 }}>× {count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* 项目列表 */}
      <Card>
        <CardHeader icon="✏️" title="标注项目列表" />
        {loading ? (
          <p style={{ color: 'var(--text-secondary)' }}>加载中...</p>
        ) : projects.length === 0 ? (
          <div style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-4)' }}>
              暂无标注项目，请创建一个新项目
            </p>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
              支持 YOLO 自动预标注和 SAM 分割标注
            </p>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 'var(--space-4)' }}>
            {projects.map((project) => (
              <Card key={project.id} variant="elevated">
                <h3 style={{ fontWeight: 600, marginBottom: 'var(--space-1)' }}>{project.name}</h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: 'var(--space-3)' }}>
                  {project.description || '无描述'}
                </p>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 'var(--space-3)' }}>
                  <p>创建时间: {new Date(project.created_at).toLocaleDateString()}</p>
                  {project.classes && <p>类别数: {project.classes.length}</p>}
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                  <Button variant="primary" size="sm">开始标注</Button>
                  <Button variant="secondary" size="sm">自动标注</Button>
                </div>
              </Card>
            ))}
          </div>
        )}
      </Card>

      {/* 功能说明 */}
      <Card style={{ marginTop: 'var(--space-6)' }}>
        <CardHeader icon="📖" title="功能说明" />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-4)' }}>
          <div style={{ padding: 'var(--space-3)', background: 'var(--gray-50)', borderRadius: 'var(--radius-md)' }}>
            <h4 style={{ marginBottom: 'var(--space-2)', display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <span>🎯</span> YOLO 预标注
            </h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
              使用 YOLO 模型自动检测图片中的目标物体，一键生成检测框和类别标签
            </p>
          </div>
          <div style={{ padding: 'var(--space-3)', background: 'var(--gray-50)', borderRadius: 'var(--radius-md)' }}>
            <h4 style={{ marginBottom: 'var(--space-2)', display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <span>✏️</span> 手动修正
            </h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
              在自动标注基础上进行手动修正，添加、删除或调整检测框
            </p>
          </div>
          <div style={{ padding: 'var(--space-3)', background: 'var(--gray-50)', borderRadius: 'var(--radius-md)' }}>
            <h4 style={{ marginBottom: 'var(--space-2)', display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <span>📦</span> 一键导出
            </h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
              导出为 YOLO 格式训练数据，直接用于模型训练
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
};
