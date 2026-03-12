import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, Button, Input } from '../../components/common';
import { annotationApi, inferenceApi, samApi } from '../../services/api';
import type { AnnotationProject, SAMAnnotation, AnnotationTool, AnnotationPoint, AnnotationBox, AnnotationMask, ClassSuggestion } from '../../types';
import { AnnotationCanvas, AnnotationToolbar, AnnotationPanel } from '../../components/Annotation';

export const Annotation: React.FC = () => {
  const [projects, setProjects] = useState<AnnotationProject[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDesc, setNewProjectDesc] = useState('');
  const [selectedProject, setSelectedProject] = useState<AnnotationProject | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [projectImages, setProjectImages] = useState<{ name: string; url: string }[]>([]);
  const [loadingImages, setLoadingImages] = useState(false);

  // 智能标注状态
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [autoLabelLoading, setAutoLabelLoading] = useState(false);
  const [autoLabelResult, setAutoLabelResult] = useState<any>(null);
  const [selectedModel, setSelectedModel] = useState('yolo11n.pt');
  const [confidence, setConfidence] = useState(0.25);
  const [showAutoLabel, setShowAutoLabel] = useState(false);

  // ============ SAM 标注功能状态 ============
  const [activeTab, setActiveTab] = useState<'yolo' | 'sam'>('yolo');
  const [samImage, setSamImage] = useState<string | null>(null);
  const [samImagePath, setSamImagePath] = useState<string>('');
  const [samFile, setSamFile] = useState<File | null>(null);
  const [samTool, setSamTool] = useState<AnnotationTool>('box');
  const [samCurrentClass, setSamCurrentClass] = useState('person');
  const [samClasses, setSamClasses] = useState(['person', 'car', 'dog', 'cat', 'bicycle', 'bird']);
  const [samPoints, setSamPoints] = useState<AnnotationPoint[]>([]);
  const [samBoxes, setSamBoxes] = useState<AnnotationBox[]>([]);
  const [samMasks, setSamMasks] = useState<AnnotationMask[]>([]);
  const [samAnnotations, setSamAnnotations] = useState<SAMAnnotation[]>([]);
  const [samSelectedId, setSamSelectedId] = useState<string | null>(null);
  const [samLoading, setSamLoading] = useState(false);
  const [samLoaded, setSamLoaded] = useState(false);
  const [samHistory, setSamHistory] = useState<{ points: AnnotationPoint[]; boxes: AnnotationBox[]; masks: AnnotationMask[] }[]>([]);
  const [samHistoryIndex, setSamHistoryIndex] = useState(-1);
  const [imgSize, setImgSize] = useState({ width: 800, height: 600 });

  // 类别建议（基于检测结果）
  const [classSuggestions, setClassSuggestions] = useState<ClassSuggestion[]>([]);

  // 初始化加载 SAM 模型
  useEffect(() => {
    const initSAM = async () => {
      try {
        const res = await samApi.loadModel('vit_b');
        setSamLoaded(res.data?.success || res.data?.data?.success || false);
      } catch (e) {
        console.error('SAM init failed:', e);
      }
    };
    if (activeTab === 'sam') {
      initSAM();
    }
  }, [activeTab]);

  // 保存历史记录
  const saveSamHistory = useCallback((newPoints: AnnotationPoint[], newBoxes: AnnotationBox[], newMasks: AnnotationMask[]) => {
    const newHistory = samHistory.slice(0, samHistoryIndex + 1);
    newHistory.push({ points: newPoints, boxes: newBoxes, masks: newMasks });
    setSamHistory(newHistory);
    setSamHistoryIndex(newHistory.length - 1);
  }, [samHistory, samHistoryIndex]);

  // 撤销
  const handleSamUndo = useCallback(() => {
    if (samHistoryIndex > 0) {
      const prev = samHistory[samHistoryIndex - 1];
      setSamPoints(prev.points);
      setSamBoxes(prev.boxes);
      setSamMasks(prev.masks);
      setSamHistoryIndex(samHistoryIndex - 1);
    }
  }, [samHistory, samHistoryIndex]);

  // 重做
  const handleSamRedo = useCallback(() => {
    if (samHistoryIndex < samHistory.length - 1) {
      const next = samHistory[samHistoryIndex + 1];
      setSamPoints(next.points);
      setSamBoxes(next.boxes);
      setSamMasks(next.masks);
      setSamHistoryIndex(samHistoryIndex + 1);
    }
  }, [samHistory, samHistoryIndex]);

  // 添加点
  const handleSamPointAdd = useCallback((point: AnnotationPoint) => {
    const newPoints = [...samPoints, point];
    setSamPoints(newPoints);
    saveSamHistory(newPoints, samBoxes, samMasks);
  }, [samPoints, samBoxes, samMasks, saveSamHistory]);

  // 添加框
  const handleSamBoxAdd = useCallback(async (box: AnnotationBox) => {
    const newBoxes = [...samBoxes, box];
    setSamBoxes(newBoxes);

    // 调用 SAM 框选分割
    if (samLoaded && samImagePath) {
      setSamLoading(true);
      try {
        const res = await samApi.predictBox([box.x1, box.y1, box.x2, box.y2]);
        const result = res.data?.data || res.data;
        if (result?.success && result.masks?.length > 0) {
          const newMasks = [...samMasks, {
            polygons: result.masks[0],
            color: getRandomColor()
          }];
          setSamMasks(newMasks);
          saveSamHistory(samPoints, newBoxes, newMasks);
        }
      } catch (e) {
        console.error('SAM predict failed:', e);
      } finally {
        setSamLoading(false);
      }
    } else {
      saveSamHistory(samPoints, newBoxes, samMasks);
    }
  }, [samBoxes, samMasks, samPoints, samLoaded, samImagePath, saveSamHistory]);

  // 清除
  const handleSamClear = useCallback(() => {
    setSamPoints([]);
    setSamBoxes([]);
    setSamMasks([]);
    setSamAnnotations([]);
    saveSamHistory([], [], []);
  }, [saveSamHistory]);

  // 自动标注
  const handleSamAutoLabel = useCallback(async () => {
    if (!samFile) return;

    setSamLoading(true);
    try {
      // 使用批量同类标注 - 传入文件对象
      const res = await samApi.batchSamLabel(samFile, samCurrentClass);
      const result = res.data?.data || res.data;

      if (result?.success) {
        setSamAnnotations(result.annotations || []);

        // 生成掩码显示
        const newMasks = (result.annotations || []).map((ann: SAMAnnotation, idx: number) => ({
          polygons: ann.segmentation?.split(' ').map(Number) || [],
          color: getRandomColor(),
        }));
        setSamMasks(newMasks);
        saveSamHistory(samPoints, samBoxes, newMasks);

        // 更新类别建议
        const classCounts: Record<string, number> = {};
        const colors: Record<string, string> = {};
        (result.annotations || []).forEach((ann: SAMAnnotation) => {
          classCounts[ann.class] = (classCounts[ann.class] || 0) + 1;
          if (!colors[ann.class]) {
            colors[ann.class] = getRandomColor();
          }
        });

        const suggestions: ClassSuggestion[] = Object.entries(classCounts).map(([name, count]) => ({
          name,
          count,
          color: colors[name],
        }));

        setClassSuggestions(suggestions);

        // 自动添加未存在的类别
        Object.keys(classCounts).forEach(cls => {
          if (!samClasses.includes(cls)) {
            setSamClasses(prev => [...prev, cls]);
          }
        });
      } else {
        alert(result?.message || '标注失败');
      }
    } catch (e) {
      console.error('Auto label failed:', e);
      alert('标注失败，请重试');
    } finally {
      setSamLoading(false);
    }
  }, [samFile, samCurrentClass, samPoints, samBoxes, samClasses, saveSamHistory]);

  // 处理 SAM 图片上传
  const handleSamFileSelect = async (files: File[]) => {
    if (files.length === 0) return;
    const file = files[0];
    const imageUrl = URL.createObjectURL(file);
    setSamImage(imageUrl);
    setSamImagePath(file.name);
    setSamFile(file);

    // 获取图片尺寸
    const img = new Image();
    img.onload = () => {
      setImgSize({ width: img.width, height: img.height });
    };
    img.src = imageUrl;
  };

  // 删除标注
  const handleSamDelete = (id: string) => {
    const idx = parseInt(id);
    const newAnnotations = samAnnotations.filter((_, i) => i !== idx);
    setSamAnnotations(newAnnotations);
  };

  // 类别修改
  const handleSamClassChange = (id: string, newClass: string) => {
    const idx = parseInt(id);
    setSamAnnotations(samAnnotations.map((a, i) => i === idx ? { ...a, class: newClass } : a));
  };

  // 随机颜色
  const getRandomColor = () => {
    const colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899'];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  // 导出
  const handleSamExport = () => {
    console.log('导出标注:', samAnnotations);
    // 可以调用导出 API
    alert(`已导出 ${samAnnotations.length} 个标注到 YOLO 格式`);
  };

  // 添加新类别
  const handleAddClass = (className: string) => {
    if (!samClasses.includes(className)) {
      setSamClasses([...samClasses, className]);
      setSamCurrentClass(className);
    }
  };

  // 删除选中的标注
  const handleDeleteSelected = () => {
    if (samSelectedId) {
      const idx = parseInt(samSelectedId);
      const newAnnotations = samAnnotations.filter((_, i) => i !== idx);
      setSamAnnotations(newAnnotations);
      setSamSelectedId(null);
    }
  };

  // 保存标注
  const handleSave = async () => {
    if (!selectedProject) {
      alert('请先选择一个项目');
      return;
    }

    if (!samFile || !samAnnotations.length) {
      alert('没有可保存的标注');
      return;
    }

    try {
      // 1. 上传图片到项目
      const uploadRes = await annotationApi.addImages(selectedProject.id || selectedProject.name, [samFile]);
      console.log('图片上传结果:', uploadRes);

      if (!uploadRes.data?.success && !uploadRes.data?.data?.success) {
        alert('图片上传失败: ' + (uploadRes.data?.message || '未知错误'));
        return;
      }

      // 2. 保存标注
      const imageName = samFile.name;
      const annotationsToSave = samAnnotations.map(ann => ({
        class: ann.class,
        class_id: ann.class_id,
        bbox: ann.bbox,
        segmentation: ann.segmentation,
        confidence: ann.confidence,
      }));

      const saveRes = await annotationApi.saveAnnotations(
        selectedProject.id || selectedProject.name,
        imageName,
        annotationsToSave
      );

      console.log('标注保存结果:', saveRes);

      if (saveRes.data?.success || saveRes.data?.data?.success) {
        alert(`成功保存 ${samAnnotations.length} 个标注到项目 "${selectedProject.name}"`);
        // 刷新项目图片列表
        const projectId = selectedProject.id || selectedProject.name;
        const res = await annotationApi.getImages(projectId);
        const data = res.data?.images || res.data?.data?.images || [];
        setProjectImages(data);
      } else {
        alert('保存标注失败: ' + (saveRes.data?.message || '未知错误'));
      }
    } catch (error) {
      console.error('保存失败:', error);
      alert('保存失败，请重试');
    }
  };

  // 键盘快捷键
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // 如果正在输入，不触发快捷键
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      // 快捷键处理
      if (activeTab === 'sam') {
        // Ctrl+Z: 撤销
        if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
          e.preventDefault();
          handleSamUndo();
        }
        // Ctrl+Y: 重做
        else if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
          e.preventDefault();
          handleSamRedo();
        }
        // Delete: 删除选中
        else if (e.key === 'Delete' || e.key === 'Backspace') {
          if (samSelectedId) {
            e.preventDefault();
            handleDeleteSelected();
          }
        }
        // P: 点标注
        else if (e.key === 'p' || e.key === 'P') {
          e.preventDefault();
          setSamTool('point');
        }
        // B: 框选
        else if (e.key === 'b' || e.key === 'B') {
          e.preventDefault();
          setSamTool('box');
        }
        // S: 选择
        else if (e.key === 's' || e.key === 'S') {
          if (!e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            setSamTool('select');
          }
        }
        // A: 自动标注
        else if (e.key === 'a' || e.key === 'A') {
          if (!e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            handleSamAutoLabel();
          }
        }
        // L: 多边形
        else if (e.key === 'l' || e.key === 'L') {
          e.preventDefault();
          setSamTool('polygon');
        }
        // 数字键 1-9: 快速选择类别
        else if (e.key >= '1' && e.key <= '9') {
          const idx = parseInt(e.key) - 1;
          if (idx < samClasses.length) {
            setSamCurrentClass(samClasses[idx]);
          }
        }
        // Esc: 取消选择
        else if (e.key === 'Escape') {
          setSamSelectedId(null);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeTab, samSelectedId, samClasses, handleSamUndo, handleSamRedo, handleDeleteSelected, handleSamAutoLabel]);

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    try {
      const res = await annotationApi.listProjects();
      const data = res.data?.data || res.data;
      setProjects(Array.isArray(data) ? data : []);
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

  // 删除项目
  const handleDeleteProject = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('确定要删除这个标注项目吗？此操作不可恢复。')) {
      return;
    }
    setDeletingId(id);
    try {
      await annotationApi.deleteProject(id);
      setProjects(projects.filter(p => p.id !== id));
      if (selectedProject?.id === id) {
        setSelectedProject(null);
      }
    } catch (error) {
      console.error('删除项目失败:', error);
      alert('删除项目失败，请重试');
    } finally {
      setDeletingId(null);
    }
  };

  // 选择项目
  const handleSelectProject = async (project: AnnotationProject) => {
    setSelectedProject(project);
    setActiveTab('sam');

    // 如果项目有预定义类别，加载它们
    if (project.classes && project.classes.length > 0) {
      setSamClasses(project.classes);
      setSamCurrentClass(project.classes[0]);
    }

    // 清除当前标注状态
    setSamPoints([]);
    setSamBoxes([]);
    setSamMasks([]);
    setSamAnnotations([]);
    setSamImage(null);
    setSamImagePath('');
    setSamFile(null);

    // 加载项目图片
    setLoadingImages(true);
    try {
      const projectId = project.id || project.name;
      const res = await annotationApi.getImages(projectId);
      const data = res.data?.images || res.data?.data?.images || [];
      setProjectImages(data);
    } catch (error) {
      console.error('加载项目图片失败:', error);
      setProjectImages([]);
    } finally {
      setLoadingImages(false);
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

      const result = res.data?.data || res.data as any;
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
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
          <CardHeader icon="✏️" title="标注项目列表" />
          <button
            onClick={loadProjects}
            disabled={loading}
            style={{
              padding: '6px 12px',
              border: '1px solid #e2e8f0',
              borderRadius: '6px',
              background: '#fff',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '13px',
            }}
            title="刷新列表"
          >
            {loading ? '⏳' : '🔄'}
          </button>
        </div>
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
              <div
                key={project.id}
                onClick={() => handleSelectProject(project)}
                style={{
                  padding: '16px',
                  borderRadius: '8px',
                  border: selectedProject?.id === project.id ? '2px solid #3b82f6' : '1px solid #e2e8f0',
                  background: selectedProject?.id === project.id ? '#eff6ff' : '#fff',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <h3 style={{ fontWeight: 600, marginBottom: 'var(--space-1)', fontSize: '16px' }}>{project.name}</h3>
                  <button
                    onClick={(e) => handleDeleteProject(project.id!, e)}
                    disabled={deletingId === project.id}
                    style={{
                      border: 'none',
                      background: 'transparent',
                      cursor: deletingId === project.id ? 'not-allowed' : 'pointer',
                      padding: '4px',
                      borderRadius: '4px',
                      color: '#94a3b8',
                      fontSize: '16px',
                    }}
                    title="删除项目"
                  >
                    {deletingId === project.id ? '⏳' : '🗑️'}
                  </button>
                </div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: 'var(--space-3)' }}>
                  {project.description || '无描述'}
                </p>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 'var(--space-3)' }}>
                  <p>创建时间: {new Date(project.created_at).toLocaleDateString()}</p>
                  {project.classes && <p>类别数: {project.classes.length}</p>}
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                  <Button
                    variant={selectedProject?.id === project.id ? "primary" : "secondary"}
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSelectProject(project);
                    }}
                  >
                    {selectedProject?.id === project.id ? '已选中' : '开始标注'}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* SAM 分割标注 */}
      <Card style={{ marginTop: 'var(--space-6)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
          <CardHeader icon="✂️" title="SAM 分割标注" />
          <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
            <button
              onClick={() => setActiveTab('yolo')}
              style={{
                padding: '8px 16px',
                border: 'none',
                borderRadius: '6px',
                background: activeTab === 'yolo' ? '#3b82f6' : '#f1f5f9',
                color: activeTab === 'yolo' ? '#fff' : '#475569',
                cursor: 'pointer',
                fontWeight: 500,
              }}
            >
              YOLO 预标注
            </button>
            <button
              onClick={() => setActiveTab('sam')}
              style={{
                padding: '8px 16px',
                border: 'none',
                borderRadius: '6px',
                background: activeTab === 'sam' ? '#3b82f6' : '#f1f5f9',
                color: activeTab === 'sam' ? '#fff' : '#475569',
                cursor: 'pointer',
                fontWeight: 500,
              }}
            >
              SAM 分割
            </button>
          </div>
        </div>

        {activeTab === 'sam' && (
          <div>
            {/* 项目信息 */}
            {selectedProject && (
              <div style={{
                padding: '12px 16px',
                background: '#f0f9ff',
                borderRadius: '8px',
                marginBottom: '16px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}>
                <div>
                  <span style={{ fontWeight: 600, color: '#0369a1' }}>
                    当前项目: {selectedProject.name}
                  </span>
                  <span style={{ marginLeft: '16px', color: '#64748b', fontSize: '13px' }}>
                    {projectImages.length} 张已标注图片
                  </span>
                </div>
                <button
                  onClick={() => {
                    setSelectedProject(null);
                    setSamImage(null);
                    setSamImagePath('');
                    setSamFile(null);
                    setSamAnnotations([]);
                  }}
                  style={{
                    padding: '6px 12px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '6px',
                    background: '#fff',
                    cursor: 'pointer',
                    fontSize: '13px',
                  }}
                >
                  关闭项目
                </button>
              </div>
            )}

            {/* 项目图片列表 */}
            {selectedProject && projectImages.length > 0 && (
              <div style={{ marginBottom: '16px' }}>
                <h4 style={{ fontSize: '14px', marginBottom: '8px', color: '#64748b' }}>已标注图片:</h4>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', maxHeight: '120px', overflowX: 'auto' }}>
                  {projectImages.map((img, idx) => (
                    <div
                      key={idx}
                      style={{
                        width: '80px',
                        height: '80px',
                        borderRadius: '6px',
                        overflow: 'hidden',
                        border: '2px solid #e2e8f0',
                        cursor: 'pointer',
                      }}
                      title={img.name}
                    >
                      <img
                        src={img.url}
                        alt={img.name}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 工具栏 */}
            <AnnotationToolbar
              tool={samTool}
              currentClass={samCurrentClass}
              classes={samClasses}
              suggestions={classSuggestions}
              onToolChange={setSamTool}
              onClassChange={setSamCurrentClass}
              onAddClass={handleAddClass}
              onAutoLabel={handleSamAutoLabel}
              onClear={handleSamClear}
              onUndo={handleSamUndo}
              onRedo={handleSamRedo}
              onDeleteSelected={handleDeleteSelected}
              onSave={handleSave}
              loading={samLoading}
            />

            {/* 主工作区 */}
            <div style={{ display: 'flex', gap: '24px', marginTop: '24px' }}>
              {/* 画布区域 */}
              <div style={{ flex: 1 }}>
                {samImage ? (
                  <AnnotationCanvas
                    image={samImage}
                    width={imgSize.width}
                    height={imgSize.height}
                    points={samPoints}
                    boxes={samBoxes}
                    masks={samMasks}
                    annotations={samAnnotations}
                    tool={samTool}
                    selectedId={samSelectedId}
                    onPointAdd={handleSamPointAdd}
                    onBoxAdd={handleSamBoxAdd}
                    onMaskSelect={(idx) => setSamSelectedId(String(idx))}
                    onAnnotationSelect={setSamSelectedId}
                  />
                ) : (
                  <div style={{
                    width: '100%',
                    height: '400px',
                    border: '2px dashed #e2e8f0',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: '#f8fafc',
                  }}>
                    <label style={{ cursor: 'pointer', textAlign: 'center' }}>
                      <div style={{ fontSize: '48px', marginBottom: '16px' }}>📁</div>
                      <div>点击或拖拽上传图片</div>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={(e) => handleSamFileSelect(e.target.files ? Array.from(e.target.files) : [])}
                        style={{ display: 'none' }}
                      />
                    </label>
                  </div>
                )}
              </div>

              {/* 标注面板 */}
              <AnnotationPanel
                annotations={samAnnotations}
                selectedId={samSelectedId}
                onSelect={setSamSelectedId}
                onDelete={handleSamDelete}
                onClassChange={handleSamClassChange}
                onExport={handleSamExport}
                classes={samClasses}
              />
            </div>

            {/* 说明 */}
            <div style={{ marginTop: '24px', color: '#64748b', fontSize: '14px' }}>
              <p><strong>使用说明:</strong></p>
              <ul>
                <li>点击图片添加正样本点 (绿色 +)</li>
                <li>Shift + 点击添加负样本点 (红色 -)</li>
                <li>选择框选工具拖动画框进行分割</li>
                <li>选择自动标注并指定类别进行批量标注</li>
              </ul>
            </div>
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
