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
  const [, setLoadingImages] = useState(false);

  // 智能标注状态
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [autoLabelLoading, setAutoLabelLoading] = useState(false);
  const [autoLabelResult, setAutoLabelResult] = useState<any>(null);
  const [selectedModel, setSelectedModel] = useState('yolo11n.pt');
  const [confidence, setConfidence] = useState(0.25);
  const [, setShowAutoLabel] = useState(false);

  // 批量标注状态
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchTargetProject, setBatchTargetProject] = useState<string>('');
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });
  const [batchResults, setBatchResults] = useState<{ success: number; failed: number; total_detections: number } | null>(null);

  // 项目批量标注状态
  const [projectBatchTarget, setProjectBatchTarget] = useState<string | null>(null);
  const [projectBatchModel, setProjectBatchModel] = useState('yolo11n.pt');
  const [projectBatchConf, setProjectBatchConf] = useState(0.25);
  const [projectBatchLoading, setProjectBatchLoading] = useState(false);
  const [projectBatchProgress, setProjectBatchProgress] = useState({ current: 0, total: 0 });
  const [projectBatchDone, setProjectBatchDone] = useState<{ success: number; failed: number } | null>(null);

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

  // 初始化加载 SAM 模型（页面加载时自动加载）
  useEffect(() => {
    const initSAM = async () => {
      try {
        const res = await samApi.loadModel('vit_b');
        setSamLoaded(res.data?.success || res.data?.data?.success || false);
      } catch (e) {
        console.error('SAM init failed:', e);
      }
    };
    initSAM();
  }, []);

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
        const newMasks = (result.annotations || []).map((ann: SAMAnnotation) => ({
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

        if ((result.annotations || []).length === 0) {
          alert(`未检测到 "${samCurrentClass}" 类别的对象，共检测到 ${result.total_detections ?? 0} 个其他对象`);
        }
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
    const newMasks = samMasks.filter((_, i) => i !== idx);
    const newBoxes = samBoxes.filter((_, i) => i !== idx);
    setSamAnnotations(newAnnotations);
    setSamMasks(newMasks);
    setSamBoxes(newBoxes);
    saveSamHistory(samPoints, newBoxes, newMasks);
    if (samSelectedId === id) setSamSelectedId(null);
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
      const newMasks = samMasks.filter((_, i) => i !== idx);
      const newBoxes = samBoxes.filter((_, i) => i !== idx);
      setSamAnnotations(newAnnotations);
      setSamMasks(newMasks);
      setSamBoxes(newBoxes);
      saveSamHistory(samPoints, newBoxes, newMasks);
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

  // 点击已标注图片缩略图，加载到编辑区
  const handleThumbnailClick = async (img: { name: string; url: string }) => {
    if (!selectedProject) return;
    setSamLoading(true);
    try {
      // 从 URL 加载图片
      const resp = await fetch(img.url);
      const blob = await resp.blob();
      const file = new File([blob], img.name, { type: blob.type || 'image/jpeg' });
      const imageUrl = URL.createObjectURL(blob);

      // 清除旧状态
      setSamPoints([]);
      setSamBoxes([]);
      setSamMasks([]);
      setSamAnnotations([]);
      setSamSelectedId(null);
      setSamHistory([]);
      setSamHistoryIndex(-1);

      setSamImage(imageUrl);
      setSamImagePath(img.name);
      setSamFile(file);

      // 获取图片尺寸
      const imgEl = new Image();
      imgEl.onload = () => setImgSize({ width: imgEl.width, height: imgEl.height });
      imgEl.src = imageUrl;

      // 加载已有标注
      const projectId = selectedProject.id || selectedProject.name;
      const annRes = await annotationApi.getAnnotations(projectId, img.name);
      const annData = annRes.data?.annotations || annRes.data?.data?.annotations || [];

      console.log('加载标注:', { projectId, imgName: img.name, annData });

      if (annData.length > 0) {
        const newBoxes: AnnotationBox[] = [];
        const newAnnotations: SAMAnnotation[] = [];
        let newPoints: AnnotationPoint[] = [];
        const extraClasses: string[] = [];

        annData.forEach((ann: any) => {
          if (ann.bbox && ann.bbox.length === 4) {
            newBoxes.push({ x1: ann.bbox[0], y1: ann.bbox[1], x2: ann.bbox[2], y2: ann.bbox[3] });
          }
          if (ann.points && ann.points.length > 0) {
            newPoints = [...newPoints, ...ann.points.map((p: number[]) => ({ x: p[0], y: p[1], label: 1 }))];
          }
          const clsName = ann.class || ann.class_name || 'unknown';
          if (!samClasses.includes(clsName) && !extraClasses.includes(clsName)) {
            extraClasses.push(clsName);
          }
          const clsIdx = samClasses.indexOf(clsName);
          newAnnotations.push({
            class: clsName,
            class_id: clsIdx >= 0 ? clsIdx : (ann.class_id ?? 0),
            bbox: ann.bbox || [],
            segmentation: '',
            confidence: ann.confidence ?? 1,
          });
        });

        // 直接设置状态，无需历史记录
        setSamBoxes(newBoxes);
        setSamMasks([]);
        setSamAnnotations(newAnnotations);
        setSamPoints(newPoints);
        if (extraClasses.length > 0) {
          setSamClasses(prev => [...prev, ...extraClasses]);
        }
      }
    } catch (e) {
      console.error('加载图片失败:', e);
    } finally {
      setSamLoading(false);
    }
  };

  // 处理文件选择并自动标注
  const handleFileSelect = async (files: File[]) => {
    if (files.length === 0) return;

    if (files.length === 1) {
      // 单张：保持原有预览行为
      const file = files[0];
      const imageUrl = URL.createObjectURL(file);
      setSelectedImage(imageUrl);
      setAutoLabelResult(null);
      setShowAutoLabel(true);
      setBatchFiles([]);
      await handleAutoLabel(file);
    } else {
      // 多张：批量模式
      setSelectedImage(null);
      setAutoLabelResult(null);
      setBatchFiles(files);
      setBatchResults(null);
    }
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

  // 批量标注（上传多张图片）
  const handleBatchAutoLabel = async () => {
    if (batchFiles.length === 0 || !batchTargetProject) return;

    setBatchLoading(true);
    setBatchProgress({ current: 0, total: batchFiles.length });
    setBatchResults(null);

    let success = 0;
    let failed = 0;
    let total_detections = 0;

    // 分批上传图片，获取实际保存的文件名映射
    const uploadRes = await annotationApi.addImages(batchTargetProject, batchFiles);
    const nameMap: Record<string, string> =
      (uploadRes.data?.data?.name_map || uploadRes.data?.name_map || {}) as Record<string, string>;
    console.log('nameMap:', nameMap);

    for (let i = 0; i < batchFiles.length; i++) {
      const file = batchFiles[i];
      setBatchProgress({ current: i + 1, total: batchFiles.length });
      // 让出主线程，使进度 UI 得以更新
      await new Promise(resolve => setTimeout(resolve, 0));
      try {
        // 推理
        const res = await inferenceApi.image(file, selectedModel, confidence);
        const result = res.data?.data || res.data as any;
        const detections = result?.detections || [];

        // 用实际保存的文件名保存标注
        const savedName = nameMap[file.name] || file.name;
        if (detections.length > 0) {
          const annotations = detections.map((det: any) => ({
            class: det.class_name,
            class_id: det.class_id ?? 0,
            bbox: [det.x1, det.y1, det.x2, det.y2],
            confidence: det.confidence,
          }));
          await annotationApi.saveAnnotations(batchTargetProject, savedName, annotations);
          console.log('保存标注:', { savedName, count: detections.length });
        }

        total_detections += detections.length;
        success++;
      } catch (e) {
        console.error(`批量标注失败 [${file.name}]:`, e);
        failed++;
      }
    }

    setBatchResults({ success, failed, total_detections });
    setBatchLoading(false);

    // 重新加载项目图片，获取服务器上的实际文件名
    if (selectedProject) {
      const projectId = selectedProject.id || selectedProject.name;
      const res = await annotationApi.getImages(projectId);
      const data = res.data?.images || res.data?.data?.images || [];
      setProjectImages(data);
    }
  };

  // 对项目内已有图片批量标注
  const handleProjectBatchLabel = async (project: AnnotationProject) => {
    setProjectBatchLoading(true);
    setProjectBatchProgress({ current: 0, total: 0 });
    setProjectBatchDone(null);

    try {
      const projectId = project.id || project.name;
      const res = await annotationApi.getImages(projectId);
      const images: { name: string; url: string }[] = res.data?.images || res.data?.data?.images || [];

      setProjectBatchProgress({ current: 0, total: images.length });

      let success = 0;
      let failed = 0;

      for (let i = 0; i < images.length; i++) {
        const img = images[i];
        setProjectBatchProgress({ current: i + 1, total: images.length });
        // 让出主线程，使进度 UI 得以更新
        await new Promise(resolve => setTimeout(resolve, 0));
        try {
          // 通过 URL 获取图片 Blob
          const blob = await fetch(img.url).then(r => r.blob());
          const file = new File([blob], img.name, { type: blob.type || 'image/jpeg' });

          const inferRes = await inferenceApi.image(file, projectBatchModel, projectBatchConf);
          const result = inferRes.data?.data || inferRes.data as any;
          const detections = result?.detections || [];

          if (detections.length > 0) {
            const annotations = detections.map((det: any) => ({
              class: det.class_name,
              class_id: det.class_id ?? 0,
              bbox: [det.x1, det.y1, det.x2, det.y2],
              confidence: det.confidence,
            }));
            await annotationApi.saveAnnotations(projectId, img.name, annotations);
          }
          success++;
        } catch (e) {
          console.error(`项目批量标注失败 [${img.name}]:`, e);
          failed++;
        }
      }

      setProjectBatchDone({ success, failed });
    } catch (e) {
      console.error('项目批量标注异常:', e);
    } finally {
      setProjectBatchLoading(false);
    }
  };

  const pageStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    gap: 'var(--space-6)',
  };

  const sectionHeaderStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 'var(--space-5)',
  };

  const tabBtnStyle = (active: boolean): React.CSSProperties => ({
    padding: '8px 20px',
    border: active ? '2px solid var(--primary-500)' : '2px solid var(--border-color)',
    borderRadius: 'var(--radius-full)',
    background: active ? 'var(--primary-500)' : 'transparent',
    color: active ? '#fff' : 'var(--text-secondary)',
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: '13px',
    transition: 'all var(--transition-base)',
  });

  const projectCardStyle = (selected: boolean): React.CSSProperties => ({
    padding: '20px',
    borderRadius: 'var(--radius-lg)',
    border: selected ? '2px solid var(--primary-500)' : '1px solid var(--border-color)',
    background: selected ? 'var(--primary-50)' : 'var(--bg-primary)',
    cursor: 'pointer',
    transition: 'all var(--transition-base)',
    boxShadow: selected ? 'var(--shadow-md)' : 'var(--shadow-sm)',
  });

  return (
    <div style={pageStyle}>
      {/* 页面标题 */}
      <div style={sectionHeaderStyle}>
        <div>
          <h1 style={{ fontWeight: 700, fontSize: '1.75rem', marginBottom: '4px' }}>智能标注</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
            支持 YOLO 预标注、SAM 分割标注和手动标注
          </p>
        </div>
        <Button variant="primary" onClick={() => setShowCreate(!showCreate)}>
          {showCreate ? '取消' : '+ 创建项目'}
        </Button>
      </div>

      {/* 创建项目 */}
      {showCreate && (
        <Card style={{ border: '1px solid var(--primary-200)', background: 'var(--primary-50)' }}>
          <CardHeader icon="➕" title="创建标注项目" />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: 'var(--space-4)', alignItems: 'flex-end' }}>
            <Input
              label="项目名称"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="输入项目名称"
            />
            <Input
              label="项目描述（可选）"
              value={newProjectDesc}
              onChange={(e) => setNewProjectDesc(e.target.value)}
              placeholder="输入项目描述"
            />
            <Button variant="primary" onClick={handleCreateProject}>
              创建
            </Button>
          </div>
        </Card>
      )}

      {/* YOLO 智能预标注 */}
      <Card>
        <div style={sectionHeaderStyle}>
          <CardHeader icon="🤖" title="YOLO 智能预标注" />
          <span style={{
            fontSize: '12px', padding: '3px 10px',
            background: 'var(--success-light)', color: 'var(--success)',
            borderRadius: 'var(--radius-full)', fontWeight: 600
          }}>自动检测</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
          <div>
            <label style={{ display: 'block', marginBottom: 'var(--space-2)', fontWeight: 500, fontSize: '13px', color: 'var(--text-secondary)' }}>
              检测模型
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                width: '100%', padding: '8px 12px',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-color)',
                background: 'var(--bg-primary)',
                fontSize: '14px', outline: 'none',
              }}
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
            min={0} max={1} step={0.05}
          />
          <div>
            <label style={{ display: 'block', marginBottom: 'var(--space-2)', fontWeight: 500, fontSize: '13px', color: 'var(--text-secondary)' }}>
              保存到项目（批量时必选）
            </label>
            <select
              value={batchTargetProject}
              onChange={(e) => setBatchTargetProject(e.target.value)}
              style={{
                width: '100%', padding: '8px 12px',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-color)',
                background: 'var(--bg-primary)',
                fontSize: '14px', outline: 'none',
              }}
            >
              <option value="">— 不保存 —</option>
              {projects.map(p => (
                <option key={p.id} value={p.id || p.name}>{p.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label style={{ display: 'block', marginBottom: 'var(--space-2)', fontWeight: 500, fontSize: '13px', color: 'var(--text-secondary)' }}>
              上传图片（支持多选）
            </label>
            <label style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              gap: 'var(--space-2)', cursor: 'pointer',
              padding: '8px 12px', height: '38px',
              background: 'var(--primary-50)',
              border: '1.5px dashed var(--primary-300)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--primary-600)', fontWeight: 500, fontSize: '14px',
              transition: 'all var(--transition-base)',
            }}>
              📁 选择图片或拖拽
              <input type="file" accept="image/*" multiple
                onChange={(e) => handleFileSelect(e.target.files ? Array.from(e.target.files) : [])}
                style={{ display: 'none' }} />
            </label>
          </div>
        </div>

        {/* 批量模式：文件列表 + 启动按钮 */}
        {batchFiles.length > 1 && (
          <div style={{ marginBottom: 'var(--space-4)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '13px', fontWeight: 600 }}>已选择 {batchFiles.length} 张图片</span>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button onClick={() => { setBatchFiles([]); setBatchResults(null); }} style={{
                  padding: '5px 12px', border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-md)', background: 'var(--bg-primary)',
                  cursor: 'pointer', fontSize: '13px', color: 'var(--text-secondary)',
                }}>清空</button>
                <Button
                  variant="primary" size="sm"
                  onClick={handleBatchAutoLabel}
                  disabled={batchLoading || !batchTargetProject}
                >
                  {batchLoading ? `处理中 ${batchProgress.current}/${batchProgress.total}...` : '开始批量标注'}
                </Button>
              </div>
            </div>
            {!batchTargetProject && (
              <p style={{ fontSize: '12px', color: 'var(--warning)', marginBottom: '8px' }}>⚠️ 请先选择「保存到项目」</p>
            )}
            {batchLoading && (
              <div style={{ marginBottom: '8px' }}>
                <div style={{ height: '6px', background: 'var(--bg-tertiary)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', background: 'var(--primary-500)', borderRadius: '3px',
                    width: `${batchProgress.total ? (batchProgress.current / batchProgress.total) * 100 : 0}%`,
                    transition: 'width 0.3s ease',
                  }} />
                </div>
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                  正在处理 {batchProgress.current}/{batchProgress.total}：{batchFiles[batchProgress.current - 1]?.name}
                </p>
              </div>
            )}
            {batchResults && (
              <div style={{
                padding: '10px 14px', borderRadius: 'var(--radius-md)',
                background: 'var(--success-light)', border: '1px solid var(--success)',
                fontSize: '13px', display: 'flex', gap: '20px',
              }}>
                <span>✅ 成功 <strong>{batchResults.success}</strong> 张</span>
                {batchResults.failed > 0 && <span>❌ 失败 <strong>{batchResults.failed}</strong> 张</span>}
                <span>共检测 <strong>{batchResults.total_detections}</strong> 个对象</span>
              </div>
            )}
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', maxHeight: '80px', overflowY: 'auto', marginTop: '8px' }}>
              {batchFiles.map((f, i) => (
                <div key={i} style={{
                  padding: '3px 10px', borderRadius: 'var(--radius-full)',
                  background: 'var(--bg-secondary)', border: '1px solid var(--border-color)',
                  fontSize: '12px', color: 'var(--text-secondary)',
                }}>{f.name}</div>
              ))}
            </div>
          </div>
        )}

        {/* 单张预览模式 */}
        {selectedImage && batchFiles.length <= 1 && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-5)' }}>
            <div>
              <p style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 'var(--space-2)' }}>原图</p>
              <div style={{
                borderRadius: 'var(--radius-lg)', overflow: 'auto',
                border: '1px solid var(--border-color)', background: 'var(--gray-900)',
                maxHeight: '400px', display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
                padding: 'var(--space-4)'
              }}>
                <img
                  src={selectedImage}
                  alt="Original"
                  style={{
                    maxWidth: '100%',
                    height: 'auto',
                    objectFit: 'scale-down',
                    display: 'block'
                  }}
                />
              </div>
            </div>
            <div>
              <p style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 'var(--space-2)' }}>标注结果</p>
              <div style={{
                borderRadius: 'var(--radius-lg)', overflow: 'auto',
                border: '1px solid var(--border-color)', background: 'var(--gray-900)',
                maxHeight: '400px', display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
                padding: 'var(--space-4)'
              }}>
                {autoLabelLoading ? (
                  <div style={{ textAlign: 'center', color: '#fff', padding: 'var(--space-8)' }}>
                    <div style={{ fontSize: '2rem', marginBottom: '8px' }}>⏳</div>
                    <p style={{ fontSize: '14px' }}>正在智能标注中...</p>
                  </div>
                ) : autoLabelResult?.annotated_image ? (
                  <img
                    src={autoLabelResult.annotated_image}
                    alt="Annotated"
                    style={{
                      maxWidth: '100%',
                      height: 'auto',
                      objectFit: 'scale-down',
                      display: 'block'
                    }}
                  />
                ) : (
                  <div style={{ textAlign: 'center', color: 'var(--gray-400)', padding: 'var(--space-8)' }}>
                    <div style={{ fontSize: '2rem', marginBottom: '8px' }}>🎯</div>
                    <p style={{ fontSize: '14px' }}>等待检测结果</p>
                  </div>
                )}
              </div>
              {autoLabelResult?.detections && (
                <div style={{
                  marginTop: 'var(--space-3)', padding: 'var(--space-3)',
                  background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-color)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-2)', fontSize: '13px' }}>
                    <span style={{ fontWeight: 600 }}>检测到 <span style={{ color: 'var(--primary-500)' }}>{autoLabelResult.detections.length}</span> 个对象</span>
                    <span style={{ color: 'var(--text-secondary)' }}>⏱ {autoLabelResult.inference_time?.toFixed(2)}s</span>
                  </div>
                  <div style={{ maxHeight: '120px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    {Object.entries(
                      autoLabelResult.detections.reduce((acc: any, det: any) => {
                        acc[det.class_name] = (acc[det.class_name] || 0) + 1;
                        return acc;
                      }, {})
                    ).map(([className, count]: [string, any]) => (
                      <div key={className} style={{
                        display: 'flex', justifyContent: 'space-between',
                        padding: '4px 8px', borderRadius: 'var(--radius-sm)',
                        background: 'var(--bg-primary)', fontSize: '13px'
                      }}>
                        <span style={{ textTransform: 'capitalize' }}>{className}</span>
                        <span style={{
                          background: 'var(--primary-100)', color: 'var(--primary-700)',
                          padding: '1px 8px', borderRadius: 'var(--radius-full)', fontWeight: 600
                        }}>×{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </Card>

      {/* 项目列表 */}
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-5)' }}>
          <CardHeader icon="✏️" title="标注项目列表" />
          <button onClick={loadProjects} disabled={loading} style={{
            padding: '6px 14px', border: '1px solid var(--border-color)',
            borderRadius: 'var(--radius-md)', background: 'var(--bg-primary)',
            cursor: loading ? 'not-allowed' : 'pointer', fontSize: '13px',
            color: 'var(--text-secondary)', transition: 'all var(--transition-base)',
          }}>
            {loading ? '⏳' : '🔄 刷新'}
          </button>
        </div>
        {loading ? (
          <div style={{ textAlign: 'center', padding: 'var(--space-10)', color: 'var(--text-secondary)' }}>
            <div style={{ fontSize: '2rem', marginBottom: '8px' }}>⏳</div>
            <p>加载中...</p>
          </div>
        ) : projects.length === 0 ? (
          <div style={{
            textAlign: 'center', padding: 'var(--space-10)',
            border: '2px dashed var(--border-color)', borderRadius: 'var(--radius-lg)',
            background: 'var(--bg-secondary)'
          }}>
            <div style={{ fontSize: '3rem', marginBottom: '12px' }}>📂</div>
            <p style={{ fontWeight: 600, marginBottom: '6px' }}>暂无标注项目</p>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
              点击右上角「+ 创建项目」开始
            </p>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 'var(--space-4)' }}>
            {projects.map((project) => (
              <div key={project.id}>
                <div onClick={() => handleSelectProject(project)} style={projectCardStyle(selectedProject?.id === project.id)}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{
                        width: '36px', height: '36px', borderRadius: 'var(--radius-md)',
                        background: selectedProject?.id === project.id ? 'var(--primary-500)' : 'var(--bg-tertiary)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '18px'
                      }}>✏️</div>
                      <h3 style={{ fontWeight: 600, fontSize: '15px' }}>{project.name}</h3>
                    </div>
                    <button onClick={(e) => handleDeleteProject(project.id!, e)} disabled={deletingId === project.id} style={{
                      border: 'none', background: 'transparent',
                      cursor: deletingId === project.id ? 'not-allowed' : 'pointer',
                      padding: '4px', borderRadius: 'var(--radius-sm)',
                      color: 'var(--text-muted)', fontSize: '15px',
                    }}>
                      {deletingId === project.id ? '⏳' : '🗑️'}
                    </button>
                  </div>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '13px', marginBottom: '12px', minHeight: '20px' }}>
                    {project.description || '暂无描述'}
                  </p>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                      {new Date(project.created_at).toLocaleDateString()}
                      {project.classes && <span style={{ marginLeft: '8px' }}>· {project.classes.length} 类别</span>}
                    </div>
                    <div style={{ display: 'flex', gap: '6px' }}>
                      <Button variant="secondary" size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          setProjectBatchTarget(projectBatchTarget === (project.id || project.name) ? null : (project.id || project.name));
                          setProjectBatchDone(null);
                        }}>
                        批量标注
                      </Button>
                      <Button variant={selectedProject?.id === project.id ? "primary" : "secondary"} size="sm"
                        onClick={(e) => { e.stopPropagation(); handleSelectProject(project); }}>
                        {selectedProject?.id === project.id ? '✓ 已选中' : '开始标注'}
                      </Button>
                    </div>
                  </div>
                </div>

                {/* 项目批量标注面板 */}
                {projectBatchTarget === (project.id || project.name) && (
                  <div style={{
                    marginTop: '8px', padding: '14px 16px',
                    borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--primary-200)',
                    background: 'var(--primary-50)',
                  }} onClick={(e) => e.stopPropagation()}>
                    <p style={{ fontWeight: 600, fontSize: '13px', marginBottom: '10px' }}>对「{project.name}」内图片批量标注</p>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '10px' }}>
                      <div>
                        <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', color: 'var(--text-secondary)' }}>模型</label>
                        <select value={projectBatchModel} onChange={(e) => setProjectBatchModel(e.target.value)}
                          style={{ width: '100%', padding: '6px 10px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', background: 'var(--bg-primary)', fontSize: '13px', outline: 'none' }}>
                          <option value="yolo11n.pt">YOLO11n</option>
                          <option value="yolo11s.pt">YOLO11s</option>
                          <option value="yolo11m.pt">YOLO11m</option>
                          <option value="yolo26n.pt">YOLO26n</option>
                          <option value="yolo26m.pt">YOLO26m</option>
                        </select>
                      </div>
                      <div>
                        <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', color: 'var(--text-secondary)' }}>置信度</label>
                        <input type="number" value={projectBatchConf}
                          onChange={(e) => setProjectBatchConf(parseFloat(e.target.value))}
                          min={0} max={1} step={0.05}
                          style={{ width: '100%', padding: '6px 10px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', background: 'var(--bg-primary)', fontSize: '13px', outline: 'none' }} />
                      </div>
                    </div>
                    {projectBatchLoading && (
                      <div style={{ marginBottom: '8px' }}>
                        <div style={{ height: '6px', background: 'var(--bg-tertiary)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{
                            height: '100%', background: 'var(--primary-500)', borderRadius: '3px',
                            width: `${projectBatchProgress.total ? (projectBatchProgress.current / projectBatchProgress.total) * 100 : 0}%`,
                            transition: 'width 0.3s ease',
                          }} />
                        </div>
                        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                          处理中 {projectBatchProgress.current}/{projectBatchProgress.total}
                        </p>
                      </div>
                    )}
                    {projectBatchDone && (
                      <div style={{ marginBottom: '8px', fontSize: '13px', color: 'var(--success)', fontWeight: 500 }}>
                        ✅ 完成！成功 {projectBatchDone.success} 张{projectBatchDone.failed > 0 ? `，失败 ${projectBatchDone.failed} 张` : ''}
                      </div>
                    )}
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <Button variant="primary" size="sm"
                        onClick={() => handleProjectBatchLabel(project)}
                        disabled={projectBatchLoading}>
                        {projectBatchLoading ? '标注中...' : '开始'}
                      </Button>
                      <Button variant="secondary" size="sm"
                        onClick={() => { setProjectBatchTarget(null); setProjectBatchDone(null); }}>
                        关闭
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* SAM 分割标注 */}
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-5)' }}>
          <CardHeader icon="✂️" title="SAM 分割标注" />
          <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
            <button onClick={() => setActiveTab('yolo')} style={tabBtnStyle(activeTab === 'yolo')}>YOLO 预标注</button>
            <button onClick={() => setActiveTab('sam')} style={tabBtnStyle(activeTab === 'sam')}>SAM 分割</button>
          </div>
        </div>

        {activeTab === 'sam' && (
          <div>
            {selectedProject && (
              <div style={{
                padding: '12px 16px', borderRadius: 'var(--radius-md)', marginBottom: '16px',
                background: 'var(--info-light)', border: '1px solid #bfdbfe',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <span style={{ fontSize: '18px' }}>📂</span>
                  <div>
                    <span style={{ fontWeight: 600, color: 'var(--primary-700)' }}>{selectedProject.name}</span>
                    <span style={{ marginLeft: '12px', color: 'var(--text-secondary)', fontSize: '13px' }}>
                      {projectImages.length} 张已标注图片
                    </span>
                  </div>
                </div>
                <button onClick={() => { setSelectedProject(null); setSamImage(null); setSamImagePath(''); setSamFile(null); setSamAnnotations([]); }}
                  style={{ padding: '5px 12px', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)', background: 'var(--bg-primary)', cursor: 'pointer', fontSize: '13px' }}>
                  关闭项目
                </button>
              </div>
            )}

            {selectedProject && projectImages.length > 0 && (
              <div style={{ marginBottom: '16px' }}>
                <p style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '8px' }}>已标注图片</p>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', maxHeight: '400px', overflowY: 'auto' }}>
                  {projectImages.map((img, i) => (
                    <div key={i} title={img.name}
                      onClick={() => handleThumbnailClick(img)}
                      style={{
                        width: '72px', height: '72px', borderRadius: 'var(--radius-md)',
                        overflow: 'hidden', border: '2px solid var(--border-color)', cursor: 'pointer',
                        transition: 'border-color var(--transition-base)',
                      }}>
                      <img src={img.url} alt={img.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    </div>
                  ))}
                </div>
              </div>
            )}

            <AnnotationToolbar
              tool={samTool} currentClass={samCurrentClass} classes={samClasses}
              suggestions={classSuggestions} onToolChange={setSamTool}
              onClassChange={setSamCurrentClass} onAddClass={handleAddClass}
              onAutoLabel={handleSamAutoLabel} onClear={handleSamClear}
              onUndo={handleSamUndo} onRedo={handleSamRedo}
              onDeleteSelected={handleDeleteSelected} onSave={handleSave}
              loading={samLoading}
            />

            <div style={{ display: 'flex', gap: '20px', marginTop: '20px' }}>
              <div style={{ flex: 1 }}>
                {samImage ? (
                  <AnnotationCanvas
                    image={samImage} width={imgSize.width} height={imgSize.height}
                    points={samPoints} boxes={samBoxes} masks={samMasks}
                    annotations={samAnnotations} tool={samTool} selectedId={samSelectedId}
                    onPointAdd={handleSamPointAdd} onBoxAdd={handleSamBoxAdd}
                    onMaskAdd={(mask) => {
                      const newMasks = [...samMasks, mask];
                      setSamMasks(newMasks);
                      saveSamHistory(samPoints, samBoxes, newMasks);
                    }}
                    onMaskSelect={(idx) => setSamSelectedId(String(idx))}
                    onAnnotationSelect={setSamSelectedId}
                    currentClass={samCurrentClass}
                  />
                ) : (
                  <label style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                    width: '100%', height: '420px', cursor: 'pointer',
                    border: '2px dashed var(--border-color)', borderRadius: 'var(--radius-xl)',
                    background: 'var(--bg-secondary)', transition: 'all var(--transition-base)',
                  }}>
                    <div style={{ fontSize: '52px', marginBottom: '16px' }}>🖼️</div>
                    <p style={{ fontWeight: 600, marginBottom: '4px' }}>点击或拖拽上传图片</p>
                    <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>支持 JPG、PNG、WebP</p>
                    <input type="file" accept="image/*"
                      onChange={(e) => handleSamFileSelect(e.target.files ? Array.from(e.target.files) : [])}
                      style={{ display: 'none' }} />
                  </label>
                )}
              </div>
              <AnnotationPanel
                annotations={samAnnotations} selectedId={samSelectedId}
                onSelect={setSamSelectedId} onDelete={handleSamDelete}
                onClassChange={handleSamClassChange} onExport={handleSamExport}
                classes={samClasses}
              />
            </div>

            <div style={{
              marginTop: '20px', padding: '12px 16px',
              background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-color)',
              display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px',
              fontSize: '13px', color: 'var(--text-secondary)'
            }}>
              <span>🟢 点击 — 正样本点</span>
              <span>🔴 Shift+点击 — 负样本点</span>
              <span>⬜ 拖动 — 框选分割</span>
              <span>⚡ 自动标注 — 批量检测</span>
            </div>
          </div>
        )}
      </Card>

    </div>
  );
};
