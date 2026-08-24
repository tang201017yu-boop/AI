import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { Card, CardHeader, Button, Input } from '../../components/common';
import { annotationApi, inferenceApi, samApi, modelApi, datasetApi } from '../../services/api';
import type {
  AnnotationProject,
  SAMAnnotation,
  AnnotationTool,
  AnnotationPoint,
  AnnotationBox,
  AnnotationMask,
  ClassSuggestion,
  ArbitrationEvaluatePayload,
  AgentSuggestionItem,
} from '../../types';
import { useDisputeDetection } from '../../modules/annotation/hooks/useDisputeDetection';
import { DisputeAlert } from '../../modules/annotation/components/DisputeAlert';
import { AgentSuggestion } from '../../modules/annotation/components/AgentSuggestion';
import { buildYoloSingleImageExport, stemFromImageName, triggerDownload } from './exportSamYolo';
import { buildYoloSingleImageZipBlob, YoloZipExportError } from './exportSamYoloZip';
import { AnnotationCanvas, AnnotationToolbar, AnnotationPanel } from '../../components/Annotation';
import { useSearchParams } from 'react-router-dom';

/** 与推理页一致：训练/上传的权重，用于智能标注里选「自己的模型」 */
interface UserYoloModelOption {
  path: string;
  label: string;
}

/** 与 AnnotationToolbar 中 SAM 档位一致，供 loadModel 映射 */
type SamToolbarVersion = 'sam2_lite' | 'sam2_base' | 'sam2_large' | 'sam3';

const SMART_HOVER_IOU = 0.45;
const DETECT_MERGE_IOU = 0.45;
const DETECT_NMS_IOU = 0.5;

/** 智能标注类别下拉：trim 后去重、保序（避免一键检测里多框同类 + 闭包陈旧 includes 造成 person 连刷） */
function dedupeClassesPreserveOrder(classes: readonly string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const raw of classes) {
    const c = String(raw ?? '').trim();
    if (!c || seen.has(c)) continue;
    seen.add(c);
    out.push(c);
  }
  return out;
}

function bboxIoU(a: number[], b: number[]): number {
  if (!a || !b || a.length < 4 || b.length < 4) return 0;
  const [ax1, ay1, ax2, ay2] = a.map(Number);
  const [bx1, by1, bx2, by2] = b.map(Number);
  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
  const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
  const union = areaA + areaB - inter;
  return union > 1e-9 ? inter / union : 0;
}

/** 仲裁服务缺陷类型：由类别名粗映射（对齐专利：裂缝/渗水/脱落/空鼓），其余走通用物理规则 */
function defectTypeFromClassLabel(label: string): string {
  const c = (label || '').toLowerCase();
  if (c.includes('crack') || c.includes('裂缝') || c.includes('fissure')) return 'crack';
  if (c.includes('seepage') || c.includes('渗') || c.includes('leak')) return 'seepage';
  if (c.includes('spalling') || c.includes('脱落')) return 'spalling';
  if (c.includes('hollow') || c.includes('void') || c.includes('空鼓')) return 'hollow';
  return 'defect';
}

function metricsPayloadFromBboxTuple(b: [number, number, number, number]): Record<string, number> {
  const [x1, y1, x2, y2] = b;
  const w = Math.abs(x2 - x1);
  const h = Math.abs(y2 - y1);
  const longSide = Math.max(w, h);
  const shortSide = Math.min(w, h);
  return {
    area: w * h,
    width_length_ratio: longSide > 0 ? shortSide / longSide : 0,
    wetness_index: 0.5,
    connected_components: 1,
  };
}

/** 无模型候选时的弱代理框：与手绘框保持较高 IoU（≥T_iou 时进入专利所述仲裁入口） */
function syntheticAgentBboxFromHuman(h: [number, number, number, number]): [number, number, number, number] {
  const [x1, y1, x2, y2] = h;
  const w = x2 - x1;
  const hgt = y2 - y1;
  const dx = Math.max(4, Math.abs(w) * 0.04);
  const dy = Math.max(3, Math.abs(hgt) * 0.04);
  return [
    Math.round(x1 + dx * 0.2),
    Math.round(y1 + dy * 0.15),
    Math.round(x2 - dx * 0.15),
    Math.round(y2 - dy * 0.2),
  ];
}

function pickAgentDetectionForArbitration(
  humanBbox: [number, number, number, number],
  humanLabel: string,
  detections: SAMAnnotation[],
): SAMAnnotation | null {
  const hLabel = (humanLabel || '').trim().toLowerCase();
  let best: { ann: SAMAnnotation; iou: number } | null = null;
  for (const d of detections) {
    if (!d.bbox || d.bbox.length < 4) continue;
    const iou = bboxIoU(humanBbox, d.bbox);
    const dLabel = String(d.class ?? '').trim().toLowerCase();
    if (hLabel && dLabel && dLabel !== hLabel) continue;
    if (!best || iou > best.iou) best = { ann: d, iou };
  }
  if (best && best.iou >= 0.01) return best.ann;
  best = null;
  for (const d of detections) {
    if (!d.bbox || d.bbox.length < 4) continue;
    const iou = bboxIoU(humanBbox, d.bbox);
    if (iou >= 0.01 && (!best || iou > best.iou)) best = { ann: d, iou };
  }
  return best?.ann ?? null;
}

function bboxOverlapsExistingAnnotations(bbox: number[], annotations: SAMAnnotation[], iouThreshold: number): boolean {
  for (const ann of annotations) {
    if (!ann.bbox || ann.bbox.length < 4) continue;
    if (bboxIoU(bbox, ann.bbox) >= iouThreshold) return true;
  }
  return false;
}

/** 单次检测内框去重（高置信度优先保留） */
function nmsAnnotationsByBoxIou(annotations: SAMAnnotation[], iouThreshold: number): SAMAnnotation[] {
  const valid = annotations.filter((a) => a.bbox && a.bbox.length >= 4);
  const sorted = [...valid].sort((a, b) => (b.confidence ?? 1) - (a.confidence ?? 1));
  const kept: SAMAnnotation[] = [];
  for (const cand of sorted) {
    const bb = cand.bbox!;
    let ok = true;
    for (const k of kept) {
      if (bboxIoU(bb, k.bbox!) >= iouThreshold) {
        ok = false;
        break;
      }
    }
    if (ok) kept.push(cand);
  }
  return kept;
}

function smartHoverKey(cls: string, bbox: number[]): string {
  const [x1, y1, x2, y2] = bbox.map(Number);
  return `${cls}|${Math.round(x1)}|${Math.round(y1)}|${Math.round(x2)}|${Math.round(y2)}`;
}

/** 项目侧栏 / API 返回的相对路径在「中文文件名」等场景下需统一编码，fetch 时才能稳定命中静态原图。 */
function safeEncodePathnameSegments(pathname: string): string {
  if (!pathname || pathname === '/') return pathname;
  return pathname
    .split('/')
    .map((part) => {
      if (part === '') return '';
      try {
        return encodeURIComponent(decodeURIComponent(part));
      } catch {
        return encodeURIComponent(part);
      }
    })
    .join('/');
}

/** 将标注项目图片的 href 规范为可 fetch 的地址（同域相对路径会保留查询串）。 */
function resolveAnnotationImageFetchUrl(href: string): string {
  if (!href) return href;
  if (href.startsWith('http://') || href.startsWith('https://')) {
    const u = new URL(href);
    u.pathname = safeEncodePathnameSegments(u.pathname);
    return u.toString();
  }
  const q = href.indexOf('?');
  const path = q >= 0 ? href.slice(0, q) : href;
  const search = q >= 0 ? href.slice(q) : '';
  return safeEncodePathnameSegments(path) + search;
}

export const Annotation: React.FC = () => {
  const [searchParams] = useSearchParams();
  const compactWorkbenchMode = searchParams.get('compact') === 'workspace';
  const samFileInputRef = useRef<HTMLInputElement | null>(null);
  const projectUploadInputRef = useRef<HTMLInputElement | null>(null);
  const [projects, setProjects] = useState<AnnotationProject[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDesc, setNewProjectDesc] = useState('');
  const [linkedDataset, setLinkedDataset] = useState<string>('');
  const [linkedDatasetClasses, setLinkedDatasetClasses] = useState<string[]>([]);
  const [selectedProject, setSelectedProject] = useState<AnnotationProject | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [uploadingProjectId, setUploadingProjectId] = useState<string | null>(null);
  const [projectUploadTargetId, setProjectUploadTargetId] = useState<string | null>(null);
  const [projectImages, setProjectImages] = useState<
    { name: string; url: string; original_url?: string }[]
  >([]);
  const [, setLoadingImages] = useState(false);

  // 智能标注状态
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [autoLabelLoading, setAutoLabelLoading] = useState(false);
  const [autoLabelResult, setAutoLabelResult] = useState<any>(null);
  const [selectedModel, setSelectedModel] = useState('yolo11n.pt');
  const [userYoloModels, setUserYoloModels] = useState<UserYoloModelOption[]>([]);
  const [confidence, setConfidence] = useState(0.1);
  /** 框选后调用 YOLO 推断类别（对齐 Ultralytics Hub：画框即匹配检测类名） */
  const [autoClassifyOnBox, setAutoClassifyOnBox] = useState(true);
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
  const [projectBatchDone, setProjectBatchDone] = useState<{ success: number; failed: number; total_detections?: number } | null>(null);

  // ============ 生成数据集版本（Generate New Version）状态 ============
  const [showVersionModal, setShowVersionModal] = useState(false);
  const [versionLoading, setVersionLoading] = useState(false);
  const [versionForm, setVersionForm] = useState<{
    dataset_name: string;
    val_ratio: number;
    test_ratio: number;
    seed: number;
    overwrite: boolean;
    resize: boolean;
    resize_size: number;
    aug_enabled: boolean;
    aug_horizontal_flip: boolean;
    aug_vertical_flip: boolean;
    aug_rotate: boolean;
    aug_brightness_contrast: boolean;
    aug_num: number;
  }>({
    dataset_name: '',
    val_ratio: 0.2,
    test_ratio: 0,
    seed: 42,
    overwrite: false,
    resize: false,
    resize_size: 640,
    aug_enabled: false,
    aug_horizontal_flip: true,
    aug_vertical_flip: false,
    aug_rotate: false,
    aug_brightness_contrast: true,
    aug_num: 2,
  });
  const [generatedVersions, setGeneratedVersions] = useState<Array<Record<string, any>>>([]);

  // ============ SAM 标注功能状态 ============
  /** 与 Ultralytics Hub 一致：Draw=手绘优先布局；Smart=AI 辅助优先（功能相同） */
  const [annotationMode, setAnnotationMode] = useState<'draw' | 'smart'>('draw');
  const [samImage, setSamImage] = useState<string | null>(null);
  const [samImagePath, setSamImagePath] = useState<string>('');
  const [samFile, setSamFile] = useState<File | null>(null);
  const [samTool, setSamTool] = useState<AnnotationTool>('box');
  const [samCurrentClass, setSamCurrentClass] = useState('');
  const [samClasses, setSamClasses] = useState<string[]>([]);
  const [samPoints, setSamPoints] = useState<AnnotationPoint[]>([]);
  const [samBoxes, setSamBoxes] = useState<AnnotationBox[]>([]);
  const [samMasks, setSamMasks] = useState<AnnotationMask[]>([]);
  const [samAnnotations, setSamAnnotations] = useState<SAMAnnotation[]>([]);
  const [samSelectedId, setSamSelectedId] = useState<string | null>(null);
  const [samHoveredId, setSamHoveredId] = useState<string | null>(null);
  const [samPanelSelectedIds, setSamPanelSelectedIds] = useState<string[]>([]);
  const [smartCandidates, setSmartCandidates] = useState<SAMAnnotation[]>([]);
  /** 与 Smart 共用一次 detectAll：作为仲裁里「Agent 候选框」来源 */
  const [arbitrationModelDetections, setArbitrationModelDetections] = useState<SAMAnnotation[]>([]);
  const {
    alerts: arbitrationAlerts,
    evaluation: arbitrationEvaluation,
    loading: arbitrationScanLoading,
    scan: scanArbitrationForBox,
    clear: clearArbitrationScan,
  } = useDisputeDetection();
  const [samLoading, setSamLoading] = useState(false);
  const [samLoaded, setSamLoaded] = useState(false);
  const [samVersion, setSamVersion] = useState<SamToolbarVersion>('sam2_base');
  /** 撤销栈须同时包含 annotations，否则多边形只写了 masks+列表时 Ctrl+Z 会只回滚 masks，列表与画布不一致、表现为「清不掉」 */
  const [samHistory, setSamHistory] = useState<
    { points: AnnotationPoint[]; boxes: AnnotationBox[]; masks: AnnotationMask[]; annotations: SAMAnnotation[] }[]
  >([]);
  const [samHistoryIndex, setSamHistoryIndex] = useState(-1);
  const [imgSize, setImgSize] = useState({ width: 800, height: 600 });
  /** 全屏图预览（中栏画布上当前打开的原图） */
  const [imageFullscreenOpen, setImageFullscreenOpen] = useState(false);

  /** 工具栏色条：仅本图已标注类别（去重保序）+ 当前绘制类别（若尚未出现在图中则排在最前） */
  const samClassPalette = useMemo(() => {
    const used: string[] = [];
    const seen = new Set<string>();
    for (const ann of samAnnotations) {
      const c = (ann.class || '').trim();
      if (!c || seen.has(c)) continue;
      seen.add(c);
      used.push(c);
    }
    const cur = (samCurrentClass || '').trim();
    if (cur && !seen.has(cur)) return [cur, ...used];
    return used;
  }, [samAnnotations, samCurrentClass]);

  useEffect(() => {
    if (annotationMode === 'draw' && samTool === 'auto') {
      setSamTool('box');
    }
  }, [annotationMode, samTool]);

  useEffect(() => {
    if (!samImage) {
      setImageFullscreenOpen(false);
    }
  }, [samImage]);

  useEffect(() => {
    if (!imageFullscreenOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setImageFullscreenOpen(false);
    };
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener('keydown', onKey);
    };
  }, [imageFullscreenOpen]);

  // 类别建议（基于检测结果）
  const [classSuggestions, setClassSuggestions] = useState<ClassSuggestion[]>([]);
  const smartAddedRef = React.useRef<Set<string>>(new Set());
  const smartLastMoveAtRef = React.useRef<number>(0);

  // SAM 档位 → 后端 loadModel 参数（与 sam_service 中 vit_b / vit_l / vit_h 一致）
  const mapSamToolbarToLoadModel = useCallback((version: SamToolbarVersion): string => {
    switch (version) {
      case 'sam3':
        return 'vit_h';
      case 'sam2_large':
        return 'vit_l';
      case 'sam2_lite':
      case 'sam2_base':
      default:
        return 'vit_b';
    }
  }, []);

  // 初始化加载 SAM 模型（页面加载时自动检查并加载）
  useEffect(() => {
    const initSAM = async () => {
      try {
        // 先检查模型是否已加载
        const statusRes = await samApi.getStatus();
        const isLoaded = statusRes.data?.loaded || statusRes.data?.data?.loaded || false;
        if (isLoaded) {
          console.log('[SAM] 模型已加载');
          setSamLoaded(true);
          return;
        }
        // 未加载则自动加载（按当前选择的 SAM 版本）
        console.log('[SAM] 模型未加载，正在加载...');
        const res = await samApi.loadModel(mapSamToolbarToLoadModel(samVersion));
        setSamLoaded(res.data?.success || res.data?.data?.success || false);
      } catch (e) {
        console.error('SAM init failed:', e);
      }
    };
    initSAM();
  }, [mapSamToolbarToLoadModel, samVersion]);

  // 切换 SAM 版本
  const handleSamVersionChange = useCallback(async (version: SamToolbarVersion) => {
    if (version === samVersion) return;
    setSamVersion(version);
    setSamLoading(true);
    try {
      const modelType = mapSamToolbarToLoadModel(version);
      const res = await samApi.loadModel(modelType);
      const ok = res.data?.success || res.data?.data?.success || false;
      setSamLoaded(ok);
      if (!ok) {
        alert(`切换到 ${version} 失败，请检查后端模型环境`);
      }
    } catch (e) {
      console.error('SAM version switch failed:', e);
      alert(`切换到 ${version} 失败`);
    } finally {
      setSamLoading(false);
    }
  }, [mapSamToolbarToLoadModel, samVersion]);

  // 保存历史记录（annotations 与 masks/boxes 索引对齐）
  const saveSamHistory = useCallback((
    newPoints: AnnotationPoint[],
    newBoxes: AnnotationBox[],
    newMasks: AnnotationMask[],
    newAnnotations: SAMAnnotation[],
  ) => {
    const newHistory = samHistory.slice(0, samHistoryIndex + 1);
    newHistory.push({ points: newPoints, boxes: newBoxes, masks: newMasks, annotations: newAnnotations });
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
      setSamAnnotations(prev.annotations);
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
      setSamAnnotations(next.annotations);
      setSamHistoryIndex(samHistoryIndex + 1);
    }
  }, [samHistory, samHistoryIndex]);

  /** 手绘/多边形落盘前：必须有已选类别名 */
  const pickAnnotationClass = useCallback((): string | null => {
    const cls = (samCurrentClass || '').trim();
    if (cls) return cls;
    if (samClasses.length === 0) {
      alert('请先在工具栏类别旁点击「+」添加类别');
    } else {
      alert('请先在类别下拉框中选择要标注的类别');
    }
    return null;
  }, [samCurrentClass, samClasses]);

  // 添加点
  const handleSamPointAdd = useCallback((point: AnnotationPoint) => {
    const newPoints = [...samPoints, point];
    setSamPoints(newPoints);
    saveSamHistory(newPoints, samBoxes, samMasks, samAnnotations);
  }, [samPoints, samBoxes, samMasks, samAnnotations, saveSamHistory]);

  // 添加框（手动框选必须同步写入 samAnnotations，右侧列表才能改类别/保存）
  const handleSamBoxAdd = useCallback(async (box: AnnotationBox) => {
    const cls = pickAnnotationClass();
    if (!cls) return;

    const newBoxes = [...samBoxes, box];
    const cid = samClasses.indexOf(cls);
    const newAnn: SAMAnnotation = {
      class: cls,
      class_id: cid >= 0 ? cid : 0,
      bbox: [box.x1, box.y1, box.x2, box.y2],
      segmentation: '',
      confidence: 1,
    };
    const newAnnotations = [...samAnnotations, newAnn];
    const annIdx = newAnnotations.length - 1;
    setSamBoxes(newBoxes);
    setSamAnnotations(newAnnotations);
    setSamSelectedId(String(annIdx));

    const emptyMask = (): AnnotationMask => ({ polygons: [], color: getRandomColor() });
    let newMasks = samMasks;

    if (samLoaded && samImagePath) {
      setSamLoading(true);
      try {
        const res = await samApi.predictBox([box.x1, box.y1, box.x2, box.y2]);
        const result = res.data?.data || res.data;
        if (result?.success && result.masks?.length > 0) {
          newMasks = [...samMasks, { polygons: result.masks[0], color: getRandomColor() }];
        } else {
          newMasks = [...samMasks, emptyMask()];
        }
      } catch (e) {
        console.error('SAM predict failed:', e);
        newMasks = [...samMasks, emptyMask()];
      } finally {
        setSamLoading(false);
      }
    } else {
      newMasks = [...samMasks, emptyMask()];
    }

    setSamMasks(newMasks);
    saveSamHistory(samPoints, newBoxes, newMasks, newAnnotations);

    let resolvedLabel = cls;

    // 仅在「用户没有手动选类别」时才让 YOLO 推断；否则用户手动选的 person/xxx 视为最终意图，不被覆盖。
    // 即便启用自动识别也不再 setSamCurrentClass —— 避免下一次画框时全局当前类别被切走。
    const userPickedClass = String(samCurrentClass || '').trim();
    if (autoClassifyOnBox && !userPickedClass && samFile && annIdx >= 0) {
      try {
        const res = await samApi.classifyBbox(
          samFile,
          [box.x1, box.y1, box.x2, box.y2],
          selectedModel,
          confidence
        );
        const r = res.data?.data ?? res.data;
        if (r?.success && r.class_name) {
          const clsName = String(r.class_name);
          resolvedLabel = clsName;
          setSamClasses((prev) => {
            const next = prev.includes(clsName) ? prev : [...prev, clsName];
            const cid = next.indexOf(clsName);
            setSamAnnotations((aprev) => {
              const a = [...aprev];
              if (annIdx >= 0 && annIdx < a.length) {
                const confVal = typeof r.confidence === 'number' ? r.confidence : a[annIdx].confidence;
                a[annIdx] = {
                  ...a[annIdx],
                  class: clsName,
                  class_id: cid,
                  confidence: confVal,
                };
              }
              return a;
            });
            return next;
          });
        }
      } catch (e) {
        console.warn('[classify-bbox]', e);
      }
    }

    if (samFile) {
      const hb: [number, number, number, number] = [box.x1, box.y1, box.x2, box.y2];
      let dets = arbitrationModelDetections;
      if (dets.length === 0) {
        try {
          const detRes = await samApi.detectAll(samFile, selectedModel, confidence);
          const detData = detRes.data?.data || detRes.data;
          dets = detData?.annotations || [];
        } catch {
          dets = [];
        }
      }
      const agentAnn = pickAgentDetectionForArbitration(hb, resolvedLabel, dets);
      let agentBbox: [number, number, number, number];
      let agentLabel: string;
      if (agentAnn?.bbox && agentAnn.bbox.length >= 4) {
        agentBbox = [agentAnn.bbox[0], agentAnn.bbox[1], agentAnn.bbox[2], agentAnn.bbox[3]];
        agentLabel = String(agentAnn.class ?? resolvedLabel);
      } else {
        agentBbox = syntheticAgentBboxFromHuman(hb);
        agentLabel = resolvedLabel;
      }
      const arbPayload: ArbitrationEvaluatePayload = {
        annotation_id: `${samImagePath || samFile.name}:${annIdx}:${Date.now()}`,
        defect_type: defectTypeFromClassLabel(resolvedLabel),
        iou_threshold: 0.3,
        source: {
          project: selectedProject?.name,
          project_id: selectedProject?.id,
          image: samImagePath || samFile.name,
        },
        agent_annotation: {
          label: agentLabel,
          bbox: agentBbox,
          attributes: metricsPayloadFromBboxTuple(agentBbox),
        },
        human_annotation: {
          label: resolvedLabel,
          bbox: hb,
          attributes: metricsPayloadFromBboxTuple(hb),
        },
      };
      await scanArbitrationForBox(arbPayload);
    }
  }, [
    samBoxes,
    samMasks,
    samPoints,
    samLoaded,
    samImagePath,
    saveSamHistory,
    samAnnotations,
    samClasses,
    pickAnnotationClass,
    autoClassifyOnBox,
    samCurrentClass,
    samFile,
    selectedModel,
    confidence,
    arbitrationModelDetections,
    scanArbitrationForBox,
    selectedProject,
  ]);

  // 全图检测：Smart 悬停采纳 + 仲裁 Agent 候选（单次 detectAll）
  useEffect(() => {
    if (!samFile) {
      setSmartCandidates([]);
      setArbitrationModelDetections([]);
      smartAddedRef.current.clear();
      return;
    }
    let cancelled = false;
    const load = async () => {
      try {
        const res = await samApi.detectAll(samFile, selectedModel, confidence);
        const result = res.data?.data || res.data;
        const anns: SAMAnnotation[] = result?.annotations || [];
        if (cancelled) return;
        setArbitrationModelDetections(anns);
        if (annotationMode === 'smart') {
          smartAddedRef.current.clear();
          setSmartCandidates(anns);
        } else {
          smartAddedRef.current.clear();
          setSmartCandidates([]);
        }
      } catch {
        if (!cancelled) {
          setArbitrationModelDetections([]);
          setSmartCandidates([]);
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [annotationMode, samFile, selectedModel, confidence]);

  useEffect(() => {
    clearArbitrationScan();
  }, [samImagePath, clearArbitrationScan]);

  const arbitrationSuggestionItems = useMemo<AgentSuggestionItem[]>(() => {
    if (!arbitrationEvaluation) return [];
    const winner = arbitrationEvaluation.decisionWinner;
    const reason = `${arbitrationEvaluation.scenarioLabel}，${arbitrationEvaluation.decisionAction}`;
    return [
      {
        id: 'arbitration-winner',
        className:
          winner === 'agent'
            ? '采信 Agent（模型检测）'
            : winner === 'human'
              ? '采信人类（手绘框）'
              : winner === 'expert'
                ? '升级专家复核'
                : '双方均合法 · 可融合',
        confidence: Math.max(0.55, arbitrationEvaluation.iou),
        reason,
      },
    ];
  }, [arbitrationEvaluation]);

  const handleSmartCursorMove = useCallback((x: number, y: number) => {
    if (annotationMode !== 'smart') return;
    const now = Date.now();
    if (now - smartLastMoveAtRef.current < 80) return;
    smartLastMoveAtRef.current = now;

    for (const c of smartCandidates) {
      const b = c.bbox || [];
      if (b.length < 4) continue;
      const [x1, y1, x2, y2] = b;
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        if (bboxOverlapsExistingAnnotations(b, samAnnotations, SMART_HOVER_IOU)) return;
        const key = smartHoverKey(String(c.class || ''), b);
        if (smartAddedRef.current.has(key)) return;
        smartAddedRef.current.add(key);

        const clsName = String(c.class ?? '').trim();
        const displayClass = clsName || String(c.class ?? 'unknown');
        const existingIdx = samClasses.indexOf(displayClass);
        const resolvedClassId =
          existingIdx >= 0 ? existingIdx : clsName ? samClasses.length : (c.class_id ?? 0);
        const ann: SAMAnnotation = {
          class: displayClass,
          class_id: resolvedClassId,
          bbox: [x1, y1, x2, y2],
          segmentation: '',
          confidence: c.confidence ?? 1,
        };
        setSamAnnotations((prev) => {
          const next = [...prev, ann];
          setSamSelectedId(String(next.length - 1));
          return next;
        });
        setSamBoxes((prev) => [...prev, { x1, y1, x2, y2 }]);
        setSamMasks((prev) => [...prev, { polygons: [], color: getRandomColor() }]);
        if (clsName) {
          setSamClasses((prev) => (prev.includes(clsName) ? prev : [...prev, clsName]));
        }
        return;
      }
    }
  }, [annotationMode, smartCandidates, samClasses, samAnnotations]);

  // 清除
  const handleSamClear = useCallback(() => {
    smartAddedRef.current.clear();
    clearArbitrationScan();
    setSamPoints([]);
    setSamBoxes([]);
    setSamMasks([]);
    setSamAnnotations([]);
    saveSamHistory([], [], [], []);
  }, [saveSamHistory, clearArbitrationScan]);

  // 一键自动标注（检测所有类别）
  const handleDetectAll = useCallback(async () => {
    if (!samFile) {
      alert('请先选择图片');
      return;
    }
    setSamLoading(true);
    try {
      const res = await samApi.detectAll(samFile, selectedModel, confidence);
      const result = res.data?.data || res.data;

      if (result?.success) {
        const anns: SAMAnnotation[] = result.annotations || [];
        if (anns.length === 0) {
          alert('未检测到任何对象，请尝试降低置信度阈值');
          return;
        }

        const deduped = nmsAnnotationsByBoxIou(anns, DETECT_NMS_IOU);
        const fresh = deduped.filter(
          (a) => a.bbox && a.bbox.length >= 4
            && !bboxOverlapsExistingAnnotations(a.bbox, samAnnotations, DETECT_MERGE_IOU),
        );
        if (fresh.length === 0) {
          alert('检测结果与当前图上已有框高度重叠，未重复加入。可删除重叠框或调低置信度后再试。');
          return;
        }

        // 合并到现有标注（仅加入与已有标注不高度重叠的框）
        const merged = [...samAnnotations, ...fresh];
        setSamAnnotations(merged);

        // 生成边界框显示
        const newBoxes: AnnotationBox[] = fresh.map((a: SAMAnnotation) => ({
          x1: a.bbox[0], y1: a.bbox[1], x2: a.bbox[2], y2: a.bbox[3]
        }));
        const allBoxes = [...samBoxes, ...newBoxes];
        setSamBoxes(allBoxes);

        // 与 annotations 索引对齐（检测无分割掩码时用空 mask，避免删除时错位）
        const newMaskPlaceholders = fresh.map(() => ({ polygons: [] as number[], color: getRandomColor() }));
        const allMasks = [...samMasks, ...newMaskPlaceholders];
        setSamMasks(allMasks);

        // 自动添加新检测到的类别（单次函数式合并，避免同批多框同类重复）
        const detectedNames = dedupeClassesPreserveOrder(fresh.map((a) => String(a.class ?? '')));
        setSamClasses((prev) => dedupeClassesPreserveOrder([...prev, ...detectedNames]));

        // 更新类别建议
        const classCounts: Record<string, number> = {};
        merged.forEach((a: SAMAnnotation) => {
          classCounts[a.class] = (classCounts[a.class] || 0) + 1;
        });
        setClassSuggestions(Object.entries(classCounts).map(([name, count]) => ({
          name, count, color: getRandomColor(),
        })));

        saveSamHistory(samPoints, allBoxes, allMasks, merged);
        const skipped = anns.length - fresh.length;
        alert(
          skipped > 0
            ? `自动标注完成，新增 ${fresh.length} 个对象（已跳过与已有标注重叠或检测重复的 ${skipped} 个）`
            : `自动标注完成，新增 ${fresh.length} 个对象`,
        );
      } else {
        alert(result?.message || '自动标注失败');
      }
    } catch (e) {
      console.error('detectAll failed:', e);
      alert('自动标注失败，请重试');
    } finally {
      setSamLoading(false);
    }
  }, [samFile, samAnnotations, samBoxes, samMasks, samPoints, samClasses, selectedModel, confidence, saveSamHistory]);

  // 自动标注（指定类别）
  const handleSamAutoLabel = useCallback(async () => {
    if (!samFile) return;
    if (!(samCurrentClass || '').trim()) {
      alert('请先在类别下拉框中选择要自动标注的类别');
      return;
    }

    setSamLoading(true);
    try {
      // 使用批量同类标注 - 传入文件对象
      const res = await samApi.batchSamLabel(samFile, samCurrentClass, selectedModel, confidence);
      const result = res.data?.data || res.data;

      if (result?.success) {
        const nextAnns = result.annotations || [];
        setSamAnnotations(nextAnns);

        // 生成掩码显示
        const newMasks = nextAnns.map((ann: SAMAnnotation) => ({
          polygons: ann.segmentation?.split(' ').map(Number) || [],
          color: getRandomColor(),
        }));
        setSamMasks(newMasks);
        saveSamHistory(samPoints, samBoxes, newMasks, nextAnns);

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

        // 自动添加未存在的类别（函数式合并，避免闭包陈旧）
        const newNames = dedupeClassesPreserveOrder(Object.keys(classCounts));
        setSamClasses((prev) => dedupeClassesPreserveOrder([...prev, ...newNames]));

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
  }, [samFile, samCurrentClass, samPoints, samBoxes, samClasses, selectedModel, confidence, saveSamHistory]);

  // 处理 SAM 图片上传
  const handleSamFileSelect = async (files: File[]) => {
    if (files.length === 0) return;
    const file = files[0];
    const imageUrl = URL.createObjectURL(file);
    setSamImage(imageUrl);
    setSamImagePath(file.name);
    setSamFile(file);

    // 选中项目时，上传图片即加入项目（不必等保存标注）
    if (selectedProject) {
      try {
        const projectId = selectedProject.id || selectedProject.name;
        const uploadRes = await annotationApi.addImages(projectId, [file]);
        const ok = uploadRes.data?.success || uploadRes.data?.data?.success;
        if (!ok) {
          console.warn('图片加入项目失败:', uploadRes.data);
        } else {
          const listRes = await annotationApi.getImages(projectId);
          const data = listRes.data?.images || listRes.data?.data?.images || [];
          setProjectImages(data);
        }
      } catch (e) {
        console.error('加入项目失败:', e);
      }
    }

    // 获取图片尺寸
    const img = new Image();
    img.onload = () => {
      setImgSize({ width: img.naturalWidth || img.width, height: img.naturalHeight || img.height });
    };
    img.src = imageUrl;
  };

  const triggerSamFilePick = () => {
    if (samFileInputRef.current) {
      // 重置 value，允许重复选择同一张图片也触发 onChange
      samFileInputRef.current.value = '';
      samFileInputRef.current.click();
    }
  };

  // 删除标注
  const handleSamDelete = (id: string) => {
    const idx = parseInt(id);
    const removed = samAnnotations[idx];
    if (removed?.bbox && removed.bbox.length >= 4) {
      smartAddedRef.current.delete(smartHoverKey(String(removed.class || ''), removed.bbox));
    }
    const newAnnotations = samAnnotations.filter((_, i) => i !== idx);
    const newMasks = samMasks.filter((_, i) => i !== idx);
    const newBoxes = samBoxes.filter((_, i) => i !== idx);
    setSamAnnotations(newAnnotations);
    setSamMasks(newMasks);
    setSamBoxes(newBoxes);
    saveSamHistory(samPoints, newBoxes, newMasks, newAnnotations);
    if (samSelectedId === id) setSamSelectedId(null);
  };

  // 类别修改
  const handleSamClassChange = (id: string, newClass: string) => {
    const idx = parseInt(id);
    const cid = samClasses.indexOf(newClass);
    setSamAnnotations(samAnnotations.map((a, i) => i === idx ? { ...a, class: newClass, class_id: cid >= 0 ? cid : a.class_id } : a));
  };

  // 批量改类
  const handleSamBulkClassChange = (ids: string[], newClass: string) => {
    const idSet = new Set(ids);
    const cid = samClasses.indexOf(newClass);
    setSamAnnotations((prev) => prev.map((a, i) => (
      idSet.has(String(i))
        ? { ...a, class: newClass, class_id: cid >= 0 ? cid : a.class_id }
        : a
    )));
  };

  const handleSamBulkDelete = (ids: string[]) => {
    const idSet = new Set(ids);
    ids.forEach((sid) => {
      const i = parseInt(sid, 10);
      const ann = samAnnotations[i];
      if (ann?.bbox && ann.bbox.length >= 4) {
        smartAddedRef.current.delete(smartHoverKey(String(ann.class || ''), ann.bbox));
      }
    });
    const newAnnotations = samAnnotations.filter((_, i) => !idSet.has(String(i)));
    const newMasks = samMasks.filter((_, i) => !idSet.has(String(i)));
    const newBoxes = samBoxes.filter((_, i) => !idSet.has(String(i)));
    setSamAnnotations(newAnnotations);
    setSamMasks(newMasks);
    setSamBoxes(newBoxes);
    setSamSelectedId(null);
    setSamHoveredId(null);
    saveSamHistory(samPoints, newBoxes, newMasks, newAnnotations);
  };

  // 拖拽移动标注框（对齐 Ultralytics 标注器：选中后可直接拖动）
  const handleSamBoxDrag = useCallback((index: number, box: AnnotationBox) => {
    setSamBoxes((prev) => prev.map((b, i) => i === index ? box : b));
    setSamAnnotations((prev) => prev.map((a, i) => (
      i === index ? { ...a, bbox: [box.x1, box.y1, box.x2, box.y2] } : a
    )));
  }, []);

  // 随机颜色
  const getRandomColor = () => {
    const colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899'];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  /** 导出当前图为 YOLO 标签（txt）+ 仅本图出现的类别列表；txt 内 class_id 与 classes 行一一对应 */
  const handleSamExport = useCallback(() => {
    if (!samAnnotations.length) {
      alert('当前没有标注可导出');
      return;
    }
    const w = imgSize.width;
    const h = imgSize.height;
    if (!w || !h) {
      alert('无法获取图片尺寸，请先加载一张图片');
      return;
    }
    const { lines, classNames } = buildYoloSingleImageExport(samAnnotations, samClasses, w, h);
    if (!lines.length) {
      alert('没有可导出的行：每条标注需要「分割多边形」或「检测框」至少一种');
      return;
    }
    const stem = stemFromImageName(samFile?.name || samImagePath || 'labels');
    triggerDownload(`${stem}.txt`, lines.join('\n'));
    triggerDownload(`${stem}_classes.txt`, classNames.join('\n'));
    alert(
      `已下载：\n• ${stem}.txt（${lines.length} 行；class_id 已按本图类别从 0 连续编号）\n• ${stem}_classes.txt（仅 ${classNames.length} 个本图出现的类别，与 class_id 顺序一致）\n请将 txt 放入 labels/，图片放入 images/，主文件名一致。`
    );
  }, [samAnnotations, imgSize, samFile, samImagePath, samClasses]);

  /** 一键 ZIP：images/ + labels/ + classes.txt + README */
  const handleSamExportZip = useCallback(async () => {
    if (!samAnnotations.length) {
      alert('当前没有标注可导出');
      return;
    }
    const w = imgSize.width;
    const h = imgSize.height;
    if (!w || !h) {
      alert('无法获取图片尺寸，请先加载一张图片');
      return;
    }
    let imageBytes: ArrayBuffer;
    let imageBaseName: string;
    if (samFile) {
      imageBytes = await samFile.arrayBuffer();
      imageBaseName = samFile.name;
    } else if (samImage?.startsWith('blob:')) {
      try {
        const r = await fetch(samImage);
        imageBytes = await r.arrayBuffer();
        imageBaseName = samImagePath || 'image.jpg';
      } catch {
        alert('无法读取当前预览图，请用「打开图片」重新选择本地文件后再导出 ZIP');
        return;
      }
    } else {
      alert('一键 ZIP 需要本地图片文件。请使用「打开图片」选择文件，或从项目图集打开（会下载为本地副本）后再试。');
      return;
    }
    try {
      const blob = await buildYoloSingleImageZipBlob({
        annotations: samAnnotations,
        samClasses,
        imgW: w,
        imgH: h,
        imageBaseName,
        imageBytes,
      });
      const stem = stemFromImageName(imageBaseName);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${stem}_yolo_export.zip`;
      a.rel = 'noopener';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      alert(
        `已下载 ${stem}_yolo_export.zip：\n• images/ 内为原图\n• labels/ 内为 ${stem}.txt\n• classes.txt 为类别表\n• README.txt 为说明`,
      );
    } catch (e) {
      if (e instanceof YoloZipExportError) {
        alert(e.message);
        return;
      }
      console.error('zip export', e);
      alert('导出 ZIP 失败，请重试');
    }
  }, [samAnnotations, imgSize, samFile, samImage, samImagePath, samClasses]);

  /** 导出当前项目全部图片的标注为 NDJSON（依赖后端 labels/ 下已有 txt） */
  const handleProjectExportNdjson = useCallback(async () => {
    if (!selectedProject) {
      alert('请先选择项目');
      return;
    }
    const pid = String(selectedProject.id || selectedProject.name);
    try {
      const res = await annotationApi.exportNdjson(pid);
      const blob = res.data as unknown as Blob;
      if (!(blob instanceof Blob) || blob.size === 0) {
        alert('导出为空：请确认项目中已有图片，且 labels/ 目录下有对应 .txt');
        return;
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${pid.replace(/[^\w.\-]+/g, '_')}.ndjson`;
      a.rel = 'noopener';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error('export ndjson', e);
      alert('NDJSON 导出失败，请检查网络与后端');
    }
  }, [selectedProject]);

  /** 加载当前项目已有的数据集版本列表 */
  const refreshDatasetVersions = useCallback(async () => {
    if (!selectedProject) {
      setGeneratedVersions([]);
      return;
    }
    const pid = String(selectedProject.id || selectedProject.name);
    try {
      const res = await annotationApi.listVersions(pid);
      const data = (res.data as any)?.versions ?? [];
      setGeneratedVersions(Array.isArray(data) ? data : []);
    } catch (e) {
      console.warn('listVersions failed', e);
      setGeneratedVersions([]);
    }
  }, [selectedProject]);

  useEffect(() => {
    refreshDatasetVersions();
  }, [refreshDatasetVersions]);

  /** 打开「生成新版本」弹窗，自动建议下一个版本号作为数据集名 */
  const handleOpenVersionModal = useCallback(() => {
    if (!selectedProject) {
      alert('请先选择项目');
      return;
    }
    const baseName = String(selectedProject.id || selectedProject.name || 'dataset')
      .replace(/[^\w\u4e00-\u9fa5\-]+/g, '_')
      .replace(/^_+|_+$/g, '') || 'dataset';
    const nextIndex = (generatedVersions?.length || 0) + 1;
    setVersionForm((prev) => ({
      ...prev,
      dataset_name: `${baseName}_v${nextIndex}`,
    }));
    setShowVersionModal(true);
  }, [selectedProject, generatedVersions]);

  /** 提交生成数据集版本 */
  const handleGenerateVersion = useCallback(async () => {
    if (!selectedProject) return;
    const pid = String(selectedProject.id || selectedProject.name);
    if (versionForm.val_ratio + versionForm.test_ratio >= 1) {
      alert('val + test 比例之和必须小于 1');
      return;
    }
    setVersionLoading(true);
    try {
      const payload = {
        dataset_name: versionForm.dataset_name?.trim() || undefined,
        val_ratio: Number(versionForm.val_ratio) || 0,
        test_ratio: Number(versionForm.test_ratio) || 0,
        seed: Number(versionForm.seed) || 42,
        overwrite: versionForm.overwrite,
        preprocessing: {
          resize: versionForm.resize,
          size: Number(versionForm.resize_size) || 640,
        },
        augmentation: {
          enabled: versionForm.aug_enabled,
          horizontal_flip: versionForm.aug_horizontal_flip,
          vertical_flip: versionForm.aug_vertical_flip,
          rotate: versionForm.aug_rotate,
          brightness_contrast: versionForm.aug_brightness_contrast,
          num_augmented: Math.max(1, Number(versionForm.aug_num) || 2),
        },
      };
      const res = await annotationApi.generateVersion(pid, payload);
      const data = res.data as any;
      const v = data?.version || {};
      const counts = v.split_counts || {};
      alert(
        `数据集版本已生成 ✅\n\n` +
          `名称：${v.dataset_name}\n` +
          `路径：${v.dataset_path || data?.dataset_path}\n` +
          `切分：train ${counts.train || 0} / val ${counts.val || 0} / test ${counts.test || 0}\n` +
          (v.augmentation?.augmented_images
            ? `增强：${v.augmentation.augmented_images} 张\n`
            : '') +
          `\n现在可以前往「数据集」页面，或直接到「模型训练」选择该数据集开始训练。`,
      );
      setShowVersionModal(false);
      await refreshDatasetVersions();
    } catch (e: any) {
      console.error('generateVersion failed', e);
      const msg = e?.response?.data?.detail || e?.message || '生成失败';
      alert(`生成数据集版本失败：${msg}`);
    } finally {
      setVersionLoading(false);
    }
  }, [selectedProject, versionForm, refreshDatasetVersions]);

  // 添加新类别
  const handleAddClass = (className: string) => {
    const name = className.trim();
    if (!name) return;
    setSamClasses((prev) => (prev.includes(name) ? prev : [...prev, name]));
    setSamCurrentClass(name);
  };

  // 重命名类别：同步更新所有引用此类别的标注（class_id 因索引未变保持不变）
  const handleRenameClass = (oldName: string, newName: string) => {
    const o = oldName.trim();
    const n = newName.trim();
    if (!o || !n || o === n) return;
    setSamClasses((prev) => {
      if (!prev.includes(o)) return prev;
      if (prev.includes(n)) return prev;
      return prev.map((c) => (c === o ? n : c));
    });
    setSamAnnotations((prev) => prev.map((a) => (a.class === o ? { ...a, class: n } : a)));
    setClassSuggestions((prev) => prev.map((s) => (s.name === o ? { ...s, name: n } : s)));
    setSamCurrentClass((cur) => (cur === o ? n : cur));
  };

  // 删除类别：同步删除所有引用此类别的标注（标注框 + 掩膜 + 选框），其余标注按新顺序重算 class_id
  const handleRemoveClass = (className: string) => {
    const target = className.trim();
    if (!target) return;
    if (!samClasses.includes(target)) return;
    const usedCount = samAnnotations.filter((a) => a.class === target).length;
    const tip =
      usedCount > 0
        ? `类别「${target}」已被 ${usedCount} 个标注引用，删除后这些标注框也会一并删除。是否继续？`
        : `确认删除类别「${target}」？`;
    if (!window.confirm(tip)) return;

    const nextClasses = samClasses.filter((c) => c !== target);
    const keepIdx: number[] = [];
    samAnnotations.forEach((a, i) => {
      if (a.class === target) {
        if (a.bbox && a.bbox.length >= 4) {
          smartAddedRef.current.delete(smartHoverKey(String(a.class || ''), a.bbox));
        }
      } else {
        keepIdx.push(i);
      }
    });

    const newAnnotations = keepIdx.map((i) => {
      const a = samAnnotations[i];
      const idx = nextClasses.indexOf(a.class);
      return idx === a.class_id ? a : { ...a, class_id: idx };
    });
    const newMasks = keepIdx.map((i) => samMasks[i]);
    const newBoxes = keepIdx.map((i) => samBoxes[i]);

    setSamClasses(nextClasses);
    setSamAnnotations(newAnnotations);
    setSamMasks(newMasks);
    setSamBoxes(newBoxes);
    setClassSuggestions((prev) => prev.filter((s) => s.name !== target));
    saveSamHistory(samPoints, newBoxes, newMasks, newAnnotations);

    setSamCurrentClass((cur) => (cur === target ? '' : cur));
    setSamSelectedId((cur) => {
      if (cur == null) return cur;
      const oldIdx = parseInt(cur, 10);
      if (Number.isNaN(oldIdx)) return cur;
      const newIdx = keepIdx.indexOf(oldIdx);
      return newIdx >= 0 ? String(newIdx) : null;
    });
  };

  // 删除选中的标注
  const handleDeleteSelected = () => {
    if (samSelectedId) {
      const idx = parseInt(samSelectedId);
      const removed = samAnnotations[idx];
      if (removed?.bbox && removed.bbox.length >= 4) {
        smartAddedRef.current.delete(smartHoverKey(String(removed.class || ''), removed.bbox));
      }
      const newAnnotations = samAnnotations.filter((_, i) => i !== idx);
      const newMasks = samMasks.filter((_, i) => i !== idx);
      const newBoxes = samBoxes.filter((_, i) => i !== idx);
      setSamAnnotations(newAnnotations);
      setSamMasks(newMasks);
      setSamBoxes(newBoxes);
      saveSamHistory(samPoints, newBoxes, newMasks, newAnnotations);
      setSamSelectedId(null);
    }
  };

  /** 工具栏 🗑️：有选中则删该条标注；否则移除当前图片（含 blob URL 释放） */
  const handleToolbarTrash = useCallback(() => {
    if (samSelectedId) {
      const idx = parseInt(samSelectedId, 10);
      if (!Number.isNaN(idx)) {
        const removed = samAnnotations[idx];
        if (removed?.bbox && removed.bbox.length >= 4) {
          smartAddedRef.current.delete(smartHoverKey(String(removed.class || ''), removed.bbox));
        }
        const newAnnotations = samAnnotations.filter((_, i) => i !== idx);
        const newMasks = samMasks.filter((_, i) => i !== idx);
        const newBoxes = samBoxes.filter((_, i) => i !== idx);
        setSamAnnotations(newAnnotations);
        setSamMasks(newMasks);
        setSamBoxes(newBoxes);
        saveSamHistory(samPoints, newBoxes, newMasks, newAnnotations);
      }
      setSamSelectedId(null);
      return;
    }
    if (!samImage && !samFile) return;
    if (!window.confirm('确定移除当前图片？未保存的标注将丢失。')) return;
    smartAddedRef.current.clear();
    if (samImage?.startsWith('blob:')) {
      URL.revokeObjectURL(samImage);
    }
    setSamImage(null);
    setSamImagePath('');
    setSamFile(null);
    setSamPoints([]);
    setSamBoxes([]);
    setSamMasks([]);
    setSamAnnotations([]);
    setSamSelectedId(null);
    setSamHistory([]);
    setSamHistoryIndex(-1);
    saveSamHistory([], [], [], []);
  }, [samSelectedId, samImage, samFile, samAnnotations, samMasks, samBoxes, samPoints, saveSamHistory]);

  /** 保存按钮置灰原因（后端按 projectId 写库，无项目则无法落盘） */
  const samSaveHint = useMemo(() => {
    if (!selectedProject) {
      return '标注与图片保存在服务器的「项目」下。请先在下方「标注项目」里选一个项目并点「开始标注」，再保存。';
    }
    if (!samFile) return '请先点击「打开图片」';
    if (!samAnnotations.length) return '当前没有可写入的标注';
    return '';
  }, [selectedProject, samFile, samAnnotations.length]);
  const samSaveDisabled = Boolean(samSaveHint);

  // 保存标注
  const handleSave = async () => {
    if (!selectedProject) {
      alert(
        '无法保存：后端接口需要「项目 ID」，把每张图的标注 JSON 与图片放在该项目的目录里。\n\n' +
          '请先在下方「标注项目」列表中点击某个项目的「开始标注」，进入工作台后再点「保存」。',
      );
      return;
    }

    if (!samFile || !samAnnotations.length) {
      alert('没有可保存的标注（需已打开图片且画布上有标注）');
      return;
    }

    try {
      const projectId = selectedProject.id || selectedProject.name;
      const alreadyInProject = projectImages.some((img) => img.name === samFile.name);

      // 1. 若当前图片尚未在项目中，先上传图片到项目
      if (!alreadyInProject) {
        const uploadRes = await annotationApi.addImages(projectId, [samFile]);
        console.log('图片上传结果:', uploadRes);
        if (!uploadRes.data?.success && !uploadRes.data?.data?.success) {
          alert('图片上传失败: ' + (uploadRes.data?.message || '未知错误'));
          return;
        }
      }

      // 2. 保存标注
      const imageName = samFile.name;
      // 关键：必须把图片真实尺寸带给后端，否则后端按默认 640×640 归一化，
      // 切换图片再回来反归一化会按真实尺寸还原 → bbox 位置漂移
      const sizeForSave: [number, number] = [
        imgSize.width || 640,
        imgSize.height || 640,
      ];
      const annotationsToSave = samAnnotations.map(ann => ({
        class: ann.class,
        class_id: ann.class_id,
        bbox: ann.bbox,
        segmentation: ann.segmentation,
        confidence: ann.confidence,
        image_size: sizeForSave,
      }));

      const saveRes = await annotationApi.saveAnnotations(
        projectId,
        imageName,
        annotationsToSave
      );

      console.log('标注保存结果:', saveRes);

      if (saveRes.data?.success || saveRes.data?.data?.success) {
        alert(`成功保存 ${samAnnotations.length} 个标注到项目 "${selectedProject.name}"`);
        // 刷新项目图片列表
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

      // 画布标注快捷键（Draw / Smart 模式通用）
      // Ctrl+Z: 撤销
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        handleSamUndo();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
        e.preventDefault();
        handleSamRedo();
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        // 多选优先：Delete 一次删所有选中项
        if (samPanelSelectedIds.length > 1) {
          e.preventDefault();
          handleSamBulkDelete(samPanelSelectedIds);
        } else if (samSelectedId) {
          e.preventDefault();
          handleDeleteSelected();
        }
      } else if (e.key === 'p' || e.key === 'P') {
        e.preventDefault();
        setSamTool('point');
      } else if (e.key === 'b' || e.key === 'B') {
        e.preventDefault();
        setSamTool('box');
      } else if (e.key === 's' || e.key === 'S') {
        if (!e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          setSamTool('select');
        }
      } else if (e.key === 'a' || e.key === 'A') {
        if (!e.ctrlKey && !e.metaKey && annotationMode === 'smart') {
          e.preventDefault();
          handleSamAutoLabel();
        }
      } else if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        setSamTool('polygon');
      } else if (e.key >= '1' && e.key <= '9') {
        const idx = parseInt(e.key, 10) - 1;
        if (idx < samClassPalette.length) {
          e.preventDefault();
          const nextClass = samClassPalette[idx];
          const cid = samClasses.indexOf(nextClass);
          setSamCurrentClass(nextClass);

          // 有选中框时，数字键直接改类（对齐主流标注器：数字=赋类）
          if (samSelectedId !== null) {
            const selIdx = parseInt(samSelectedId, 10);
            if (!Number.isNaN(selIdx)) {
              setSamAnnotations((prev) => prev.map((a, i) => (
                i === selIdx
                  ? {
                      ...a,
                      class: nextClass,
                      class_id: cid >= 0 ? cid : idx,
                    }
                  : a
              )));
            }
          }
        }
      } else if (e.key === 'Enter') {
        // 回车：确认当前项并切到下一项（连续标注节奏）
        if (samAnnotations.length > 0) {
          e.preventDefault();
          setSamSelectedId((prev) => {
            const cur = prev ? parseInt(prev, 10) : -1;
            const next = Number.isNaN(cur) ? 0 : (cur + 1) % samAnnotations.length;
            return String(next);
          });
        }
      } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
        e.preventDefault();
        setSamSelectedId((prev) => {
          const total = samAnnotations.length;
          if (total === 0) return null;
          const cur = prev ? parseInt(prev, 10) : 0;
          const next = Number.isNaN(cur) ? 0 : (cur - 1 + total) % total;
          return String(next);
        });
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
        e.preventDefault();
        setSamSelectedId((prev) => {
          const total = samAnnotations.length;
          if (total === 0) return null;
          const cur = prev ? parseInt(prev, 10) : -1;
          const next = Number.isNaN(cur) ? 0 : (cur + 1) % total;
          return String(next);
        });
      } else if (e.key === 'd' || e.key === 'D') {
        if (!e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          setAnnotationMode('draw');
        }
      } else if (e.key === 'm' || e.key === 'M') {
        if (!e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          setAnnotationMode('smart');
        }
      } else if (e.key === 'Escape') {
        setSamSelectedId(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [annotationMode, samSelectedId, samClasses, samClassPalette, handleSamUndo, handleSamRedo, handleDeleteSelected, handleSamAutoLabel, samAnnotations.length, samPanelSelectedIds]);

  useEffect(() => {
    loadProjects();
    loadUserYoloModels();
  }, []);

  useEffect(() => {
    const datasetName = searchParams.get('dataset') || '';
    const imageUrlParam = searchParams.get('image_url') || '';
    const imageNameParam = searchParams.get('image') || '';

    setLinkedDataset(datasetName);
    if (!datasetName) {
      setLinkedDatasetClasses([]);
    } else {
      (async () => {
        try {
          const res = await datasetApi.get(datasetName);
          const body: any = res.data;
          const ds = body?.dataset || body?.data?.dataset || body?.data || {};
          const classes = ds?.classes || [];
          setLinkedDatasetClasses(Array.isArray(classes) ? classes : []);
          if (!newProjectName) {
            setNewProjectName(`${datasetName}_标注`);
          }
        } catch (e) {
          console.warn('加载联动数据集失败:', e);
          setLinkedDatasetClasses([]);
        }
      })();
    }

    if (!imageUrlParam) {
      return;
    }

    (async () => {
      setSamLoading(true);
      try {
        const resp = await fetch(imageUrlParam);
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}`);
        }
        const blob = await resp.blob();
        const fallbackName = imageUrlParam.split('/').pop() || 'dataset_image.jpg';
        const imageName = decodeURIComponent(imageNameParam || fallbackName);
        const file = new File([blob], imageName, { type: blob.type || 'image/jpeg' });
        const imageObjectUrl = URL.createObjectURL(blob);

        setSamPoints([]);
        setSamBoxes([]);
        setSamMasks([]);
        setSamAnnotations([]);
        setSamSelectedId(null);
        setSamHistory([]);
        setSamHistoryIndex(-1);

        setSelectedImage(null);
        setBatchFiles([]);
        setAutoLabelResult(null);

        setSamImage(imageObjectUrl);
        setSamImagePath(imageName);
        setSamFile(file);
        setAnnotationMode('draw');

        const imgEl = new Image();
        imgEl.onload = () => setImgSize({ width: imgEl.naturalWidth || imgEl.width, height: imgEl.naturalHeight || imgEl.height });
        imgEl.src = imageObjectUrl;
      } catch (e) {
        console.warn('加载联动图片失败:', e);
      } finally {
        setSamLoading(false);
      }
    })();
  }, [searchParams]);

  const loadUserYoloModels = async () => {
    try {
      const res = await modelApi.getUserModels();
      const body = res.data as {
        models?: Array<{ path?: string; file_path?: string; name?: string; project?: string; source?: string }>;
        data?: { models?: Array<{ path?: string; file_path?: string; name?: string; project?: string; source?: string }> };
      };
      const allModels = body?.models ?? body?.data?.models ?? [];
      const userModels = allModels.filter(
        (m) => m.source === 'uploaded' || m.source === 'training' || m.source === 'project_model',
      );
      const options: UserYoloModelOption[] = userModels
        .map((m) => {
          const path = m.path || m.file_path || '';
          const base = path ? path.split(/[/\\]/).pop() || path : '';
          const short = m.name?.includes('best') ? '最优权重' : m.name?.includes('last') ? '最终权重' : (m.name || base);
          const label = m.project ? `[${m.project}] ${short}` : (short || path);
          return { path, label: label || path };
        })
        .filter((o) => o.path);
      setUserYoloModels(options);
    } catch (e) {
      console.error('加载用户训练模型失败:', e);
    }
  };

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
      const classesFromInput = newProjectDesc
        .split(',')
        .map(x => x.trim())
        .filter(Boolean);
      const finalClasses = classesFromInput.length > 0 ? classesFromInput : linkedDatasetClasses;
      await annotationApi.createProject({
        name: newProjectName,
        description: newProjectDesc,
        classes: finalClasses,
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

  // 项目上传图片
  const triggerProjectUpload = (projectId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setProjectUploadTargetId(projectId);
    if (projectUploadInputRef.current) {
      projectUploadInputRef.current.value = '';
      projectUploadInputRef.current.click();
    }
  };

  const handleProjectUploadSelect = async (files: FileList | null) => {
    if (!files || files.length === 0 || !projectUploadTargetId) return;
    const list = Array.from(files);
    setUploadingProjectId(projectUploadTargetId);
    try {
      const uploadRes = await annotationApi.addImages(projectUploadTargetId, list);
      const ok = uploadRes.data?.success || uploadRes.data?.data?.success;
      if (!ok) {
        alert('上传图片失败: ' + (uploadRes.data?.message || '未知错误'));
        return;
      }
      // 当前选中项目时，刷新右侧/工作台图片列表
      if ((selectedProject?.id || selectedProject?.name) === projectUploadTargetId) {
        const res = await annotationApi.getImages(projectUploadTargetId);
        const data = res.data?.images || res.data?.data?.images || [];
        setProjectImages(data);
      }
      alert(`上传成功：${list.length} 张图片`);
    } catch (error) {
      console.error('上传项目图片失败:', error);
      alert('上传项目图片失败，请重试');
    } finally {
      setUploadingProjectId(null);
      setProjectUploadTargetId(null);
    }
  };

  // 选择项目
  const handleSelectProject = async (project: AnnotationProject) => {
    setSelectedProject(project);
    setAnnotationMode('draw');

    // 如果项目有预定义类别，加载它们
    if (project.classes && project.classes.length > 0) {
      const uniq = dedupeClassesPreserveOrder(project.classes);
      setSamClasses(uniq);
      setSamCurrentClass(uniq[0] || '');
    } else {
      setSamClasses([]);
      setSamCurrentClass('');
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

  // 点击项目图集缩略图：在画布中加载该图的完整原图（不沿用过期的缓存响应）
  const handleThumbnailClick = async (img: { name: string; url: string; original_url?: string }) => {
    if (!selectedProject) return;
    setSamLoading(true);
    try {
      if (samImage?.startsWith('blob:')) {
        URL.revokeObjectURL(samImage);
      }
      const sourceHref = img.original_url || img.url;
      const fetchUrl = resolveAnnotationImageFetchUrl(sourceHref);
      const resp = await fetch(fetchUrl, { cache: 'no-store', credentials: 'same-origin' });
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status} ${resp.statusText}`);
      }
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
      imgEl.onload = () => setImgSize({ width: imgEl.naturalWidth || imgEl.width, height: imgEl.naturalHeight || imgEl.height });
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
          setSamClasses((prev) => dedupeClassesPreserveOrder([...prev, ...extraClasses]));
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

    /** 量出原图像素尺寸，用于把 bbox 正确归一化（后端无 image_size 默认按 640） */
    const measureFileSize = (file: File): Promise<[number, number]> =>
      new Promise((resolve) => {
        const url = URL.createObjectURL(file);
        const im = new Image();
        im.onload = () => {
          const w = im.naturalWidth || im.width || 640;
          const h = im.naturalHeight || im.height || 640;
          URL.revokeObjectURL(url);
          resolve([w, h]);
        };
        im.onerror = () => {
          URL.revokeObjectURL(url);
          resolve([640, 640]);
        };
        im.src = url;
      });

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
          const [imgW, imgH] = await measureFileSize(file);
          // 后端 schema：DetectionResult.bbox = [x1,y1,x2,y2]（不是 det.x1/x2/y1/y2）
          const annotations = detections
            .map((det: any) => {
              const raw = Array.isArray(det.bbox) ? det.bbox : null;
              const x1 = raw ? Number(raw[0]) : Number(det.x1);
              const y1 = raw ? Number(raw[1]) : Number(det.y1);
              const x2 = raw ? Number(raw[2]) : Number(det.x2);
              const y2 = raw ? Number(raw[3]) : Number(det.y2);
              if (![x1, y1, x2, y2].every((v) => Number.isFinite(v))) return null;
              return {
                class: det.class_name ?? det.class ?? 'object',
                class_id: det.class_id ?? 0,
                bbox: [x1, y1, x2, y2],
                confidence: typeof det.confidence === 'number' ? det.confidence : undefined,
                image_size: [imgW, imgH],
              };
            })
            .filter(Boolean);
          if (annotations.length > 0) {
            await annotationApi.saveAnnotations(batchTargetProject, savedName, annotations);
            console.log('保存标注:', { savedName, count: annotations.length });
          }
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

    /** 拿到图片真实像素尺寸，用于把推理 bbox 正确归一化（后端无 image_size 默认按 640 计算会压扁） */
    const measureImageSize = (objectUrl: string): Promise<[number, number]> =>
      new Promise((resolve) => {
        const im = new Image();
        im.onload = () => resolve([im.naturalWidth || im.width || 640, im.naturalHeight || im.height || 640]);
        im.onerror = () => resolve([640, 640]);
        im.src = objectUrl;
      });

    try {
      const projectId = project.id || project.name;
      const res = await annotationApi.getImages(projectId);
      const images: { name: string; url: string }[] = res.data?.images || res.data?.data?.images || [];

      setProjectBatchProgress({ current: 0, total: images.length });

      let success = 0;
      let failed = 0;
      let totalBoxes = 0;

      for (let i = 0; i < images.length; i++) {
        const img = images[i];
        setProjectBatchProgress({ current: i + 1, total: images.length });
        await new Promise(resolve => setTimeout(resolve, 0));
        let blobUrl: string | null = null;
        try {
          // 通过 URL 获取图片 Blob，并量出原图尺寸（推理在原图坐标系下返回 bbox）
          const fetchUrl = resolveAnnotationImageFetchUrl(img.url);
          const blob = await fetch(fetchUrl, { cache: 'no-store' }).then(r => r.blob());
          const file = new File([blob], img.name, { type: blob.type || 'image/jpeg' });
          blobUrl = URL.createObjectURL(blob);
          const [imgW, imgH] = await measureImageSize(blobUrl);

          const inferRes = await inferenceApi.image(file, projectBatchModel, projectBatchConf);
          const result = (inferRes.data as any)?.data ?? (inferRes.data as any);
          const detections: any[] = result?.detections || [];

          if (detections.length > 0) {
            // 后端 schema：DetectionResult.bbox = [x1,y1,x2,y2]（不是 det.x1/x2/y1/y2）
            const annotations = detections
              .map((det: any) => {
                const raw = Array.isArray(det.bbox) ? det.bbox : null;
                const x1 = raw ? Number(raw[0]) : Number(det.x1);
                const y1 = raw ? Number(raw[1]) : Number(det.y1);
                const x2 = raw ? Number(raw[2]) : Number(det.x2);
                const y2 = raw ? Number(raw[3]) : Number(det.y2);
                if (![x1, y1, x2, y2].every((v) => Number.isFinite(v))) return null;
                return {
                  class: det.class_name ?? det.class ?? 'object',
                  class_id: det.class_id ?? 0,
                  bbox: [x1, y1, x2, y2],
                  confidence: typeof det.confidence === 'number' ? det.confidence : undefined,
                  // 关键：把原图尺寸一起带上，后端会按真实尺寸归一化
                  image_size: [imgW, imgH],
                };
              })
              .filter(Boolean);

            if (annotations.length > 0) {
              await annotationApi.saveAnnotations(projectId, img.name, annotations);
              totalBoxes += annotations.length;
            }
          }
          success++;
        } catch (e) {
          console.error(`项目批量标注失败 [${img.name}]:`, e);
          failed++;
        } finally {
          if (blobUrl) URL.revokeObjectURL(blobUrl);
        }
      }

      console.log('[batch label] done', { success, failed, totalBoxes });
      setProjectBatchDone({ success, failed, total_detections: totalBoxes });

      // 若当前正打开的图片属于刚批量标注完的项目，刷新画布上的标注
      if (selectedProject && (selectedProject.id || selectedProject.name) === projectId && samImagePath) {
        try {
          const annRes = await annotationApi.getAnnotations(projectId, samImagePath);
          const annData = (annRes.data as any)?.annotations
            ?? (annRes.data as any)?.data?.annotations
            ?? [];
          const newAnnotations: SAMAnnotation[] = [];
          const newBoxes: AnnotationBox[] = [];
          const extraClasses: string[] = [];
          (annData as any[]).forEach((ann: any) => {
            if (ann.bbox && ann.bbox.length === 4) {
              newBoxes.push({ x1: ann.bbox[0], y1: ann.bbox[1], x2: ann.bbox[2], y2: ann.bbox[3] });
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
          if (extraClasses.length > 0) {
            setSamClasses((prev) => [...prev, ...extraClasses]);
          }
          setSamBoxes(newBoxes);
          setSamAnnotations(newAnnotations);
        } catch (e) {
          console.warn('刷新当前图标注失败:', e);
        }
      }
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

  const projectCardStyle = (selected: boolean): React.CSSProperties => ({
    padding: '20px',
    borderRadius: 'var(--radius-lg)',
    border: selected ? '2px solid var(--primary-500)' : '1px solid var(--border-color)',
    background: selected ? 'var(--primary-50)' : 'var(--bg-primary)',
    cursor: 'pointer',
    transition: 'all var(--transition-base)',
    boxShadow: selected ? 'var(--shadow-md)' : 'var(--shadow-sm)',
  });

  /** Hub 顶栏 Draw / Smart 分段按钮（浅色） */
  const hubTabBtn = (active: boolean): React.CSSProperties => ({
    padding: '6px 14px',
    borderRadius: '6px',
    border: active ? 'none' : '1px solid var(--border-color)',
    background: active ? '#3b82f6' : 'var(--bg-primary)',
    color: active ? '#fff' : 'var(--text-secondary)',
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: '12px',
  });

  const closeSamProjectContext = () => {
    setSelectedProject(null);
    setSamImage(null);
    setSamImagePath('');
    setSamFile(null);
    setSamAnnotations([]);
    setSamPoints([]);
    setSamBoxes([]);
    setSamMasks([]);
    setSamSelectedId(null);
  };

  return (
    <div style={pageStyle}>
      {!compactWorkbenchMode && (
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
      )}

      {/* 创建项目 */}
      {!compactWorkbenchMode && showCreate && (
        <Card style={{ border: '1px solid var(--primary-200)', background: 'var(--primary-50)' }}>
          <CardHeader icon="➕" title="创建标注项目" />
          {linkedDataset && (
            <p style={{ marginBottom: 'var(--space-3)', color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
              当前来自数据集联动：`{linkedDataset}`，将自动继承类别（也可在下方手动输入逗号分隔类别覆盖）。
            </p>
          )}
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
      {!compactWorkbenchMode && (
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
              {userYoloModels.length > 0 && (
                <optgroup label="我的模型（训练/上传）">
                  {userYoloModels.map((m) => (
                    <option key={m.path} value={m.path}>{m.label}</option>
                  ))}
                </optgroup>
              )}
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
      )}

      {/* 项目列表 */}
      {!compactWorkbenchMode && (
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
                        onClick={(e) => triggerProjectUpload(project.id || project.name, e)}>
                        {uploadingProjectId === (project.id || project.name) ? '上传中...' : '上传图片'}
                      </Button>
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
                          {userYoloModels.length > 0 && (
                            <optgroup label="我的模型">
                              {userYoloModels.map((m) => (
                                <option key={`pb-${m.path}`} value={m.path}>{m.label}</option>
                              ))}
                            </optgroup>
                          )}
                          <optgroup label="预训练">
                            <option value="yolo11n.pt">YOLO11n</option>
                            <option value="yolo11s.pt">YOLO11s</option>
                            <option value="yolo11m.pt">YOLO11m</option>
                            <option value="yolo26n.pt">YOLO26n</option>
                            <option value="yolo26m.pt">YOLO26m</option>
                          </optgroup>
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
                        ✅ 完成！成功 {projectBatchDone.success} 张
                        {typeof projectBatchDone.total_detections === 'number' && `，共写入 ${projectBatchDone.total_detections} 个框`}
                        {projectBatchDone.failed > 0 ? `，失败 ${projectBatchDone.failed} 张` : ''}
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
      )}

      {!compactWorkbenchMode && (
      <input
        ref={projectUploadInputRef}
        type="file"
        accept="image/*"
        multiple
        onChange={(e) => handleProjectUploadSelect(e.target.files)}
        style={{ display: 'none' }}
      />
      )}

      {/* 生成新版本（数据集）弹窗 */}
      {showVersionModal && selectedProject && (
        <div
          role="dialog"
          aria-modal="true"
          onClick={() => !versionLoading && setShowVersionModal(false)}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 9999,
            background: 'rgba(15, 23, 42, 0.55)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '24px',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: 'min(640px, 100%)',
              maxHeight: '90vh',
              overflowY: 'auto',
              background: 'var(--bg-primary)',
              borderRadius: '12px',
              boxShadow: '0 24px 60px rgba(15, 23, 42, 0.35)',
              border: '1px solid var(--border-color)',
              padding: '20px 24px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <div>
                <div style={{ fontSize: '17px', fontWeight: 700, color: 'var(--text-primary)' }}>
                  🚀 生成数据集新版本
                </div>
                <div style={{ marginTop: '4px', fontSize: '12px', color: 'var(--text-muted)' }}>
                  从项目「{selectedProject.name}」派生一个 YOLO 标准结构的数据集，可直接用于模型训练。
                </div>
              </div>
              <button
                type="button"
                onClick={() => !versionLoading && setShowVersionModal(false)}
                style={{
                  border: 'none',
                  background: 'transparent',
                  fontSize: '20px',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                }}
              >
                ✕
              </button>
            </div>

            <div style={{ marginTop: '16px', display: 'grid', gap: '14px' }}>
              {/* 基础设置 */}
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  数据集名称
                </label>
                <input
                  type="text"
                  value={versionForm.dataset_name}
                  onChange={(e) => setVersionForm((p) => ({ ...p, dataset_name: e.target.value }))}
                  placeholder="留空将自动命名为 <project>_v{N}"
                  style={{
                    width: '100%',
                    padding: '8px 10px',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    background: 'var(--bg-secondary)',
                    color: 'var(--text-primary)',
                    fontSize: '13px',
                  }}
                />
                <div style={{ marginTop: '4px', fontSize: '11px', color: 'var(--text-muted)' }}>
                  目录会创建在 data/datasets/&lt;name&gt;/，并附带 data.yaml
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                    Val 比例
                  </label>
                  <input
                    type="number" min={0} max={0.9} step={0.05}
                    value={versionForm.val_ratio}
                    onChange={(e) => setVersionForm((p) => ({ ...p, val_ratio: Number(e.target.value) }))}
                    style={{ width: '100%', padding: '8px 10px', border: '1px solid var(--border-color)', borderRadius: '6px', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: '13px' }}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                    Test 比例
                  </label>
                  <input
                    type="number" min={0} max={0.9} step={0.05}
                    value={versionForm.test_ratio}
                    onChange={(e) => setVersionForm((p) => ({ ...p, test_ratio: Number(e.target.value) }))}
                    style={{ width: '100%', padding: '8px 10px', border: '1px solid var(--border-color)', borderRadius: '6px', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: '13px' }}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                    随机种子
                  </label>
                  <input
                    type="number" step={1}
                    value={versionForm.seed}
                    onChange={(e) => setVersionForm((p) => ({ ...p, seed: Number(e.target.value) }))}
                    style={{ width: '100%', padding: '8px 10px', border: '1px solid var(--border-color)', borderRadius: '6px', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: '13px' }}
                  />
                </div>
              </div>

              {/* 预处理 */}
              <fieldset style={{ border: '1px solid var(--border-color)', borderRadius: '8px', padding: '10px 14px' }}>
                <legend style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-secondary)', padding: '0 6px' }}>
                  预处理（可选）
                </legend>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: 'var(--text-primary)' }}>
                  <input
                    type="checkbox"
                    checked={versionForm.resize}
                    onChange={(e) => setVersionForm((p) => ({ ...p, resize: e.target.checked }))}
                  />
                  对图像进行 Resize（按长边等比缩放，YOLO 归一化标签无需改写）
                </label>
                {versionForm.resize && (
                  <div style={{ marginTop: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>长边（像素）</span>
                    <input
                      type="number" min={64} max={2048} step={32}
                      value={versionForm.resize_size}
                      onChange={(e) => setVersionForm((p) => ({ ...p, resize_size: Number(e.target.value) }))}
                      style={{ width: '120px', padding: '6px 8px', border: '1px solid var(--border-color)', borderRadius: '6px', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: '13px' }}
                    />
                  </div>
                )}
              </fieldset>

              {/* 数据增强 */}
              <fieldset style={{ border: '1px solid var(--border-color)', borderRadius: '8px', padding: '10px 14px' }}>
                <legend style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-secondary)', padding: '0 6px' }}>
                  数据增强（仅作用于 train，可选）
                </legend>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: 'var(--text-primary)' }}>
                  <input
                    type="checkbox"
                    checked={versionForm.aug_enabled}
                    onChange={(e) => setVersionForm((p) => ({ ...p, aug_enabled: e.target.checked }))}
                  />
                  开启增强（首次生成 v1 通常可保持关闭）
                </label>
                {versionForm.aug_enabled && (
                  <div style={{ marginTop: '8px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 14px' }}>
                    <label style={{ fontSize: '12px', color: 'var(--text-primary)' }}>
                      <input
                        type="checkbox" style={{ marginRight: '6px' }}
                        checked={versionForm.aug_horizontal_flip}
                        onChange={(e) => setVersionForm((p) => ({ ...p, aug_horizontal_flip: e.target.checked }))}
                      />
                      水平翻转
                    </label>
                    <label style={{ fontSize: '12px', color: 'var(--text-primary)' }}>
                      <input
                        type="checkbox" style={{ marginRight: '6px' }}
                        checked={versionForm.aug_vertical_flip}
                        onChange={(e) => setVersionForm((p) => ({ ...p, aug_vertical_flip: e.target.checked }))}
                      />
                      垂直翻转
                    </label>
                    <label style={{ fontSize: '12px', color: 'var(--text-primary)' }}>
                      <input
                        type="checkbox" style={{ marginRight: '6px' }}
                        checked={versionForm.aug_rotate}
                        onChange={(e) => setVersionForm((p) => ({ ...p, aug_rotate: e.target.checked }))}
                      />
                      旋转
                    </label>
                    <label style={{ fontSize: '12px', color: 'var(--text-primary)' }}>
                      <input
                        type="checkbox" style={{ marginRight: '6px' }}
                        checked={versionForm.aug_brightness_contrast}
                        onChange={(e) => setVersionForm((p) => ({ ...p, aug_brightness_contrast: e.target.checked }))}
                      />
                      亮度 / 对比度
                    </label>
                    <label style={{ gridColumn: '1 / -1', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--text-primary)' }}>
                      每张训练图生成
                      <input
                        type="number" min={1} max={10} step={1}
                        value={versionForm.aug_num}
                        onChange={(e) => setVersionForm((p) => ({ ...p, aug_num: Number(e.target.value) }))}
                        style={{ width: '70px', padding: '4px 8px', border: '1px solid var(--border-color)', borderRadius: '6px', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
                      />
                      张增强
                    </label>
                  </div>
                )}
              </fieldset>

              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--text-secondary)' }}>
                <input
                  type="checkbox"
                  checked={versionForm.overwrite}
                  onChange={(e) => setVersionForm((p) => ({ ...p, overwrite: e.target.checked }))}
                />
                若数据集名称已存在则覆盖（谨慎勾选）
              </label>
            </div>

            <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <Button
                variant="secondary"
                onClick={() => setShowVersionModal(false)}
                disabled={versionLoading}
              >
                取消
              </Button>
              <Button
                variant="primary"
                onClick={handleGenerateVersion}
                disabled={versionLoading}
              >
                {versionLoading ? '生成中…' : '生成版本并写入 data.yaml'}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* SAM / Ultralytics Hub 风格三栏工作台 */}
      <Card>
        {!compactWorkbenchMode && (
          <div style={{ marginBottom: 'var(--space-4)' }}>
            <CardHeader icon="✂️" title="标注工作台" />
          </div>
        )}

        {(annotationMode === 'draw' || annotationMode === 'smart') && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: compactWorkbenchMode
                ? 'minmax(120px, 148px) minmax(0, 1fr) minmax(200px, 236px)'
                : 'minmax(168px, 200px) minmax(0, 1fr) minmax(252px, 300px)',
              // 固定高度，确保左侧缩略图区域能正确计算高度并出现滚动条；略增高以放大中间画布区
              height: compactWorkbenchMode ? 'min(94vh, 1040px)' : 'min(92vh, 1040px)',
              maxHeight: '98vh',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-lg)',
              overflow: 'hidden',
              background: 'var(--bg-primary)',
            }}
          >
            {/* 左：图集（Hub 左侧条带） */}
            <aside
              style={{
                display: 'flex',
                flexDirection: 'column',
                background: 'var(--bg-secondary)',
                borderRight: '1px solid var(--border-color)',
                minHeight: 0,
              }}
            >
              <div
                style={{
                  padding: '10px 12px',
                  borderBottom: '1px solid var(--border-color)',
                  color: 'var(--text-primary)',
                  fontWeight: 600,
                  fontSize: '13px',
                }}
              >
                Images
              </div>
              {selectedProject && (
                <div style={{ padding: '10px 12px', borderBottom: '1px solid var(--border-color)' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Project</div>
                  <div style={{ color: 'var(--text-primary)', fontSize: '13px', fontWeight: 600, marginTop: '4px' }}>{selectedProject.name}</div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginTop: '4px' }}>{projectImages.length} 张</div>
                  <button
                    type="button"
                    onClick={handleOpenVersionModal}
                    title="把当前项目的所有标注切分为 train/val(/test)，写入数据集目录并生成 data.yaml，可直接用于模型训练"
                    style={{
                      marginTop: '10px',
                      width: '100%',
                      padding: '6px 10px',
                      borderRadius: '6px',
                      border: '1px solid var(--success-300, #86efac)',
                      background: 'var(--success-50, #f0fdf4)',
                      color: 'var(--success-700, #15803d)',
                      cursor: 'pointer',
                      fontSize: '12px',
                      fontWeight: 600,
                    }}
                  >
                    🚀 生成新版本（v{(generatedVersions?.length || 0) + 1}）
                  </button>
                  {generatedVersions.length > 0 && (
                    <div
                      style={{
                        marginTop: '6px',
                        fontSize: '11px',
                        color: 'var(--text-muted)',
                        lineHeight: 1.5,
                      }}
                    >
                      <div style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>
                        已生成 {generatedVersions.length} 个版本
                      </div>
                      {generatedVersions.slice(0, 3).map((v) => (
                        <div key={v.version || v.dataset_name} title={v.dataset_path}>
                          • {v.version || ''} {v.dataset_name}（
                          {(v.split_counts?.train || 0) + (v.split_counts?.val || 0) + (v.split_counts?.test || 0)} 张）
                        </div>
                      ))}
                    </div>
                  )}
                  <button
                    type="button"
                    onClick={handleProjectExportNdjson}
                    style={{
                      marginTop: '8px',
                      width: '100%',
                      padding: '6px 10px',
                      borderRadius: '6px',
                      border: '1px solid var(--primary-300, #93c5fd)',
                      background: 'var(--primary-50, #eff6ff)',
                      color: 'var(--text-primary)',
                      cursor: 'pointer',
                      fontSize: '12px',
                    }}
                  >
                    导出项目 NDJSON
                  </button>
                  <button
                    type="button"
                    onClick={closeSamProjectContext}
                    style={{
                      marginTop: '8px',
                      width: '100%',
                      padding: '6px 10px',
                      borderRadius: '6px',
                      border: '1px solid var(--border-color)',
                      background: 'var(--bg-primary)',
                      color: 'var(--text-primary)',
                      cursor: 'pointer',
                      fontSize: '12px',
                    }}
                  >
                    关闭项目
                  </button>
                </div>
              )}
              <div style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden', padding: '8px', minHeight: 0 }}>
                {selectedProject && projectImages.length > 0 ? (
                  projectImages.map((img, i) => {
                    const active = samImagePath === img.name || samFile?.name === img.name;
                    return (
                      <button
                        key={i}
                        type="button"
                        onClick={() => handleThumbnailClick(img)}
                        title={`${img.name} — 点击在画布中加载原图`}
                        style={{
                          display: 'block',
                          width: '100%',
                          padding: '4px',
                          marginBottom: '8px',
                          border: active ? '2px solid #22c55e' : '2px solid transparent',
                          borderRadius: '8px',
                          background: active ? 'var(--bg-tertiary)' : 'transparent',
                          cursor: 'pointer',
                        }}
                      >
                        <img
                          src={resolveAnnotationImageFetchUrl(img.url)}
                          alt=""
                          style={{
                            width: '100%',
                            aspectRatio: '1',
                            objectFit: 'contain',
                            background: 'var(--bg-primary)',
                            borderRadius: '6px',
                            display: 'block',
                          }}
                        />
                        <div
                          style={{
                            fontSize: '10px',
                            color: 'var(--text-muted)',
                            marginTop: '4px',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            textAlign: 'left',
                          }}
                        >
                          {img.name}
                        </div>
                      </button>
                    );
                  })
                ) : (
                  <p style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.5, margin: '4px' }}>
                    在下方「标注项目」中点击「开始标注」后，此处显示该项目图集；保存也会写入该项目。
                    仅「打开图片」而未选项目时，可先标注，但保存前必须先进入某个项目。
                  </p>
                )}
              </div>
            </aside>

            {/* 中：模式条 + 工具栏 + 画布 */}
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                minWidth: 0,
                background: 'var(--bg-primary)',
                minHeight: 0,
                height: '100%',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  flexWrap: 'wrap',
                  padding: compactWorkbenchMode ? '6px 10px' : '8px 12px',
                  borderBottom: '1px solid var(--border-color)',
                  background: 'var(--bg-secondary)',
                }}
              >
                <span style={{ color: 'var(--text-muted)', fontSize: '11px', fontWeight: 600, textTransform: 'uppercase' }}>Mode</span>
                <button type="button" onClick={() => setAnnotationMode('draw')} style={hubTabBtn(annotationMode === 'draw')}>
                  Draw
                </button>
                <button type="button" onClick={() => setAnnotationMode('smart')} style={hubTabBtn(annotationMode === 'smart')}>
                  Smart
                </button>
                <span style={{ flex: 1, minWidth: '8px' }} />
                <button
                  type="button"
                  onClick={triggerSamFilePick}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '6px',
                    border: '1px solid var(--border-color)',
                    background: 'var(--bg-primary)',
                    color: 'var(--text-primary)',
                    cursor: 'pointer',
                    fontSize: '12px',
                  }}
                >
                  打开图片
                </button>
              </div>

              <AnnotationToolbar
                tool={samTool}
                currentClass={samCurrentClass}
                classes={samClasses}
                classPalette={samClassPalette}
                suggestions={classSuggestions}
                onToolChange={setSamTool}
                onClassChange={setSamCurrentClass}
                onAddClass={handleAddClass}
                onRenameClass={handleRenameClass}
                onRemoveClass={handleRemoveClass}
                onAutoLabel={handleSamAutoLabel}
                onDetectAll={handleDetectAll}
                onClear={handleSamClear}
                detectAllModel={selectedModel}
                detectAllConfidence={confidence}
                onDetectAllModelChange={setSelectedModel}
                onDetectAllConfidenceChange={setConfidence}
                userDetectModels={userYoloModels}
                onUndo={handleSamUndo}
                onRedo={handleSamRedo}
                onDeleteSelected={handleToolbarTrash}
                onSave={handleSave}
                saveDisabled={samSaveDisabled}
                saveHint={samSaveHint || undefined}
                loading={samLoading}
                autoClassifyOnBox={autoClassifyOnBox}
                onAutoClassifyChange={setAutoClassifyOnBox}
                workMode={annotationMode}
                samVersion={samVersion}
                onSamVersionChange={handleSamVersionChange}
                variant="card"
                compact={compactWorkbenchMode}
              />

              <input
                ref={samFileInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => {
                  handleSamFileSelect(e.target.files ? Array.from(e.target.files) : []);
                  e.currentTarget.value = '';
                }}
                style={{ display: 'none' }}
              />

              <div
                style={{
                  flex: 1,
                  /* 无图时占位区用 flex 撑满，避免内部再出现滚动条 */
                  overflow: samImage ? 'auto' : 'hidden',
                  padding: compactWorkbenchMode ? '6px 8px' : '10px 12px',
                  display: 'flex',
                  flexDirection: 'column',
                  minHeight: 0,
                }}
              >
                {samImage ? (
                  <div
                    style={{
                      position: 'relative',
                      flex: 1,
                      minHeight: 0,
                      width: '100%',
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center',
                    }}
                  >
                    <button
                      type="button"
                      aria-label="全屏查看当前图"
                      title="全屏查看当前原图；Smart 模式下可双击画布同样打开"
                      onClick={() => setImageFullscreenOpen(true)}
                      style={{
                        position: 'absolute',
                        top: 6,
                        right: 6,
                        zIndex: 3,
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                        padding: '6px 10px',
                        borderRadius: '8px',
                        border: '1px solid var(--border-color)',
                        background: 'var(--bg-primary)',
                        color: 'var(--text-primary)',
                        fontSize: '12px',
                        fontWeight: 600,
                        cursor: 'pointer',
                        boxShadow: '0 1px 6px rgba(0,0,0,0.12)',
                      }}
                    >
                      <span aria-hidden>⛶</span> 全屏
                    </button>
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
                      hoveredId={samHoveredId}
                      onPointAdd={handleSamPointAdd}
                      onBoxAdd={handleSamBoxAdd}
                      onBoxDrag={handleSamBoxDrag}
                      smartMode={annotationMode === 'smart'}
                      onSmartImageDoubleClick={() => setImageFullscreenOpen(true)}
                      onCursorMove={handleSmartCursorMove}
                      onMaskAdd={(mask) => {
                        const cls = pickAnnotationClass();
                        if (!cls) return false;
                        const poly = mask.polygons;
                        let minXN = 1;
                        let minYN = 1;
                        let maxXN = 0;
                        let maxYN = 0;
                        for (let j = 0; j + 1 < poly.length; j += 2) {
                          minXN = Math.min(minXN, poly[j]);
                          minYN = Math.min(minYN, poly[j + 1]);
                          maxXN = Math.max(maxXN, poly[j]);
                          maxYN = Math.max(maxYN, poly[j + 1]);
                        }
                        const x1 = minXN * imgSize.width;
                        const y1 = minYN * imgSize.height;
                        const x2 = maxXN * imgSize.width;
                        const y2 = maxYN * imgSize.height;
                        const cid = samClasses.indexOf(cls);
                        const newAnn: SAMAnnotation = {
                          class: cls,
                          class_id: cid >= 0 ? cid : 0,
                          bbox: [x1, y1, x2, y2],
                          segmentation: mask.polygons.join(' '),
                          confidence: 1,
                        };
                        const newBox: AnnotationBox = { x1, y1, x2, y2 };
                        const newMasks = [...samMasks, mask];
                        const newBoxes = [...samBoxes, newBox];
                        const nextAnnotations = [...samAnnotations, newAnn];
                        setSamMasks(newMasks);
                        setSamAnnotations(nextAnnotations);
                        setSamBoxes(newBoxes);
                        saveSamHistory(samPoints, newBoxes, newMasks, nextAnnotations);
                      }}
                      onMaskSelect={(idx) => setSamSelectedId(String(idx))}
                      onAnnotationSelect={setSamSelectedId}
                      currentClass={samCurrentClass}
                    />
                  </div>
                ) : (
                  <label
                    onClick={triggerSamFilePick}
                    style={{
                      flex: 1,
                      minHeight: 0,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: '100%',
                      boxSizing: 'border-box',
                      cursor: 'pointer',
                      border: '2px dashed var(--border-color)',
                      borderRadius: '12px',
                      background: 'var(--bg-secondary)',
                    }}
                  >
                    <div style={{ fontSize: 'clamp(40px, 8vmin, 64px)', marginBottom: '12px' }}>🖼️</div>
                    <p style={{ fontWeight: 600, marginBottom: '4px', color: 'var(--text-primary)', fontSize: 'clamp(15px, 2vmin, 18px)' }}>点击选择图片</p>
                    <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>JPG、PNG、WebP</p>
                  </label>
                )}
              </div>

              {!compactWorkbenchMode && (
                <div
                  style={{
                    padding: '10px 14px',
                    borderTop: '1px solid var(--border-color)',
                    background: 'var(--bg-secondary)',
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)',
                    gap: '8px',
                    fontSize: '11px',
                    color: 'var(--text-secondary)',
                  }}
                >
                  <span>🟢 点击 — 正样本点</span>
                  <span>🔴 Shift+点击 — 负样本点</span>
                  <span>⬜ 拖动 — 框选（SAM 分割）</span>
                  <span>🔺 L — 多边形 · Esc / Backspace 撤销点</span>
                  <span style={{ gridColumn: '1 / -1' }}>⛶ 画布右上角「全屏」查看大图；Smart 下可双击画布同效</span>
                </div>
              )}
            </div>

            {/* 右：对象列表 */}
            <aside
              style={{
                display: 'flex',
                flexDirection: 'column',
                minHeight: 0,
                borderLeft: '1px solid var(--border-color)',
                background: 'var(--bg-primary)',
              }}
            >
              <div
                style={{
                  flexShrink: 0,
                  maxHeight: compactWorkbenchMode ? '38%' : '42%',
                  overflowY: 'auto',
                  borderBottom: '1px solid var(--border-color)',
                  padding: compactWorkbenchMode ? '8px' : '10px 12px',
                  background: 'var(--bg-secondary)',
                }}
              >
                <div
                  style={{
                    fontSize: '12px',
                    fontWeight: 700,
                    color: 'var(--text-primary)',
                    marginBottom: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                  }}
                >
                  仲裁提示
                  {arbitrationScanLoading ? (
                    <span style={{ fontWeight: 500, fontSize: '11px', color: 'var(--text-muted)' }}>分析中…</span>
                  ) : null}
                </div>
                <DisputeAlert items={arbitrationAlerts} />
                {!arbitrationScanLoading && arbitrationEvaluation ? (
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr 1fr',
                      gap: '6px',
                      marginBottom: '10px',
                      fontSize: '11px',
                      color: 'var(--text-secondary)',
                    }}
                  >
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>触发仲裁</span>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                        {arbitrationEvaluation.triggered ? '是 (IoU≥T_iou)' : '否'}
                      </div>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>IoU</span>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                        {arbitrationEvaluation.iou.toFixed(3)}
                      </div>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>T_iou</span>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                        {(arbitrationEvaluation.case?.iouThreshold ?? 0.3).toFixed(2)}
                      </div>
                    </div>
                  </div>
                ) : null}
                <AgentSuggestion items={arbitrationSuggestionItems} />
                {!samFile ? (
                  <p style={{ fontSize: '11px', color: 'var(--text-muted)', margin: '8px 0 0' }}>
                    打开图片后，手绘框完成即可对比模型检测与手绘结果（IoU≥0.3 时进入仲裁）。
                  </p>
                ) : null}
              </div>
              <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                <AnnotationPanel
                  annotations={samAnnotations}
                  selectedId={samSelectedId}
                  hoveredId={samHoveredId}
                  onSelect={setSamSelectedId}
                  onDelete={handleSamDelete}
                  onHover={setSamHoveredId}
                  onClassChange={handleSamClassChange}
                  onExport={handleSamExport}
                  onExportZip={handleSamExportZip}
                  onBulkClassChange={handleSamBulkClassChange}
                  onBulkDelete={handleSamBulkDelete}
                  onSelectionIdsChange={setSamPanelSelectedIds}
                  classes={samClasses}
                  variant="card"
                />
              </div>
            </aside>
          </div>
        )}
      </Card>

      {imageFullscreenOpen && samImage && (
        <div
          role="dialog"
          aria-modal
          aria-label="全屏图片预览"
          onClick={() => setImageFullscreenOpen(false)}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 10000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(15, 23, 42, 0.96)',
            padding: '12px',
            boxSizing: 'border-box',
          }}
        >
          <button
            type="button"
            aria-label="关闭全屏"
            onClick={(e) => {
              e.stopPropagation();
              setImageFullscreenOpen(false);
            }}
            style={{
              position: 'absolute',
              top: 12,
              right: 12,
              zIndex: 1,
              width: 40,
              height: 40,
              border: '1px solid rgba(255,255,255,0.25)',
              borderRadius: '8px',
              background: 'rgba(0,0,0,0.45)',
              color: '#fff',
              fontSize: '20px',
              lineHeight: 1,
              cursor: 'pointer',
            }}
          >
            ×
          </button>
          <div
            role="presentation"
            onClick={(e) => e.stopPropagation()}
            style={{
              width: 'calc(100vw - 24px)',
              height: 'calc(100dvh - 24px)',
              maxWidth: '100%',
              maxHeight: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxSizing: 'border-box',
              overflow: 'hidden',
              filter: 'drop-shadow(0 4px 24px rgba(0,0,0,0.45))',
            }}
          >
            <AnnotationCanvas
              viewOnly
              fitMode="contain"
              maxHeightOverride="100%"
              image={samImage}
              width={imgSize.width}
              height={imgSize.height}
              points={samPoints}
              boxes={samBoxes}
              masks={samMasks}
              annotations={samAnnotations}
              tool="select"
              selectedId={samSelectedId}
              hoveredId={samHoveredId}
              onPointAdd={() => {}}
              onBoxAdd={() => {}}
              onBoxDrag={() => {}}
              smartMode={false}
              onMaskAdd={() => false}
              onMaskSelect={() => {}}
              onAnnotationSelect={() => {}}
              currentClass={samCurrentClass}
            />
          </div>
        </div>
      )}
    </div>
  );
};
