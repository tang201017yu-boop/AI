import React, { useState, useEffect, useLayoutEffect, useRef } from 'react';
import { Card, CardHeader, Button } from '../../components/common';
import { solutionsApi, modelApi, inferenceApi } from '../../services/api';
import { SolutionFeatureIcon, IconMediaImage, IconMediaVideo, IconResultChart } from './solutionIcons';

/** 兼容 axios 体为扁平结构或嵌套 { data: { success, output_path, ... } } */
function unwrapSolutionPayload(res: { data?: unknown }): any {
  const d = res?.data as any;
  if (d == null) return null;
  if (d.data != null && typeof d.data === 'object') {
    const inner = d.data as Record<string, unknown>;
    const looksLikePayload =
      typeof inner.success === 'boolean' ||
      inner.output_path != null ||
      inner.results != null ||
      typeof inner.message === 'string';
    if (looksLikePayload) return inner;
  }
  return d;
}

/** 是否为可内嵌的图片/视频路径（排除旧后端把目录当 output_path 的 /uploads/cropped-objects） */
function looksLikeRenderableUploadUrl(url: unknown): boolean {
  if (url == null || typeof url !== 'string') return false;
  const path = url.split('?')[0].toLowerCase();
  if (!path.startsWith('/uploads/') && !path.includes('/uploads/')) return true;
  return /\.(jpe?g|png|gif|webp|bmp|mp4|webm|avi|mov|mkv)$/i.test(path);
}

function aliasOutputPathToResultImage(payload: any) {
  if (!payload || typeof payload !== 'object') return payload;
  if (payload.output_path && !payload.result_image && !payload.output_image) {
    if (!looksLikeRenderableUploadUrl(payload.output_path)) return payload;
    return { ...payload, result_image: payload.output_path };
  }
  return payload;
}

/** YOLO 默认 COCO 类别英文名 → 中文（自定义训练类别无映射时保留原文） */
const COCO_CLASS_NAME_ZH: Record<string, string> = {
  person: '人',
  bicycle: '自行车',
  car: '汽车',
  motorcycle: '摩托车',
  airplane: '飞机',
  bus: '公交车',
  train: '火车',
  truck: '卡车',
  boat: '船',
  'traffic light': '交通灯',
  'fire hydrant': '消防栓',
  'stop sign': '停车标志',
  'parking meter': '停车计时器',
  bench: '长椅',
  bird: '鸟',
  cat: '猫',
  dog: '狗',
  horse: '马',
  sheep: '羊',
  cow: '牛',
  elephant: '大象',
  bear: '熊',
  zebra: '斑马',
  giraffe: '长颈鹿',
  backpack: '背包',
  umbrella: '雨伞',
  handbag: '手提包',
  tie: '领带',
  suitcase: '行李箱',
  frisbee: '飞盘',
  skis: '滑雪板',
  snowboard: '单板滑雪',
  'sports ball': '运动球',
  kite: '风筝',
  'baseball bat': '棒球棒',
  'baseball glove': '棒球手套',
  skateboard: '滑板',
  surfboard: '冲浪板',
  'tennis racket': '网球拍',
  bottle: '瓶子',
  'wine glass': '酒杯',
  cup: '杯子',
  fork: '叉子',
  knife: '刀',
  spoon: '勺子',
  bowl: '碗',
  banana: '香蕉',
  apple: '苹果',
  sandwich: '三明治',
  orange: '橙子',
  broccoli: '西兰花',
  carrot: '胡萝卜',
  'hot dog': '热狗',
  pizza: '披萨',
  donut: '甜甜圈',
  cake: '蛋糕',
  chair: '椅子',
  couch: '沙发',
  'potted plant': '盆栽',
  bed: '床',
  'dining table': '餐桌',
  toilet: '马桶',
  tv: '电视',
  laptop: '笔记本电脑',
  mouse: '鼠标',
  remote: '遥控器',
  keyboard: '键盘',
  'cell phone': '手机',
  microwave: '微波炉',
  oven: '烤箱',
  toaster: '烤面包机',
  sink: '水槽',
  refrigerator: '冰箱',
  book: '书',
  clock: '时钟',
  vase: '花瓶',
  scissors: '剪刀',
  'teddy bear': '泰迪熊',
  'hair drier': '吹风机',
  toothbrush: '牙刷',
};

function detectionClassLabelZh(englishName: string): string {
  const key = String(englishName).trim().toLowerCase();
  return COCO_CLASS_NAME_ZH[key] ?? englishName;
}

/** 速度估算接口返回的 results.speeds 中单条结构（与 Ultralytics 一致） */
function pickLastMeaningfulSpeed(speeds: unknown[]): { track: number; solution: number } | null {
  if (!Array.isArray(speeds) || speeds.length === 0) return null;
  for (let i = speeds.length - 1; i >= 0; i--) {
    const s = speeds[i] as { track?: number; solution?: number } | null;
    if (s && typeof s === 'object') {
      const tr = Number(s.track ?? 0);
      const sol = Number(s.solution ?? 0);
      if (tr > 0 || sol > 0) return { track: tr, solution: sol };
    }
  }
  return null;
}

/** 从单行推理日志解析 `…ms, 3 person, 1 sports ball | track …` 中的各类数量（与后端写入日志的口径一致） */
function parseClassCountsFromInferenceLogLine(line: string): Array<[string, number]> {
  const summaryMatch = String(line).match(/ms,\s*(.*?)\s*\|\s*track/i);
  if (!summaryMatch?.[1]) return [];
  const summary = summaryMatch[1].trim();
  if (!summary || summary.toLowerCase() === 'no objects') return [];
  const parsed: Record<string, number> = {};
  summary.split(',').forEach((part) => {
    const item = part.trim();
    const m = item.match(/^(\d+)\s+(.+)$/);
    if (!m) return;
    const count = Number(m[1]);
    const cls = m[2].trim();
    if (count > 0 && cls) {
      parsed[cls] = (parsed[cls] || 0) + count;
    }
  });
  return Object.entries(parsed);
}

function deriveObjectsByClass(counting: any): Array<[string, number]> {
  // 优先使用后端直接返回的分类统计
  const raw = counting?.objects_by_class;
  if (raw && typeof raw === 'object') {
    const entries = Object.entries(raw as Record<string, number>)
      .map(([cls, num]) => [cls, Number(num)] as [string, number])
      .filter(([, num]) => Number.isFinite(num) && num > 0);
    if (entries.length > 0) return entries;
  }

  // 兜底：与日志最后一行同一解析逻辑
  const logs = (counting?.recent_inference_logs || []) as string[];
  const lastLog = logs.length ? String(logs[logs.length - 1]) : '';
  return parseClassCountsFromInferenceLogLine(lastLog);
}

// 可用的检测模型列表
const DETECTION_MODELS = [
  // 训练项目模型
  { value: 'wuyu11', label: '安全帽检测 (训练模型)' },
  // YOLO26 系列 (最新)
  { value: 'yolo26n.pt', label: 'YOLO26n (最快)' },
  { value: 'yolo26s.pt', label: 'YOLO26s (轻量)' },
  { value: 'yolo26m.pt', label: 'YOLO26m (平衡)' },
  { value: 'yolo26l.pt', label: 'YOLO26l (高精度)' },
  { value: 'yolo26x.pt', label: 'YOLO26x (最高精度)' },
  // YOLO11 系列
  { value: 'yolo11n.pt', label: 'YOLO11n' },
  { value: 'yolo11s.pt', label: 'YOLO11s' },
  { value: 'yolo11m.pt', label: 'YOLO11m' },
  { value: 'yolo11l.pt', label: 'YOLO11l' },
  { value: 'yolo11x.pt', label: 'YOLO11x' },
];

// 解决方案配置 - 参考 Ultralytics 官方文档
const SOLUTIONS = {
  'object-counting': {
    name: 'object-counting',
    title: '目标计数',
    description: '统计图片或视频中的目标数量，支持区域计数和分类统计',
    color: '#3b82f6',
    supportsVideo: true,
    params: [
      { name: 'model_name', label: '检测模型', type: 'select', default: 'yolo26n.pt', options: DETECTION_MODELS,
        help: '模型越小越快；精度要求高可选 s/m。' },
      { name: 'conf', label: '置信度阈值', type: 'number', default: 0.25, min: 0, max: 1, step: 0.05,
        help: '越高框越少但更准；越低检出更多但易误报。' },
      { name: 'iou', label: '检测 NMS IoU', type: 'number', default: 0.7, min: 0.1, max: 0.95, step: 0.05,
        help: '传给 Ultralytics track：目标重叠多时略降低(如 0.55~0.65)可减轻「多目标被压成一条轨迹」；过低易同一物体多框。' },
      { name: 'max_det', label: '单帧最大检测数', type: 'number', default: 300, min: 1, max: 1000, step: 1,
        help: '密集场景漏检时可提高；过大更耗显存与时间。' },
      {
        name: 'tracker',
        label: '跟踪器配置',
        type: 'text',
        default: '',
        placeholder: '留空使用 botsort.yaml',
        help: '可选：bytetrack.yaml 等（需 Ultralytics 可解析）。跨线进出计数依赖稳定 track。',
      },
      { name: 'region_type', label: '区域类型', type: 'select', default: 'polygon',
        options: [
          { value: 'polygon', label: '多边形区域' },
          { value: 'line', label: '直线(进出计数)' },
        ],
        help: '多边形：框定一片区域做统计；直线：适合跨线进出计数（视频效果更好）。',
      },
      {
        name: 'region_points',
        label: '区域坐标(JSON)',
        type: 'text',
        placeholder: '留空则用画面默认区域。多边形示例: [[20,400],[1260,400],[1260,720],[20,720]]',
        help: '像素坐标 [[x,y],...] 至少 3 点构成多边形；直线为 2 点 [[x1,y1],[x2,y2]]。与上传分辨率一致。',
      },
      { name: 'show_in', label: '在画面上标注「进入」计数', type: 'checkbox', default: true,
        help: '关闭可减少画面文字干扰。' },
      { name: 'show_out', label: '在画面上标注「离开」计数', type: 'checkbox', default: true,
        help: '进出计数依赖视频连续帧；单张图多为 0。' },
      { name: 'line_width', label: '线条宽度', type: 'number', default: 2, min: 1, max: 10,
        help: '区域边线、检测框线粗细。' },
    ]
  },
  'heatmap': {
    name: 'heatmap',
    title: '热力图生成',
    description: '生成目标检测密度热力图，可视化热点区域',
    color: '#ef4444',
    supportsVideo: true,
    params: [
      { name: 'model_name', label: '检测模型', type: 'select', default: 'yolo26n.pt', options: DETECTION_MODELS },
      { name: 'conf', label: '置信度阈值', type: 'number', default: 0.25, min: 0, max: 1, step: 0.05 },
      { name: 'colormap', label: '热力图颜色', type: 'select', default: 'COLORMAP_JET',
        options: [
          { value: 'COLORMAP_JET', label: 'JET (蓝-青-黄-红)' },
          { value: 'COLORMAP_VIRIDIS', label: 'VIRIDIS (绿-黄)' },
          { value: 'COLORMAP_PLASMA', label: 'PLASMA (蓝-紫-黄)' },
          { value: 'COLORMAP_INFERNO', label: 'INFERNO (黑-紫-橙-黄)' },
          { value: 'COLORMAP_MAGMA', label: 'MAGMA (黑-紫-橙-白)' },
        ]
      },
    ]
  },
  'speed-estimation': {
    name: 'speed-estimation',
    title: '速度估算',
    description: '估算视频中移动目标的速度（需提供参考距离）',
    color: '#f97316',
    supportsVideo: true,
    params: [
      { name: 'model_name', label: '检测模型', type: 'select', default: 'yolo26n.pt', options: DETECTION_MODELS },
      { name: 'conf', label: '置信度阈值', type: 'number', default: 0.25, min: 0, max: 1, step: 0.05 },
      {
        name: 'pixel_to_meter',
        label: '像素距离标定',
        type: 'number',
        default: 10,
        min: 0.1,
        step: 0.1,
        help: '与 Ultralytics 测速标定相关：数值含义以后端实现为准，一般表示「每米对应多少像素」类比例，可按画面标定调整。',
      },
      { name: 'line_width', label: '线条宽度', type: 'number', default: 2, min: 1, max: 10 },
    ]
  },
  'distance-calculation': {
    name: 'distance-calculation',
    title: '距离计算',
    description: '计算图像中检测目标之间的距离',
    color: '#8b5cf6',
    supportsVideo: false,
    params: [
      { name: 'model_name', label: '检测模型', type: 'select', default: 'yolo26n.pt', options: DETECTION_MODELS },
      { name: 'conf', label: '置信度阈值', type: 'number', default: 0.25, min: 0, max: 1, step: 0.05 },
    ]
  },
  'object-blur': {
    name: 'object-blur',
    title: '目标模糊',
    description: '对检测到的目标进行模糊处理，保护隐私',
    color: '#64748b',
    supportsVideo: true,
    params: [
      { name: 'model_name', label: '检测模型', type: 'select', default: 'yolo26n.pt', options: DETECTION_MODELS },
      { name: 'blur_ratio', label: '模糊强度', type: 'number', default: 20, min: 5, max: 50 },
      { name: 'conf', label: '置信度阈值', type: 'number', default: 0.25, min: 0, max: 1, step: 0.05 },
    ]
  },
  'object-crop': {
    name: 'object-crop',
    title: '目标裁剪',
    description: '从图像中自动裁剪出检测到的目标',
    color: '#10b981',
    supportsVideo: false,
    params: [
      { name: 'model_name', label: '检测模型', type: 'select', default: 'yolo26n.pt', options: DETECTION_MODELS },
      { name: 'conf', label: '置信度阈值', type: 'number', default: 0.25, min: 0, max: 1, step: 0.05 },
      { name: 'padding', label: '边距(%)', type: 'number', default: 10, min: 0, max: 50 },
    ]
  },
  'queue-management': {
    name: 'queue-management',
    title: '队列管理',
    description: '监控队列长度，分析等待时间',
    color: '#06b6d4',
    supportsVideo: true,
    params: [
      { name: 'model_name', label: '检测模型', type: 'select', default: 'yolo26n.pt', options: DETECTION_MODELS },
      { name: 'conf', label: '置信度阈值', type: 'number', default: 0.25, min: 0, max: 1, step: 0.05 },
      { name: 'line_width', label: '线条宽度', type: 'number', default: 2, min: 1, max: 10 },
    ]
  },
  'parking-management': {
    name: 'parking-management',
    title: '停车管理',
    description: '检测停车位占用情况，管理车辆进出',
    color: '#eab308',
    supportsVideo: true,
    params: [
      { name: 'model_name', label: '检测模型', type: 'select', default: 'yolo26n.pt', options: DETECTION_MODELS },
      { name: 'conf', label: '置信度阈值', type: 'number', default: 0.25, min: 0, max: 1, step: 0.05 },
      { name: 'line_width', label: '线条宽度', type: 'number', default: 2, min: 1, max: 10 },
      {
        name: 'parking_slots',
        label: '车位坐标(JSON)',
        type: 'text',
        placeholder: '留空则用后端默认示例车位。格式: 每个车位为 [[x,y],...] 至少 3 点的多边形数组',
        help: '三维数组：多个车位，每个车位一圈顶点。坐标需与画面分辨率一致。',
      },
    ]
  },
  'vision-eye': {
    name: 'vision-eye',
    title: '视觉安防',
    description: '周界入侵检测，异常行为识别',
    color: '#ec4899',
    supportsVideo: true,
    params: [
      { name: 'model_name', label: '检测模型', type: 'select', default: 'yolo26n.pt', options: DETECTION_MODELS },
      { name: 'roi', label: '监控区域', type: 'text', placeholder: 'x1,y1,x2,y2' },
    ]
  },
  'workout-monitoring': {
    name: 'workout-monitoring',
    title: '健身监测',
    description: '人体/器械检测与计数（可选姿态模型）',
    color: '#14b8a6',
    supportsVideo: true,
    params: [
      { name: 'model_name', label: '检测模型', type: 'select', default: 'yolo11n.pt', options: DETECTION_MODELS },
      { name: 'conf', label: '置信度阈值', type: 'number', default: 0.25, min: 0, max: 1, step: 0.05 },
    ]
  },
};

/** 显示「统一处理统计」灰区（进出计数/队列/热力图等）；距离/裁剪等仅用各自专属块 */
const SOLUTIONS_WITH_STATS_PANEL = new Set<string>([
  'object-counting',
  'vision-eye',
  'workout-monitoring',
  'parking-management',
  'queue-management',
  'speed-estimation',
  'heatmap',
  'object-blur',
]);

interface Param {
  name: string;
  label: string;
  type: string;
  default?: any;
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  /** 参数说明，显示在控件下方 */
  help?: string;
}

interface SolutionConfig {
  name: string;
  title: string;
  description: string;
  color: string;
  supportsVideo: boolean;
  params: Param[];
}

export const SolutionRunner: React.FC = () => {
  const [selectedSolution, setSelectedSolution] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [resultVideoError, setResultVideoError] = useState(false);
  const [loading, setLoading] = useState(false);
  const [params, setParams] = useState<Record<string, any>>({});

  /** 切换方案或返回列表时递增，丢弃尚未完成的异步 setResult */
  const solutionRunRef = useRef(0);
  /** 健身监测：推理日志滚到底，便于与「末帧摘要」对照 */
  const inferenceLogScrollRef = useRef<HTMLDivElement>(null);

  const solution = selectedSolution ? SOLUTIONS[selectedSolution as keyof typeof SOLUTIONS] : null;

  useLayoutEffect(() => {
    solutionRunRef.current += 1;
    setResult(null);
    setResultVideoError(false);
    setLoading(false);
    if (selectedSolution === null) {
      setFile(null);
      setPreview((prev) => {
        if (prev && prev.startsWith('blob:')) {
          try {
            URL.revokeObjectURL(prev);
          } catch {
            /* ignore */
          }
        }
        return null;
      });
    }
  }, [selectedSolution]);

  useLayoutEffect(() => {
    if (selectedSolution !== 'workout-monitoring') return;
    const el = inferenceLogScrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [selectedSolution, result]);

  // 初始化参数
  useEffect(() => {
    if (solution) {
      const defaultParams: Record<string, any> = {};
      solution.params.forEach(p => {
        defaultParams[p.name] = p.default;
      });
      setParams(defaultParams);
    }
  }, [solution]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setResultVideoError(false);
    }
  };

  const handleParamChange = (name: string, value: any) => {
    setParams(prev => ({ ...prev, [name]: value }));
  };

  const paramHelpStyle: React.CSSProperties = {
    margin: '6px 0 0 0',
    fontSize: '0.75rem',
    color: 'var(--text-secondary)',
    lineHeight: 1.45,
  };

  /** 区域/车位类 JSON：空为合法；非空须能 parse 且为数组 */
  const regionJsonInvalid = (raw: string): boolean => {
    const s = String(raw ?? '').trim();
    if (!s) return false;
    try {
      const v = JSON.parse(s);
      return !Array.isArray(v);
    } catch {
      return true;
    }
  };

  const handleProcess = async () => {
    if (!file || !selectedSolution) return;

    const runAtStart = solutionRunRef.current;
    const commitResult = (payload: any) => {
      if (runAtStart !== solutionRunRef.current) return;
      setResult(payload);
    };

    setLoading(true);
    setResultVideoError(false);
    try {
      // 健身监测：图片走快速推理；视频走方案接口，返回处理后视频
      if (selectedSolution === 'workout-monitoring') {
        if (file.type.startsWith('video')) {
          const formData = new FormData();
          formData.append('file', file);
          formData.append('model_name', String(params.model_name || 'yolo11n.pt'));
          const conf = Number(params.conf ?? 0.25);
          formData.append('conf', String(conf));
          const res = await solutionsApi.workoutMonitoring(formData);
          const data: any = aliasOutputPathToResultImage(unwrapSolutionPayload(res));
          commitResult(data);
          return;
        }
        const modelName = String(params.model_name || 'yolo11n.pt');
        const conf = Number(params.conf ?? 0.25);
        const res = await inferenceApi.image(file, modelName, conf);
        const data: any = unwrapSolutionPayload(res);
        // 统一结果字段，复用当前页面展示逻辑
        if (data?.annotated_image && !data?.output_image) {
          data.output_image = data.annotated_image;
        }
        commitResult(aliasOutputPathToResultImage(data));
        return;
      }

      const formData = new FormData();
      formData.append('file', file);

      if (selectedSolution === 'object-counting' && regionJsonInvalid(params.region_points ?? '')) {
        alert('区域坐标 JSON 格式不正确，请检查或为「留空」使用默认区域。');
        return;
      }
      if (selectedSolution === 'parking-management' && regionJsonInvalid(params.parking_slots ?? '')) {
        alert('车位坐标 JSON 格式不正确，请检查或为「留空」使用示例默认车位。');
        return;
      }

      // 添加参数
      Object.entries(params).forEach(([key, value]) => {
        formData.append(key, String(value));
      });

      // 视觉安防：将 roi(x1,y1,x2,y2) 转为后端可识别的 region_points(JSON)
      if (selectedSolution === 'vision-eye') {
        const roiRaw = String(params.roi || '').trim();
        if (roiRaw) {
          const nums = roiRaw.split(',').map((n: string) => Number(n.trim())).filter((n: number) => !Number.isNaN(n));
          if (nums.length === 4) {
            const [x1, y1, x2, y2] = nums;
            const region = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
            formData.set('region_points', JSON.stringify(region));
          }
        }
      }

      let apiFunc;
      switch (selectedSolution) {
        case 'object-counting':
          apiFunc = solutionsApi.objectCounting;
          break;
        case 'heatmap':
          apiFunc = solutionsApi.heatmap;
          break;
        case 'speed-estimation':
          apiFunc = solutionsApi.speedEstimation;
          break;
        case 'distance-calculation':
          apiFunc = solutionsApi.distanceCalculation;
          break;
        case 'object-blur':
          apiFunc = solutionsApi.objectBlur;
          break;
        case 'object-crop':
          apiFunc = solutionsApi.objectCrop;
          break;
        case 'queue-management':
          apiFunc = solutionsApi.queueManagement;
          break;
        case 'parking-management':
          apiFunc = solutionsApi.parkingManagement;
          break;
        case 'vision-eye':
          apiFunc = solutionsApi.visionEye;
          break;
        case 'workout-monitoring':
          apiFunc = solutionsApi.workoutMonitoring;
          break;
        default:
          alert('未实现的解决方案');
          return;
      }

      const res = await apiFunc(formData);
      let data = aliasOutputPathToResultImage(unwrapSolutionPayload(res));

      // 热力图兼容异步任务接口：若首次仅返回 task_id，则轮询状态直到拿到 output_path
      if (
        selectedSolution === 'heatmap' &&
        data?.task_id &&
        !data?.output_path &&
        !data?.result_image
      ) {
        const maxAttempts = 60; // 最长轮询约 2 分钟
        let finished = false;
        for (let i = 0; i < maxAttempts; i++) {
          await new Promise((resolve) => setTimeout(resolve, 2000));
          const statusRes = await solutionsApi.heatmapStatus(String(data.task_id));
          const statusData = aliasOutputPathToResultImage(unwrapSolutionPayload(statusRes));

          // 实时更新进度文案，避免用户误以为卡住
          commitResult({
            ...data,
            ...statusData,
            success: statusData?.status !== 'failed',
          });

          if (statusData?.status === 'completed' && (statusData?.output_path || statusData?.result_image)) {
            data = { ...data, ...statusData, success: true };
            finished = true;
            break;
          }
          if (statusData?.status === 'failed') {
            data = {
              ...data,
              ...statusData,
              success: false,
              message: statusData?.message || '热力图处理失败',
            };
            finished = true;
            break;
          }
        }
        if (!finished) {
          data = {
            ...data,
            success: false,
            message: '热力图处理超时，请稍后重试或检查后端日志',
          };
        }
      }

      commitResult(data);
      if (data?.success === false) {
        alert(`处理失败: ${data?.message || '请检查参数或后端日志'}`);
      }
    } catch (error) {
      console.error('处理失败:', error);
      const msg = error instanceof Error ? error.message : '处理失败，请重试';
      alert(`处理失败: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setResultVideoError(false);
  };

  // 渲染解决方案列表
  if (!selectedSolution) {
    return (
      <div>
        <h1 style={{ marginBottom: 'var(--space-6)', fontFamily: 'DM Sans', fontWeight: 700 }}>
          智能解决方案
        </h1>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 'var(--space-4)' }}>
          {Object.values(SOLUTIONS).map((sol) => (
            <Card
              key={sol.name}
              variant="elevated"
              style={{ cursor: 'pointer' }}
              onClick={() => setSelectedSolution(sol.name)}
            >
              <div style={{
                width: '48px',
                height: '48px',
                borderRadius: '12px',
                background: sol.color + '20',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: sol.color,
                marginBottom: 'var(--space-3)'
              }}>
                <SolutionFeatureIcon name={sol.name} size={26} />
              </div>
              <h3 style={{ fontSize: '1.125rem', fontWeight: 600, marginBottom: 'var(--space-1)' }}>
                {sol.title}
              </h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: 'var(--space-3)' }}>
                {sol.description}
              </p>
              <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                <span style={{ fontSize: '0.75rem', background: 'var(--gray-100)', padding: '2px 8px', borderRadius: '4px', color: 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <IconMediaImage size={13} />
                  图片
                </span>
                {sol.supportsVideo && (
                  <span style={{ fontSize: '0.75rem', background: 'var(--gray-100)', padding: '2px 8px', borderRadius: '4px', color: 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                    <IconMediaVideo size={13} />
                    视频
                  </span>
                )}
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  // 渲染解决方案详情
  if (!solution) return null;

  // 统一提取媒体结果路径（兼容不同后端返回结构）
  const resolveMediaPath = (data: any): string => {
    if (!data) return '';
    const crops = data?.results?.cropped_images ?? data?.cropped_images;
    const firstCrop = Array.isArray(crops) ? crops.find((x: any) => x?.crop_path) : undefined;

    let candidate =
      data.result_image ||
      data.output_path ||
      data.output_image ||
      data.annotated_image ||
      data?.results?.result_image ||
      data?.results?.output_path ||
      data?.results?.output_image ||
      data?.results?.annotated_image ||
      '';

    // 目标裁剪：旧 API 可能把 output_path/result_image 写成目录或无效路径；有裁剪列表时强制用第一张图
    if (firstCrop?.crop_path && (!candidate || !looksLikeRenderableUploadUrl(candidate))) {
      candidate = firstCrop.crop_path;
    }
    if (!candidate) {
      if (firstCrop?.crop_path) candidate = firstCrop.crop_path;
    }
    if (!candidate) return '';
    if (String(candidate).startsWith('data:')) return String(candidate);
    if (String(candidate).startsWith('http')) return String(candidate);
    const rel = String(candidate);
    // 相对路径：与当前页同源（开发时经 Vite 代理到后端）
    if (rel.startsWith('/')) return `${window.location.origin}${rel}`;
    return `${window.location.origin}/${rel}`;
  };

  const mediaUrlWithBust = (url: string, isVideo: boolean) => {
    if (!url || url.startsWith('data:')) return url;
    const sep = url.includes('?') ? '&' : '?';
    return `${url}${sep}${isVideo ? 't' : '_'}=${Date.now()}`;
  };

  return (
    <div>
      {/* 头部 */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
        <Button variant="ghost" onClick={() => setSelectedSolution(null)}>
          ← 返回
        </Button>
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '10px',
          background: solution.color + '20',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: solution.color,
        }}>
          <SolutionFeatureIcon name={solution.name} size={22} />
        </div>
        <div>
          <h1 style={{ fontFamily: 'DM Sans', fontWeight: 700, margin: 0 }}>
            {solution.title}
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', margin: 0 }}>
            {solution.description}
          </p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 'var(--space-6)' }}>
        {/* 左侧：图片/视频预览 */}
        <Card>
          <CardHeader
            icon={(
              <span style={{ color: solution.color, display: 'flex' }}>
                <SolutionFeatureIcon name={solution.name} size={22} />
              </span>
            )}
            title="输入文件"
          />
          {!preview ? (
            <label style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 'var(--space-12)',
              border: '2px dashed var(--border-color)',
              borderRadius: 'var(--radius-lg)',
              cursor: 'pointer',
              background: 'var(--gray-50)',
            }}>
              <span style={{ fontSize: '3rem', marginBottom: 'var(--space-3)' }}>📁</span>
              <span style={{ fontWeight: 500, marginBottom: 'var(--space-1)' }}>
                点击上传 {solution.supportsVideo ? '图片或视频' : '图片'}
              </span>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                支持 JPG, PNG, MP4 等格式
              </span>
              <input
                type="file"
                accept={solution.supportsVideo ? "image/*,video/*" : "image/*"}
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
            </label>
          ) : (
            <div>
              {file?.type.startsWith('video') ? (
                <video
                  src={preview}
                  controls
                  style={{ width: '100%', borderRadius: 'var(--radius-md)' }}
                />
              ) : (
                <img
                  src={preview}
                  alt="Preview"
                  style={{ width: '100%', borderRadius: 'var(--radius-md)' }}
                />
              )}
              <div style={{ marginTop: 'var(--space-3)', display: 'flex', gap: 'var(--space-2)' }}>
                <Button variant="secondary" size="sm" onClick={handleReset}>
                  更换文件
                </Button>
              </div>
            </div>
          )}

          {/* 结果显示 */}
          {result && (
            <div style={{ marginTop: 'var(--space-6)' }}>
              <CardHeader
                icon={(
                  <span style={{ color: 'var(--primary-500, #3b82f6)', display: 'flex' }}>
                    <IconResultChart size={20} />
                  </span>
                )}
                title="处理结果"
              />
              {result.success === false && (
                <div style={{
                  padding: 'var(--space-3)',
                  marginBottom: 'var(--space-3)',
                  background: '#fef2f2',
                  color: '#b91c1c',
                  borderRadius: 'var(--radius-md)',
                  fontSize: '0.875rem',
                }}>
                  {result.message || '处理失败'}
                </div>
              )}
              {selectedSolution === 'heatmap' && result.status === 'processing' && (
                <div style={{
                  padding: 'var(--space-3)',
                  marginBottom: 'var(--space-3)',
                  background: '#eff6ff',
                  color: '#1d4ed8',
                  borderRadius: 'var(--radius-md)',
                  fontSize: '0.875rem',
                }}>
                  {`热力图处理中：${result.progress ?? 0}% ${result.message ? `- ${result.message}` : ''}`}
                </div>
              )}
              {/* 处理图片/视频结果 */}
              {(() => {
                const mediaPath = resolveMediaPath(result);
                if (!mediaPath) return null;
                const lower = mediaPath.toLowerCase();
                const isVideo = lower.endsWith('.mp4') || lower.endsWith('.avi') || lower.endsWith('.mov') || lower.endsWith('.mkv') || lower.endsWith('.webm');
                return isVideo ? (
                  <div>
                    <video
                      src={mediaUrlWithBust(mediaPath, true)}
                      controls
                      playsInline
                      preload="metadata"
                      onLoadedData={() => setResultVideoError(false)}
                      onError={() => setResultVideoError(true)}
                      style={{ width: '100%', borderRadius: 'var(--radius-md)', marginBottom: 'var(--space-3)', backgroundColor: '#000' }}
                    />
                    {resultVideoError && (
                      <p style={{ fontSize: '0.8125rem', color: '#b45309', marginBottom: 'var(--space-2)' }}>
                        浏览器无法解码该视频编码时会出现黑屏或无法播放。请使用下方「下载视频」，或在服务器安装 ffmpeg 后重新处理（后端会输出 H.264 以兼容网页播放）。
                      </p>
                    )}
                    <a href={mediaPath + (mediaPath.includes('?') ? '&' : '?') + 'download=1'} download style={{ display: 'inline-block', marginTop: '8px', color: '#3b82f6', textDecoration: 'none' }}>
                      下载视频
                    </a>
                    <a href={mediaPath} target="_blank" rel="noreferrer" style={{ display: 'inline-block', marginTop: '8px', marginLeft: '12px', color: '#3b82f6', textDecoration: 'none' }}>
                      打开结果文件
                    </a>
                  </div>
                ) : (
                  <div>
                    <img
                      src={mediaUrlWithBust(mediaPath, false)}
                      alt="Result"
                      style={{ width: '100%', borderRadius: 'var(--radius-md)', marginBottom: 'var(--space-3)' }}
                    />
                    <a href={mediaPath} target="_blank" rel="noreferrer" style={{ display: 'inline-block', marginTop: '4px', color: '#3b82f6', textDecoration: 'none' }}>
                      打开结果文件
                    </a>
                  </div>
                );
              })()}

              {selectedSolution === 'object-crop' && result.success !== false && result.status !== 'processing' && (() => {
                const crops = (result.results?.cropped_images ?? result.cropped_images) as
                  | Array<{ crop_path?: string; class_name?: string }>
                  | undefined;
                if (!crops?.length) {
                  return (
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: 'var(--space-3)' }}>
                      未检测到可裁剪目标，可尝试换图或调低置信度阈值。
                    </p>
                  );
                }
                return (
                  <div style={{ marginTop: 'var(--space-4)' }}>
                    <h4 style={{ margin: '0 0 var(--space-2)', fontSize: '0.9375rem', fontWeight: 600 }}>
                      全部裁剪块（{result.results?.total_crops ?? result.total_crops ?? crops.length}）
                    </h4>
                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(112px, 1fr))',
                        gap: 'var(--space-3)',
                      }}
                    >
                      {crops.filter((item) => item.crop_path).map((item, idx) => {
                        const rel = String(item.crop_path);
                        const abs =
                          rel.startsWith('http') ? rel : rel.startsWith('/') ? `${window.location.origin}${rel}` : `${window.location.origin}/${rel}`;
                        return (
                          <a
                            key={`${rel}-${idx}`}
                            href={abs}
                            target="_blank"
                            rel="noreferrer"
                            style={{ textAlign: 'center', textDecoration: 'none', color: 'inherit' }}
                          >
                            <img
                              src={mediaUrlWithBust(abs, false)}
                              alt={item.class_name || `crop-${idx}`}
                              style={{
                                width: '100%',
                                aspectRatio: '1',
                                objectFit: 'cover',
                                borderRadius: 'var(--radius-md)',
                                border: '1px solid var(--border-color)',
                                display: 'block',
                              }}
                            />
                            <div style={{ fontSize: '0.75rem', marginTop: '6px', color: 'var(--text-secondary)' }}>
                              {detectionClassLabelZh(String(item.class_name ?? ''))}
                            </div>
                          </a>
                        );
                      })}
                    </div>
                  </div>
                );
              })()}

              {result.success !== false && result.status !== 'processing' && !resolveMediaPath(result) && !(result.results || result.counting) && selectedSolution !== 'object-crop' && (
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                  未返回结果图地址，请检查后端 `/uploads` 挂载与前端代理配置。
                </p>
              )}

              {/* 媒体无法内嵌时兜底链接 */}
              {!resolveMediaPath(result) && (result.output_path || result?.results?.output_path) && (
                <a
                  href={window.location.origin + (result.output_path || result?.results?.output_path)}
                  target="_blank"
                  rel="noreferrer"
                  style={{ display: 'inline-block', marginBottom: 'var(--space-3)', color: '#3b82f6', textDecoration: 'none' }}
                >
                  打开处理结果文件
                </a>
              )}
              {/* 处理统计：仅当前所选方案需要时展示，避免距离计算等出现进出计数块 */}
              {SOLUTIONS_WITH_STATS_PANEL.has(selectedSolution ?? '') &&
                (
                  result.results ||
                  result.counting ||
                  result?.in_count != null ||
                  result?.out_count != null ||
                  result?.total_frames != null ||
                  result?.detected_objects != null ||
                  (result?.objects_by_class &&
                    typeof result.objects_by_class === 'object' &&
                    Object.keys(result.objects_by_class).length > 0)
                ) && (
                <div style={{ padding: 'var(--space-4)', background: 'var(--gray-50)', borderRadius: 'var(--radius-md)' }}>
                  {(() => {
                    const counting = result.results || result.counting || result || {};
                    const parkingShape =
                      selectedSolution === 'parking-management' &&
                      typeof counting.total_slots === 'number' &&
                      typeof counting.current_occupied === 'number';

                    if (parkingShape) {
                      const totalSlots = Number(counting.total_slots);
                      const occupied = Number(counting.current_occupied);
                      const free =
                        counting.current_free != null
                          ? Number(counting.current_free)
                          : Math.max(0, totalSlots - occupied);
                      const avgOcc =
                        counting.avg_occupied != null ? Number(counting.avg_occupied) : null;
                      const maxOcc =
                        counting.max_occupied != null ? Number(counting.max_occupied) : null;
                      return (
                        <>
                          <h4 style={{ marginBottom: 'var(--space-2)' }}>车位占用（与画面一致）:</h4>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                            <span>占用（Occupied）:</span>
                            <span style={{ fontWeight: 600 }}>
                              {occupied} / {totalSlots}
                            </span>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                            <span>空闲（Free）:</span>
                            <span style={{ fontWeight: 600 }}>{free}</span>
                          </div>
                          {avgOcc != null && !Number.isNaN(avgOcc) && (
                            <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                              <span>全程平均占用车位数:</span>
                              <span style={{ fontWeight: 600 }}>{avgOcc.toFixed(2)}</span>
                            </div>
                          )}
                          {maxOcc != null && !Number.isNaN(maxOcc) && (
                            <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                              <span>峰值占用车位数:</span>
                              <span style={{ fontWeight: 600 }}>{maxOcc}</span>
                            </div>
                          )}
                          {counting.total_frames != null && (
                            <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                              <span>处理帧数:</span>
                              <span style={{ fontWeight: 600 }}>{String(counting.total_frames)}</span>
                            </div>
                          )}
                        </>
                      );
                    }

                    // 含 frame_counts 的即为队列接口形态；停车接口回退到队列时仍选「停车」也会走到此分支
                    const queueShape =
                      Array.isArray(counting.frame_counts) &&
                      (selectedSolution === 'queue-management' || selectedSolution === 'parking-management');
                    if (queueShape) {
                      const fc = counting.frame_counts as number[];
                      const lastCount = fc.length ? fc[fc.length - 1] : 0;
                      const maxQ =
                        counting.max_queue_count != null ? Number(counting.max_queue_count) : Math.max(0, ...fc);
                      const avgQ =
                        counting.avg_queue_count != null
                          ? Number(counting.avg_queue_count)
                          : fc.length
                            ? fc.reduce((a, b) => a + b, 0) / fc.length
                            : 0;
                      return (
                        <>
                          <h4 style={{ marginBottom: 'var(--space-2)' }}>队列统计（与画面监测区一致）:</h4>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                            <span>末帧区域内目标数:</span>
                            <span style={{ fontWeight: 600 }}>{lastCount}</span>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                            <span>全程最大:</span>
                            <span style={{ fontWeight: 600 }}>{maxQ}</span>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                            <span>全程平均:</span>
                            <span style={{ fontWeight: 600 }}>{avgQ.toFixed(1)}</span>
                          </div>
                          {counting.total_frames != null && (
                            <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                              <span>处理帧数:</span>
                              <span style={{ fontWeight: 600 }}>{String(counting.total_frames)}</span>
                            </div>
                          )}
                        </>
                      );
                    }

                    const speedShape =
                      selectedSolution === 'speed-estimation' && Array.isArray(counting.speeds);
                    if (speedShape) {
                      const speeds = counting.speeds as unknown[];
                      const last = pickLastMeaningfulSpeed(speeds);
                      const fc = counting.frame_count ?? counting.total_frames;
                      return (
                        <>
                          <h4 style={{ marginBottom: 'var(--space-2)' }}>速度估算（与画面标注一致）:</h4>
                          {fc != null && (
                            <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                              <span>处理帧数:</span>
                              <span style={{ fontWeight: 600 }}>{String(fc)}</span>
                            </div>
                          )}
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                            <span>速度采样条数:</span>
                            <span style={{ fontWeight: 600 }}>{speeds.length}</span>
                          </div>
                          {last ? (
                            <>
                              <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                                <span>末次有效 track（像素/帧）:</span>
                                <span style={{ fontWeight: 600 }}>{last.track.toFixed(2)}</span>
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                                <span>末次有效 solution（ms/帧）:</span>
                                <span style={{ fontWeight: 600 }}>{last.solution.toFixed(2)}</span>
                              </div>
                            </>
                          ) : (
                            <p style={{ margin: 'var(--space-2) 0 0', fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                              未记录到有效速度轨迹，请确认视频中有在标定区域内移动的目标。
                            </p>
                          )}
                        </>
                      );
                    }

                    if (selectedSolution === 'heatmap') {
                      const tf = Number(result.total_frames ?? counting.total_frames ?? 0);
                      return (
                        <>
                          <h4 style={{ marginBottom: 'var(--space-2)' }}>热力图处理:</h4>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                            <span>累计处理帧数:</span>
                            <span style={{ fontWeight: 600 }}>{tf}</span>
                          </div>
                          {result.message && (
                            <p style={{ margin: 'var(--space-2) 0 0', fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                              {String(result.message)}
                            </p>
                          )}
                          <p style={{ margin: 'var(--space-2) 0 0', fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                            结果画面为检测密度热力叠加，无进出计数含义。
                          </p>
                        </>
                      );
                    }

                    if (selectedSolution === 'object-blur') {
                      const tf = Number(result.total_frames ?? counting.total_frames ?? 0);
                      return (
                        <>
                          <h4 style={{ marginBottom: 'var(--space-2)' }}>目标模糊处理:</h4>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                            <span>处理帧数:</span>
                            <span style={{ fontWeight: 600 }}>{tf}</span>
                          </div>
                          <p style={{ margin: 'var(--space-2) 0 0', fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                            输出为对检测区域做模糊后的视频，无数值统计。
                          </p>
                        </>
                      );
                    }

                    if (selectedSolution === 'workout-monitoring') {
                      const logs = (counting.recent_inference_logs || []) as string[];
                      const lastLine = logs.length ? String(logs[logs.length - 1]) : '';
                      const fromLastLog = parseClassCountsFromInferenceLogLine(lastLine).sort(([a], [b]) =>
                        detectionClassLabelZh(a).localeCompare(detectionClassLabelZh(b), 'zh-CN')
                      );
                      const fallbackClasses = deriveObjectsByClass(counting).sort(([a], [b]) =>
                        detectionClassLabelZh(a).localeCompare(detectionClassLabelZh(b), 'zh-CN')
                      );
                      const classRows = fromLastLog.length > 0 ? fromLastLog : fallbackClasses;
                      const inCountW = Number(counting.in_count || 0);
                      const outCountW = Number(counting.out_count || 0);
                      const netCountW = inCountW - outCountW;
                      return (
                        <>
                          <h4 style={{ marginBottom: 'var(--space-2)' }}>健身监测统计:</h4>
                          {counting.total_frames != null && (
                            <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                              <span>处理帧数:</span>
                              <span style={{ fontWeight: 600 }}>{String(counting.total_frames)}</span>
                            </div>
                          )}
                          {counting.detected_objects != null && (
                            <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                              <span>末帧跟踪目标数:</span>
                              <span style={{ fontWeight: 600 }}>{String(counting.detected_objects)}</span>
                            </div>
                          )}
                          {classRows.length > 0 && (
                            <div style={{ marginTop: 'var(--space-3)', paddingTop: 'var(--space-2)', borderTop: '1px solid var(--border-color)' }}>
                              <div style={{ fontWeight: 600, marginBottom: 'var(--space-2)', fontSize: '0.875rem' }}>
                                {fromLastLog.length > 0
                                  ? '末帧各类检测数量（与下方「推理日志」最底一行英文摘要一致）:'
                                  : '各类数量（后端返回；无日志末行时为此项）:'}
                              </div>
                              {classRows.map(([cls, num]) => (
                                <div key={cls} style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                                  <span>{detectionClassLabelZh(cls)}</span>
                                  <span style={{ fontWeight: 600 }}>{String(num)}</span>
                                </div>
                              ))}
                            </div>
                          )}
                          <p style={{ marginTop: 'var(--space-3)', fontSize: '0.8125rem', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
                            日志为滑动窗口，仅保留末尾若干条；<strong>最底部一行</strong>即处理结束时的该帧检测摘要，与上方「末帧各类检测数量」一致。
                            向上滚动为更早帧，人数/类别会与末帧不同，属正常现象。「末帧跟踪目标数」为轨迹数，与按类相加可能略有差异。
                          </p>
                          <details style={{ marginTop: 'var(--space-2)' }}>
                            <summary style={{ cursor: 'pointer', fontWeight: 600, fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                              跨线进入 / 离开（与健身场景无直接关系，仅供参考）
                            </summary>
                            <div style={{ marginTop: 'var(--space-2)', paddingLeft: 'var(--space-2)' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                                <span>进入:</span>
                                <span style={{ fontWeight: 600 }}>{inCountW}</span>
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                                <span>离开:</span>
                                <span style={{ fontWeight: 600 }}>{outCountW}</span>
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                                <span>净计数:</span>
                                <span style={{ fontWeight: 700, color: netCountW >= 0 ? '#16a34a' : '#dc2626' }}>{netCountW}</span>
                              </div>
                            </div>
                          </details>
                        </>
                      );
                    }

                    const inCount = Number(counting.in_count || 0);
                    const outCount = Number(counting.out_count || 0);
                    const netCount = inCount - outCount;
                    const countingTitle =
                      selectedSolution === 'vision-eye'
                        ? '安防区域统计（与画面区域/跨线一致）:'
                        : '目标计数结果（跨线进出与区域内检测）:';
                    return (
                      <>
                        <h4 style={{ marginBottom: 'var(--space-2)' }}>{countingTitle}</h4>
                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                          <span>进入总数:</span>
                          <span style={{ fontWeight: 600 }}>{inCount}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                          <span>离开总数:</span>
                          <span style={{ fontWeight: 600 }}>{outCount}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                          <span>净计数 (进入-离开):</span>
                          <span style={{ fontWeight: 700, color: netCount >= 0 ? '#16a34a' : '#dc2626' }}>{netCount}</span>
                        </div>
                        {counting.total_frames != null && (
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                            <span>处理帧数:</span>
                            <span style={{ fontWeight: 600 }}>{String(counting.total_frames)}</span>
                          </div>
                        )}
                        {counting.detected_objects != null && (
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                            <span>检测到目标数（静态图/末帧）:</span>
                            <span style={{ fontWeight: 600 }}>{String(counting.detected_objects)}</span>
                          </div>
                        )}
                        {(() => {
                          const appearedClassEntries = deriveObjectsByClass(counting)
                            .sort(([a], [b]) =>
                              detectionClassLabelZh(a).localeCompare(detectionClassLabelZh(b), 'zh-CN')
                            );
                          if (appearedClassEntries.length === 0) return null;
                          return (
                            <div style={{ marginTop: 'var(--space-3)', paddingTop: 'var(--space-2)', borderTop: '1px solid var(--border-color)' }}>
                              <div style={{ fontWeight: 600, marginBottom: 'var(--space-2)', fontSize: '0.875rem' }}>出现过的物体（中文）:</div>
                              {appearedClassEntries.map(([cls, num]) => (
                                <div key={cls} style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                                  <span>{detectionClassLabelZh(cls)}</span>
                                  <span style={{ fontWeight: 600 }}>{String(num)}</span>
                                </div>
                              ))}
                            </div>
                          );
                        })()}
                      </>
                    );
                  })()}
                </div>
              )}
              {['object-counting', 'vision-eye', 'workout-monitoring'].includes(selectedSolution ?? '') &&
              ((result.results?.recent_inference_logs || result.counting?.recent_inference_logs || result?.recent_inference_logs) as string[] | undefined)?.length ? (
                <div style={{ marginTop: 'var(--space-3)', padding: 'var(--space-4)', background: 'var(--gray-50)', borderRadius: 'var(--radius-md)' }}>
                  <h4 style={{ marginBottom: 'var(--space-2)' }}>
                    {selectedSolution === 'workout-monitoring'
                      ? '推理日志（滑动窗口保留最近若干条；最底一行=处理结束时的帧摘要）:'
                      : '推理日志（最近若干帧）:'}
                  </h4>
                  <div
                    ref={selectedSolution === 'workout-monitoring' ? inferenceLogScrollRef : undefined}
                    style={{ maxHeight: '180px', overflowY: 'auto', fontFamily: 'monospace', fontSize: '0.8rem', lineHeight: 1.5 }}
                  >
                    {((result.results?.recent_inference_logs || result.counting?.recent_inference_logs || result?.recent_inference_logs) as string[]).map((line, idx) => (
                      <div key={idx}>{line}</div>
                    ))}
                  </div>
                </div>
              ) : null}
              {/* 距离计算结果 */}
              {selectedSolution === 'distance-calculation' && (result.distances || result.results?.distances) && (
                <div style={{ padding: 'var(--space-4)', background: 'var(--gray-50)', borderRadius: 'var(--radius-md)' }}>
                  <h4 style={{ marginBottom: 'var(--space-2)' }}>距离计算:</h4>
                  {(result.distances || result.results?.distances || []).slice(0, 5).map((item: any, idx: number) => (
                    <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                      <span>目标 {item.object1_index + 1} ↔ 目标 {item.object2_index + 1}:</span>
                      <span style={{ fontWeight: 600 }}>{item.pixel_distance?.toFixed(1) || item.distance?.toFixed(1)} px</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </Card>

        {/* 右侧：参数配置 */}
        <div>
          <Card style={{ marginBottom: 'var(--space-4)' }}>
            <CardHeader icon="⚙️" title="参数配置" />
            <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
              {solution.params.map(param => (
                <div key={param.name}>
                  {param.type === 'checkbox' ? (
                    <label style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', fontSize: '0.875rem', fontWeight: 500, cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={Boolean(params[param.name] ?? param.default)}
                        onChange={e => handleParamChange(param.name, e.target.checked)}
                        style={{ width: '18px', height: '18px', marginTop: '2px', flexShrink: 0 }}
                      />
                      <span>
                        {param.label}
                        {param.help ? <p style={{ ...paramHelpStyle, marginTop: '4px', fontWeight: 400 }}>{param.help}</p> : null}
                      </span>
                    </label>
                  ) : (
                    <>
                      <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: 500, marginBottom: '6px' }}>
                        {param.label}
                      </label>
                      {param.type === 'select' ? (
                        <select
                          value={params[param.name] ?? param.default}
                          onChange={e => handleParamChange(param.name, e.target.value)}
                          className="form-select"
                          style={{ width: '100%' }}
                        >
                          {param.options?.map((opt: any) => (
                            <option key={opt.value || opt} value={opt.value || opt}>
                              {opt.label || opt}
                            </option>
                          ))}
                        </select>
                      ) : param.type === 'number' ? (
                        <input
                          type="number"
                          value={params[param.name] ?? param.default}
                          onChange={e => {
                            const n = parseFloat(e.target.value);
                            handleParamChange(param.name, Number.isNaN(n) ? param.default : n);
                          }}
                          min={(param as Param).min}
                          max={(param as Param).max}
                          step={(param as Param).step}
                          className="form-input"
                          style={{ width: '100%' }}
                        />
                      ) : (
                        <>
                          <div style={{ display: 'flex', gap: '8px', alignItems: 'stretch', flexWrap: 'wrap' }}>
                            <input
                              type="text"
                              value={params[param.name] ?? ''}
                              onChange={e => handleParamChange(param.name, e.target.value)}
                              placeholder={(param as Param).placeholder}
                              className="form-input"
                              style={{ flex: '1 1 200px', minWidth: 0 }}
                            />
                            {selectedSolution === 'object-counting' && param.name === 'region_points' ? (
                              <Button
                                type="button"
                                variant="secondary"
                                size="sm"
                                onClick={() => handleParamChange('region_points', '')}
                              >
                                默认区域
                              </Button>
                            ) : null}
                            {selectedSolution === 'parking-management' && param.name === 'parking_slots' ? (
                              <Button
                                type="button"
                                variant="secondary"
                                size="sm"
                                onClick={() => handleParamChange('parking_slots', '')}
                              >
                                示例车位
                              </Button>
                            ) : null}
                          </div>
                          {param.name === 'region_points' && regionJsonInvalid(String(params[param.name] ?? '')) ? (
                            <p style={{ ...paramHelpStyle, color: '#b91c1c' }}>JSON 无法解析，请检查括号与逗号。</p>
                          ) : null}
                          {param.name === 'parking_slots' && regionJsonInvalid(String(params[param.name] ?? '')) ? (
                            <p style={{ ...paramHelpStyle, color: '#b91c1c' }}>JSON 无法解析，应为车位数组。</p>
                          ) : null}
                        </>
                      )}
                      {param.help && param.type !== 'checkbox' ? <p style={paramHelpStyle}>{param.help}</p> : null}
                    </>
                  )}
                </div>
              ))}
            </div>
          </Card>

          <Button
            variant="primary"
            style={{ width: '100%' }}
            disabled={!file || loading}
            onClick={handleProcess}
          >
            {loading ? '⏳ 处理中...' : `🚀 开始处理`}
          </Button>

          {result && (
            <Button
              variant="secondary"
              style={{ width: '100%', marginTop: 'var(--space-3)' }}
              onClick={() => {
                // 下载结果
                if (result.result_image) {
                  const link = document.createElement('a');
                  link.href = result.result_image;
                  link.download = 'result.jpg';
                  link.click();
                }
              }}
            >
              📥 下载结果
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};

export default SolutionRunner;
