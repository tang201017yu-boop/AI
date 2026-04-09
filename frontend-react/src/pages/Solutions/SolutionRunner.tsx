import React, { useState, useEffect } from 'react';
import { Card, CardHeader, Button } from '../../components/common';
import { solutionsApi, modelApi, inferenceApi } from '../../services/api';

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
    icon: '📊',
    color: '#3b82f6',
    supportsVideo: true,
    params: [
      { name: 'model_name', label: '检测模型', type: 'select', default: 'yolo26n.pt', options: DETECTION_MODELS },
      { name: 'conf', label: '置信度阈值', type: 'number', default: 0.25, min: 0, max: 1, step: 0.05 },
      { name: 'region_type', label: '区域类型', type: 'select', default: 'polygon',
        options: [
          { value: 'polygon', label: '多边形区域' },
          { value: 'line', label: '直线(进出计数)' },
        ]
      },
      { name: 'region_points', label: '区域坐标', type: 'text', placeholder: '如: [(20,400),(1260,400),(1260,360),(20,360)]' },
      { name: 'show_in', label: '显示进入计数', type: 'checkbox', default: true },
      { name: 'show_out', label: '显示离开计数', type: 'checkbox', default: true },
      { name: 'line_width', label: '线条宽度', type: 'number', default: 2, min: 1, max: 10 },
    ]
  },
  'heatmap': {
    name: 'heatmap',
    title: '热力图生成',
    description: '生成目标检测密度热力图，可视化热点区域',
    icon: '🔥',
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
    icon: '🚗',
    color: '#f97316',
    supportsVideo: true,
    params: [
      { name: 'model_name', label: '检测模型', type: 'select', default: 'yolo26n.pt', options: DETECTION_MODELS },
      { name: 'conf', label: '置信度阈值', type: 'number', default: 0.25, min: 0, max: 1, step: 0.05 },
      { name: 'pixel_to_meter', label: '像素/米比例', type: 'number', default: 10, min: 0.1, step: 0.1 },
      { name: 'line_width', label: '线条宽度', type: 'number', default: 2, min: 1, max: 10 },
    ]
  },
  'distance-calculation': {
    name: 'distance-calculation',
    title: '距离计算',
    description: '计算图像中检测目标之间的距离',
    icon: '📏',
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
    icon: '🔒',
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
    icon: '✂️',
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
    icon: '👥',
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
    icon: '🅿️',
    color: '#eab308',
    supportsVideo: true,
    params: [
      { name: 'model_name', label: '检测模型', type: 'select', default: 'yolo26n.pt', options: DETECTION_MODELS },
      { name: 'conf', label: '置信度阈值', type: 'number', default: 0.25, min: 0, max: 1, step: 0.05 },
      { name: 'line_width', label: '线条宽度', type: 'number', default: 2, min: 1, max: 10 },
      { name: 'parking_slots', label: '车位坐标(JSON)', type: 'text', placeholder: '如: [[[80,420],[260,420],[260,600],[80,600]],[[280,420],[460,420],[460,600],[280,600]]]' },
    ]
  },
  'vision-eye': {
    name: 'vision-eye',
    title: '视觉安防',
    description: '周界入侵检测，异常行为识别',
    icon: '👁️',
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
    icon: '🏋️',
    color: '#14b8a6',
    supportsVideo: true,
    params: [
      { name: 'model_name', label: '检测模型', type: 'select', default: 'yolo11n.pt', options: DETECTION_MODELS },
      { name: 'conf', label: '置信度阈值', type: 'number', default: 0.25, min: 0, max: 1, step: 0.05 },
    ]
  },
};

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
}

interface SolutionConfig {
  name: string;
  title: string;
  description: string;
  icon: string;
  color: string;
  supportsVideo: boolean;
  params: Param[];
}

export const SolutionRunner: React.FC = () => {
  const [selectedSolution, setSelectedSolution] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [params, setParams] = useState<Record<string, any>>({});

  const solution = selectedSolution ? SOLUTIONS[selectedSolution as keyof typeof SOLUTIONS] : null;

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
    }
  };

  const handleParamChange = (name: string, value: any) => {
    setParams(prev => ({ ...prev, [name]: value }));
  };

  const handleProcess = async () => {
    if (!file || !selectedSolution) return;

    setLoading(true);
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
          const data: any = res.data?.data || res.data;
          if (data?.output_path && !data?.output_image && !data?.result_image) {
            data.result_image = data.output_path;
          }
          setResult(data);
          return;
        }
        const modelName = String(params.model_name || 'yolo11n.pt');
        const conf = Number(params.conf ?? 0.25);
        const res = await inferenceApi.image(file, modelName, conf);
        const data: any = res.data?.data || res.data;
        // 统一结果字段，复用当前页面展示逻辑
        if (data?.annotated_image && !data?.output_image) {
          data.output_image = data.annotated_image;
        }
        setResult(data);
        return;
      }

      const formData = new FormData();
      formData.append('file', file);

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
      const data = res.data?.data || res.data;
      setResult(data);
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
                fontSize: '24px',
                marginBottom: 'var(--space-3)'
              }}>
                {sol.icon}
              </div>
              <h3 style={{ fontSize: '1.125rem', fontWeight: 600, marginBottom: 'var(--space-1)' }}>
                {sol.title}
              </h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: 'var(--space-3)' }}>
                {sol.description}
              </p>
              <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                <span style={{ fontSize: '0.75rem', background: 'var(--gray-100)', padding: '2px 8px', borderRadius: '4px' }}>
                  🖼️ 图片
                </span>
                {sol.supportsVideo && (
                  <span style={{ fontSize: '0.75rem', background: 'var(--gray-100)', padding: '2px 8px', borderRadius: '4px' }}>
                    🎬 视频
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
    const candidate =
      data.result_image ||
      data.output_path ||
      data.output_image ||
      data.annotated_image ||
      data?.results?.result_image ||
      data?.results?.output_path ||
      data?.results?.output_image ||
      data?.results?.annotated_image ||
      '';
    if (!candidate) return '';
    if (String(candidate).startsWith('data:')) return String(candidate);
    if (String(candidate).startsWith('http')) return String(candidate);
    return window.location.origin + String(candidate);
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
          fontSize: '20px',
        }}>
          {solution.icon}
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
          <CardHeader icon={solution.icon} title="输入文件" />
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
              <CardHeader icon="📊" title="处理结果" />
              {/* 处理图片/视频结果 */}
              {(() => {
                const mediaPath = resolveMediaPath(result);
                if (!mediaPath) return null;
                const lower = mediaPath.toLowerCase();
                const isVideo = lower.endsWith('.mp4') || lower.endsWith('.avi') || lower.endsWith('.mov') || lower.endsWith('.mkv') || lower.endsWith('.webm');
                return isVideo ? (
                  <div>
                    <video
                      src={mediaPath + (mediaPath.includes('?') ? '&' : '?') + 't=' + Date.now()}
                      controls
                      playsInline
                      style={{ width: '100%', borderRadius: 'var(--radius-md)', marginBottom: 'var(--space-3)', backgroundColor: '#000' }}
                    />
                    <a href={mediaPath + (mediaPath.includes('?') ? '&' : '?') + 'download=1'} download style={{ display: 'inline-block', marginTop: '8px', color: '#3b82f6', textDecoration: 'none' }}>
                      下载视频
                    </a>
                  </div>
                ) : (
                  <img
                    src={mediaPath}
                    alt="Result"
                    style={{ width: '100%', borderRadius: 'var(--radius-md)', marginBottom: 'var(--space-3)' }}
                  />
                );
              })()}

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
              {/* 处理计数结果 - 支持 results 或 counting */}
              {(result.results || result.counting) && (
                <div style={{ padding: 'var(--space-4)', background: 'var(--gray-50)', borderRadius: 'var(--radius-md)' }}>
                  <h4 style={{ marginBottom: 'var(--space-2)' }}>计数结果:</h4>
                  {Object.entries(result.results || result.counting).map(([key, value]: [string, any]) => (
                    <div key={key} style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                      <span>{key === 'in_count' ? '进入数量' : key === 'out_count' ? '离开数量' : key === 'total_frames' ? '总帧数' : key}:</span>
                      <span style={{ fontWeight: 600 }}>{String(value)}</span>
                    </div>
                  ))}
                </div>
              )}
              {/* 速度估算结果 */}
              {(result.speeds || result.results?.speeds) && (
                <div style={{ padding: 'var(--space-4)', background: 'var(--gray-50)', borderRadius: 'var(--radius-md)' }}>
                  <h4 style={{ marginBottom: 'var(--space-2)' }}>速度估算:</h4>
                  {(() => {
                    const speeds = result.speeds || result.results?.speeds || [];
                    // 过滤有效的速度数据
                    const validSpeeds = speeds.filter((s: any) => s && (s.track > 0 || s.solution > 0));
                    // 取最后一个有效速度（通常是最终速度）
                    const finalSpeed = validSpeeds.length > 0 ? validSpeeds[validSpeeds.length - 1] : null;
                    if (finalSpeed) {
                      return (
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                            <span>平均速度 (track):</span>
                            <span style={{ fontWeight: 600 }}>{(finalSpeed.track || 0).toFixed(2)} px/frame</span>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-1) 0' }}>
                            <span>处理速度:</span>
                            <span style={{ fontWeight: 600 }}>{(finalSpeed.solution || 0).toFixed(2)} ms/帧</span>
                          </div>
                        </div>
                      );
                    }
                    return <div>未检测到移动目标</div>;
                  })()}
                </div>
              )}
              {/* 距离计算结果 */}
              {(result.distances || result.results?.distances) && (
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
              {/* 队列管理结果 */}
              {(result.queue_info || result.results) && (
                <div style={{ padding: 'var(--space-4)', background: 'var(--gray-50)', borderRadius: 'var(--radius-md)' }}>
                  <h4 style={{ marginBottom: 'var(--space-2)' }}>队列信息:</h4>
                  <div style={{ display: 'grid', gap: 'var(--space-2)' }}>
                    {(result.queue_info || result.results || {})?.frame_counts?.[0] && (
                      <>
                        <div>当前人数: <strong>{(result.queue_info || result.results || {}).frame_counts?.[0] || 0}</strong></div>
                        <div>平均队列: <strong>{((result.queue_info || result.results || {}).avg_queue_count || 0).toFixed(1)}</strong></div>
                        <div>总帧数: <strong>{(result.queue_info || result.results || {}).total_frames || 0}</strong></div>
                      </>
                    )}
                  </div>
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
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.875rem', fontWeight: 500, cursor: 'pointer' }}>
                    {param.type === 'checkbox' ? (
                      <input
                        type="checkbox"
                        checked={params[param.name] ?? param.default}
                        onChange={e => handleParamChange(param.name, e.target.checked)}
                        style={{ width: '18px', height: '18px' }}
                      />
                    ) : (
                      param.label
                    )}
                  </label>
                  {param.type === 'checkbox' ? null : param.type === 'select' ? (
                    <select
                      value={params[param.name] || param.default}
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
                      value={params[param.name] || param.default}
                      onChange={e => handleParamChange(param.name, parseFloat(e.target.value))}
                      min={(param as any).min}
                      max={(param as any).max}
                      step={(param as any).step}
                      className="form-input"
                      style={{ width: '100%' }}
                    />
                  ) : (
                    <input
                      type="text"
                      value={params[param.name] || ''}
                      onChange={e => handleParamChange(param.name, e.target.value)}
                      placeholder={(param as any).placeholder}
                      className="form-input"
                      style={{ width: '100%' }}
                    />
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
