import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Card, Button, Input } from '../../components/common';
import { Upload } from '../../components/features';
import { inferenceApi, modelApi } from '../../services/api';
import styles from './Inference.module.css';

interface TrainedModel {
  name: string;
  path: string;
  task?: string;
  project?: string;
}

export const Inference: React.FC = () => {
  const [searchParams] = useSearchParams();
  const [selectedModel, setSelectedModel] = useState('');
  const [trainedModels, setTrainedModels] = useState<TrainedModel[]>([]);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [showAnnotated, setShowAnnotated] = useState(true);

  useEffect(() => {
    loadTrainedModels();

    // 从 URL 参数加载模型
    const modelParam = searchParams.get('model');
    if (modelParam) {
      setSelectedModel(decodeURIComponent(modelParam));
    }
  }, [searchParams]);

  const loadTrainedModels = async () => {
    try {
      const res = await modelApi.getUserModels();
      const responseData = res.data as any;
      const allModels = responseData?.models || [];
      const userModels = allModels.filter((m: any) => m.source === 'uploaded' || m.source === 'training' || m.source === 'project_model');
      setTrainedModels(userModels);
    } catch (error) {
      console.error('加载训练模型失败:', error);
    }
  };

  const handleFileSelect = (files: File[]) => {
    if (files[0]) {
      setImageFile(files[0]);
      setImagePreview(URL.createObjectURL(files[0]));
      setResult(null);
    }
  };

  const handleInference = async () => {
    if (!selectedModel || !imageFile) return;
    setLoading(true);
    try {
      const res = await inferenceApi.image(
        imageFile,
        selectedModel,
        confThreshold
      );
      setResult(res.data?.data || res.data);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // 按类别分组统计
  const getClassStats = () => {
    if (!result?.detections) return {};
    const stats: Record<string, { count: number; items: any[] }> = {};
    result.detections.forEach((det: any) => {
      const name = det.class_name;
      if (!stats[name]) {
        stats[name] = { count: 0, items: [] };
      }
      stats[name].count++;
      stats[name].items.push(det);
    });
    return stats;
  };

  const classStats = result ? getClassStats() : {};

  return (
    <div className={styles.container}>
      {/* 左侧：图片上传和显示 */}
      <div className={styles.leftPanel}>
        <div className={styles.uploadSection}>
          <h2 className={styles.panelTitle}>📤 上传图片</h2>
          <Upload
            accept="image/*"
            onFilesSelected={handleFileSelect}
            hint="点击或拖拽图片到此处"
          />
        </div>

        {imagePreview && (
          <div className={styles.imageSection}>
            <div className={styles.imageTabs}>
              <button
                className={`${styles.tab} ${!showAnnotated ? styles.activeTab : ''}`}
                onClick={() => setShowAnnotated(false)}
              >
                📷 原图
              </button>
              <button
                className={`${styles.tab} ${showAnnotated && result ? styles.activeTab : ''}`}
                onClick={() => setShowAnnotated(true)}
                disabled={!result}
              >
                🎯 标注结果
              </button>
            </div>
            <div className={styles.imageContainer}>
              {showAnnotated && result?.annotated_image ? (
                <img
                  src={result.annotated_image}
                  alt="Annotated Result"
                  className={styles.resultImage}
                />
              ) : (
                <img
                  src={imagePreview}
                  alt="Preview"
                  className={styles.resultImage}
                />
              )}
            </div>
          </div>
        )}
      </div>

      {/* 右侧：配置和结果 */}
      <div className={styles.rightPanel}>
        <div className={styles.configPanel}>
          <h2 className={styles.panelTitle}>⚙️ 推理配置</h2>
          <div className={styles.form}>
            <div>
              <label style={{ display: 'block', marginBottom: 'var(--space-1)', fontWeight: 500 }}>
                选择模型
              </label>
              <select
                className={styles.select}
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                <option value="">请选择模型</option>
                {trainedModels.length > 0 && (
                  <optgroup label="── 用户模型 ──">
                    {trainedModels.map((model, idx) => (
                      <option key={`trained-${idx}`} value={model.path}>
                        {model.project ? `[${model.project}] ${model.name.includes('best') ? '最优权重' : model.name.includes('last') ? '最终权重' : model.name}` : model.name}
                      </option>
                    ))}
                  </optgroup>
                )}
                <optgroup label="── 系统模型 ──">
                  <option value="yolo26n.pt">YOLO26n - 最新最快</option>
                  <option value="yolo26s.pt">YOLO26s - 轻量快速</option>
                  <option value="yolo26m.pt">YOLO26m - 平衡推荐</option>
                  <option value="yolo26l.pt">YOLO26l - 高精度</option>
                  <option value="yolo26x.pt">YOLO26x - 最高精度</option>
                  <option value="yolo11n.pt">YOLO11n</option>
                  <option value="yolo11s.pt">YOLO11s</option>
                  <option value="yolo11m.pt">YOLO11m</option>
                </optgroup>
              </select>
            </div>
            <Input
              type="number"
              label="置信度阈值"
              value={confThreshold}
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              min={0}
              max={1}
              step={0.05}
            />
            <Button
              variant="primary"
              onClick={handleInference}
              loading={loading}
              disabled={!selectedModel || !imageFile}
            >
              开始推理
            </Button>
          </div>
        </div>

        {/* 推理结果 */}
        {result && (
          <div className={styles.resultPanel}>
            <div className={styles.resultHeader}>
              <h3 className={styles.resultTitle}>🎯 推理结果</h3>
              <div className={styles.resultMeta}>
                <span className={styles.badge}>
                  ⏱️ {result.inference_time?.toFixed(2)}s
                </span>
                <span className={styles.badge}>
                  📐 {result.image_shape?.[1]}×{result.image_shape?.[0]}
                </span>
              </div>
            </div>

            {/* 统计摘要 */}
            <div className={styles.summary}>
              <div className={styles.summaryItem}>
                <span className={styles.summaryValue}>{result.detections?.length || 0}</span>
                <span className={styles.summaryLabel}>检测对象</span>
              </div>
              <div className={styles.summaryItem}>
                <span className={styles.summaryValue}>{Object.keys(classStats).length}</span>
                <span className={styles.summaryLabel}>类别数</span>
              </div>
            </div>

            {/* 按类别分组的结果 */}
            <div className={styles.detectionResults}>
              <h4 className={styles.sectionTitle}>📊 类别详情</h4>
              {Object.entries(classStats).map(([className, data]: [string, any]) => (
                <div key={className} className={styles.classGroup}>
                  <div className={styles.classHeader}>
                    <span className={styles.className}>{className}</span>
                    <span className={styles.classCount}>× {data.count}</span>
                  </div>
                  <div className={styles.classItems}>
                    {data.items.map((item: any, idx: number) => (
                      <div key={idx} className={styles.detectionItem}>
                        <span className={styles.detectionIndex}>#{idx + 1}</span>
                        <span className={styles.detectionBbox}>
                          [{item.bbox.map((v: number) => Math.round(v)).join(', ')}]
                        </span>
                        <span className={styles.detectionConf}>
                          {(item.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
