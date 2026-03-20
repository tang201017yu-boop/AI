import React, { useEffect, useState, useCallback, useRef } from 'react';
import { Card, CardHeader, Button, StatCard } from '../../components/common';
import { datasetApi } from '../../services/api';
import type { Dataset } from '../../types';

interface DatasetStats {
  summary: {
    total_images: number;
    total_annotations: number;
    total_classes: number;
    classes: string[];
    total_size_mb: number;
    avg_annotations_per_image: number;
  };
  class_distribution: {
    labels: string[];
    counts: number[];
    percentages: number[];
  };
  split_details: {
    splits: {
      train: { images: number; annotations: number; percentage: number };
      val: { images: number; annotations: number; percentage: number };
      test: { images: number; annotations: number; percentage: number };
    };
  };
}

interface ImageItem {
  filename: string;
  path: string;
  thumbnail: string;
  width: number;
  height: number;
  size: number;
  modified: number;
  split: string;
  label_count: number;
  labels: { class_id: number; bbox: number[] }[];
}

type ViewMode = 'grid' | 'compact' | 'table';
type SortOption = 'name_asc' | 'name_desc' | 'date_new' | 'date_old' | 'size_asc' | 'size_desc' | 'labels_asc' | 'labels_desc';

export const Datasets: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [imagesLoading, setImagesLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'images' | 'statistics'>('overview');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalImages, setTotalImages] = useState(0);

  // 视图和筛选状态
  const [viewMode, setViewMode] = useState<ViewMode>('table');
  const [splitFilter, setSplitFilter] = useState<string>('');
  const [labeledFilter, setLabeledFilter] = useState<string>('');
  const [sortBy, setSortBy] = useState<SortOption>('name_asc');

  // 全屏查看器状态
  const [fullscreenImage, setFullscreenImage] = useState<ImageItem | null>(null);
  const [showLabels, setShowLabels] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [pixelView, setPixelView] = useState(false);

  // 导出弹窗状态
  const [showExportModal, setShowExportModal] = useState(false);
  const [exporting, setExporting] = useState(false);

  // 文件输入引用
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      loadDatasetStats(selectedDataset);
      loadDatasetImages(selectedDataset);
    }
  }, [selectedDataset, currentPage, splitFilter, labeledFilter, sortBy]);

  const loadDatasets = async () => {
    try {
      const res = await datasetApi.list();
      const datasetsData = res.data?.datasets || res.data?.data?.datasets || [];
      setDatasets(datasetsData);
      if (datasetsData.length > 0 && !selectedDataset) {
        setSelectedDataset(datasetsData[0].name);
      }
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const loadDatasetStats = async (name: string) => {
    setStatsLoading(true);
    try {
      const res = await datasetApi.getStatistics(name);
      setStats(res.data || res.data?.data);
    } catch (error) {
      console.error(error);
    } finally {
      setStatsLoading(false);
    }
  };

  const loadDatasetImages = async (name: string) => {
    setImagesLoading(true);
    try {
      const params: any = {
        page: currentPage,
        page_size: viewMode === 'compact' ? 100 : 50,
        sort: sortBy,
      };
      if (splitFilter) params.split = splitFilter;
      if (labeledFilter) params.labeled = labeledFilter;

      const res = await datasetApi.getImages(name, params);
      setImages(res.data?.images || []);
      setTotalImages(res.data?.total || 0);
    } catch (error) {
      console.error(error);
    } finally {
      setImagesLoading(false);
    }
  };

  const handleUpload = async (files: File[]) => {
    if (!files[0]) return;
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', files[0]);
      await datasetApi.upload(formData);
      await loadDatasets();
    } catch (error) {
      console.error(error);
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDataset = async () => {
    if (!selectedDataset) return;
    if (!confirm(`确定要删除数据集 ${selectedDataset} 吗？此操作不可恢复。`)) return;
    try {
      await datasetApi.delete(selectedDataset);
      const res = await datasetApi.list();
      const datasetsData = res.data?.datasets || res.data?.data?.datasets || [];
      setDatasets(datasetsData);
      setSelectedDataset(datasetsData.length > 0 ? datasetsData[0].name : null);
    } catch (error) {
      console.error(error);
    }
  };

  const handleExport = async (format: string) => {
    if (!selectedDataset) return;
    setExporting(true);
    try {
      const response = await fetch(
        `/api/v1/datasets/${selectedDataset}/export?format=${format}&splits=train,val,test`
      );
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedDataset}_${format}.zip`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      setShowExportModal(false);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setExporting(false);
    }
  };

  const handleDeleteImage = async (filename: string) => {
    if (!selectedDataset || !confirm(`确定要删除 ${filename} 吗？`)) return;
    try {
      // 这里需要添加删除API
      alert('删除功能开发中');
    } catch (error) {
      console.error(error);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const getClassCount = (labels: { class_id: number }[]) => {
    const counts: Record<number, number> = {};
    labels.forEach(l => {
      counts[l.class_id] = (counts[l.class_id] || 0) + 1;
    });
    return counts;
  };

  // 键盘导航
  useEffect(() => {
    if (!fullscreenImage) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const currentIndex = images.findIndex(img => img.filename === fullscreenImage.filename);
      if (e.key === 'ArrowLeft' && currentIndex > 0) {
        setFullscreenImage(images[currentIndex - 1]);
      } else if (e.key === 'ArrowRight' && currentIndex < images.length - 1) {
        setFullscreenImage(images[currentIndex + 1]);
      } else if (e.key === 'Escape') {
        setFullscreenImage(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [fullscreenImage, images]);

  const renderOverview = () => {
    if (statsLoading) return <p>加载中...</p>;
    if (!stats) return <p>暂无数据</p>;

    const { summary, class_distribution, split_details } = stats;

    return (
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
          <StatCard value={summary.total_images} label="总图片数" />
          <StatCard value={summary.total_annotations} label="总标注数" variant="accent" />
          <StatCard value={summary.total_classes} label="类别数" variant="success" />
          <StatCard value={`${summary.total_size_mb.toFixed(1)} MB`} label="数据集大小" />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-6)' }}>
          <Card>
            <CardHeader icon="📊" title="类别分布" />
            {class_distribution && class_distribution.labels.length > 0 ? (
              <div>
                {class_distribution.labels.slice(0, 10).map((label, idx) => (
                  <div key={idx} style={{ display: 'flex', alignItems: 'center', marginBottom: 'var(--space-2)' }}>
                    <div style={{ width: '100px', fontSize: '0.875rem' }}>{label}</div>
                    <div style={{ flex: 1, height: '20px', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${class_distribution.percentages[idx]}%`,
                          height: '100%',
                          background: 'var(--primary-500)',
                          transition: 'width 0.3s'
                        }}
                      />
                    </div>
                    <div style={{ width: '60px', textAlign: 'right', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                      {class_distribution.counts[idx]} ({class_distribution.percentages[idx].toFixed(1)}%)
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: 'var(--text-secondary)' }}>暂无类别数据</p>
            )}
          </Card>

          <Card>
            <CardHeader icon="📁" title="数据集拆分" />
            {split_details && split_details.splits ? (
              <div>
                {Object.entries(split_details.splits).map(([split, data]: [string, any]) => (
                  <div key={split} style={{ marginBottom: 'var(--space-4)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-1)' }}>
                      <span style={{ fontWeight: 500 }}>{split === 'train' ? '训练集' : split === 'val' ? '验证集' : '测试集'}</span>
                      <span style={{ color: 'var(--text-secondary)' }}>{data.images} 张图片 ({data.percentage.toFixed(1)}%)</span>
                    </div>
                    <div style={{ height: '8px', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${data.percentage}%`,
                          height: '100%',
                          background: split === 'train' ? 'var(--primary-500)' : split === 'val' ? 'var(--accent-500)' : 'var(--success-500)',
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: 'var(--text-secondary)' }}>暂无拆分数据</p>
            )}
          </Card>
        </div>

        {summary.classes && summary.classes.length > 0 && (
          <Card style={{ marginTop: 'var(--space-6)' }}>
            <CardHeader icon="🏷️" title="类别列表" />
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-2)' }}>
              {summary.classes.map((cls, idx) => (
                <span key={idx} style={{
                  padding: 'var(--space-1) var(--space-3)',
                  background: 'var(--primary-100)',
                  color: 'var(--primary-700)',
                  borderRadius: 'var(--radius-md)',
                  fontSize: '0.875rem'
                }}>
                  {cls}
                </span>
              ))}
            </div>
          </Card>
        )}
      </div>
    );
  };

  const renderImageGrid = () => {
    const gridClass = viewMode === 'grid'
      ? { gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 'var(--space-3)' }
      : viewMode === 'compact'
        ? { gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))', gap: 'var(--space-2)' }
        : {};

    const imageStyle = viewMode === 'grid'
      ? { height: '140px' }
      : viewMode === 'compact'
        ? { height: '80px' }
        : {};

    return (
      <div style={gridClass as any}>
        {images.map((img, idx) => (
          <div
            key={idx}
            onClick={() => setFullscreenImage(img)}
            style={{
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)',
              overflow: 'hidden',
              cursor: 'pointer',
              transition: 'transform 0.2s, box-shadow 0.2s',
              background: 'var(--bg-card)'
            }}
          >
            <div style={{ position: 'relative' }}>
              <img
                src={img.thumbnail}
                alt={img.filename}
                style={{ width: '100%', height: imageStyle.height, objectFit: 'cover', display: 'block', imageRendering: pixelView ? 'pixelated' : 'auto' }}
                onError={(e) => {
                  (e.target as HTMLImageElement).src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect fill="%23eee" width="100" height="100"/><text x="50" y="50" text-anchor="middle" dy=".3em" fill="%23999">无图片</text></svg>';
                }}
              />
              {img.label_count > 0 && (
                <span style={{
                  position: 'absolute', top: 4, right: 4,
                  background: 'rgba(0,0,0,0.6)', color: 'white',
                  padding: '2px 6px', borderRadius: '4px', fontSize: '0.7rem'
                }}>
                  {img.label_count}
                </span>
              )}
            </div>
            {viewMode !== 'compact' && (
              <div style={{ padding: 'var(--space-2)', fontSize: '0.75rem' }}>
                <p style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{img.filename}</p>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)' }}>
                  <span>{img.width}x{img.height}</span>
                  <span>{img.split}</span>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  const renderImageTable = () => (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
      <thead>
        <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
          <th style={{ padding: 'var(--space-2)' }}>缩略图</th>
          <th style={{ padding: 'var(--space-2)' }}>文件名</th>
          <th style={{ padding: 'var(--space-2)' }}>尺寸</th>
          <th style={{ padding: 'var(--space-2)' }}>大小</th>
          <th style={{ padding: 'var(--space-2)' }}>拆分</th>
          <th style={{ padding: 'var(--space-2)' }}>标注数</th>
        </tr>
      </thead>
      <tbody>
        {images.map((img, idx) => (
          <tr
            key={idx}
            onClick={() => setFullscreenImage(img)}
            style={{ borderBottom: '1px solid var(--border)', cursor: 'pointer', ':hover': { background: 'var(--bg-hover)' } }}
          >
            <td style={{ padding: 'var(--space-2)' }}>
              <img
                src={img.thumbnail}
                alt={img.filename}
                style={{ width: '60px', height: '40px', objectFit: 'cover', borderRadius: '4px' }}
                onError={(e) => {
                  (e.target as HTMLImageElement).src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect fill="%23eee" width="100" height="100"/><text x="50" y="50" text-anchor="middle" dy=".3em" fill="%23999">无</text></svg>';
                }}
              />
            </td>
            <td style={{ padding: 'var(--space-2)' }}>{img.filename}</td>
            <td style={{ padding: 'var(--space-2)' }}>{img.width}x{img.height}</td>
            <td style={{ padding: 'var(--space-2)' }}>{formatFileSize(img.size)}</td>
            <td style={{ padding: 'var(--space-2)' }}>
              <span style={{
                padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem',
                background: img.split === 'train' ? 'var(--primary-100)' : img.split === 'val' ? 'var(--accent-100)' : 'var(--success-100)',
                color: img.split === 'train' ? 'var(--primary-700)' : img.split === 'val' ? 'var(--accent-700)' : 'var(--success-700)'
              }}>
                {img.split}
              </span>
            </td>
            <td style={{ padding: 'var(--space-2)' }}>
              {img.label_count > 0 ? (
                <span style={{ color: 'var(--success-600)', fontWeight: 500 }}>{img.label_count}</span>
              ) : (
                <span style={{ color: 'var(--text-secondary)' }}>无</span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );

  const renderImages = () => (
    <div>
      {/* 工具栏 */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)', flexWrap: 'wrap', gap: 'var(--space-3)' }}>
        {/* 左侧筛选 */}
        <div style={{ display: 'flex', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
          <select
            value={splitFilter}
            onChange={(e) => { setSplitFilter(e.target.value); setCurrentPage(1); }}
            style={{ padding: 'var(--space-2)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}
          >
            <option value="">全部拆分</option>
            <option value="train">训练集</option>
            <option value="val">验证集</option>
            <option value="test">测试集</option>
          </select>

          <select
            value={labeledFilter}
            onChange={(e) => { setLabeledFilter(e.target.value); setCurrentPage(1); }}
            style={{ padding: 'var(--space-2)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}
          >
            <option value="">全部图片</option>
            <option value="labeled">有标注</option>
            <option value="unlabeled">无标注</option>
          </select>

          <select
            value={sortBy}
            onChange={(e) => { setSortBy(e.target.value as SortOption); setCurrentPage(1); }}
            style={{ padding: 'var(--space-2)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}
          >
            <option value="name_asc">名称 A-Z</option>
            <option value="name_desc">名称 Z-A</option>
            <option value="date_new">最新添加</option>
            <option value="date_old">最早添加</option>
            <option value="size_asc">尺寸从小</option>
            <option value="size_desc">尺寸从大</option>
            <option value="labels_desc">标注最多</option>
            <option value="labels_asc">标注最少</option>
          </select>
        </div>

        {/* 右侧导出 */}
        <div style={{ display: 'flex', gap: 'var(--space-2)', alignItems: 'center' }}>
          <Button variant="secondary" size="sm" onClick={() => setShowExportModal(true)}>
            📥 导出
          </Button>
        </div>
      </div>

      {/* 图片统计 */}
      <div style={{ marginBottom: 'var(--space-3)', color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
        共 {totalImages} 张图片，当前显示 {images.length} 张
      </div>

      {imagesLoading ? (
        <p>加载中...</p>
      ) : images.length === 0 ? (
        <p style={{ color: 'var(--text-secondary)' }}>暂无图片</p>
      ) : (
        <>
          {viewMode === 'table' ? renderImageTable() : renderImageGrid()}

          {/* 分页 */}
          {totalImages > (viewMode === 'compact' ? 100 : 50) && (
            <div style={{ display: 'flex', justifyContent: 'center', gap: 'var(--space-2)', marginTop: 'var(--space-6)' }}>
              <Button
                variant="secondary"
                size="sm"
                disabled={currentPage === 1}
                onClick={() => setCurrentPage(p => p - 1)}
              >
                上一页
              </Button>
              <span style={{ display: 'flex', alignItems: 'center', padding: '0 var(--space-3)' }}>
                第 {currentPage} 页
              </span>
              <Button
                variant="secondary"
                size="sm"
                disabled={images.length < (viewMode === 'compact' ? 100 : 50)}
                onClick={() => setCurrentPage(p => p + 1)}
              >
                下一页
              </Button>
            </div>
          )}
        </>
      )}
    </div>
  );

  // 全屏查看器
  const renderFullscreenViewer = () => {
    if (!fullscreenImage) return null;

    const classCounts = getClassCount(fullscreenImage.labels);

    return (
      <div
        style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0,0,0,0.9)', zIndex: 1000,
          display: 'flex', flexDirection: 'column'
        }}
      >
        {/* 顶部栏 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: 'var(--space-3)', background: 'rgba(0,0,0,0.5)' }}>
          <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
            <Button variant="secondary" size="sm" onClick={() => {
              const idx = images.findIndex(i => i.filename === fullscreenImage.filename);
              if (idx > 0) setFullscreenImage(images[idx - 1]);
            }}>← 上一张</Button>
            <Button variant="secondary" size="sm" onClick={() => {
              const idx = images.findIndex(i => i.filename === fullscreenImage.filename);
              if (idx < images.length - 1) setFullscreenImage(images[idx + 1]);
            }}>下一张 →</Button>
          </div>
          <span style={{ color: 'white', fontSize: '1rem' }}>{fullscreenImage.filename}</span>
          <Button variant="secondary" size="sm" onClick={() => setFullscreenImage(null)}>✕ 关闭</Button>
        </div>

        {/* 图片区域 */}
        <div
          style={{ flex: 1, overflow: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onWheel={(e) => {
            if (e.ctrlKey || e.metaKey) {
              e.preventDefault();
              setZoom(z => Math.max(0.5, Math.min(3, z + (e.deltaY > 0 ? -0.1 : 0.1))));
            }
          }}
        >
          <div style={{ position: 'relative', display: 'inline-block', transform: `scale(${zoom})`, transformOrigin: 'center', transition: 'transform 0.1s' }}>
            <img
              src={fullscreenImage.path}
              alt={fullscreenImage.filename}
              style={{
                display: 'block',
                maxWidth: '80vw',
                maxHeight: '72vh',
                imageRendering: pixelView ? 'pixelated' : 'auto',
              }}
            />
            {/* 标注叠加层 */}
            {showLabels && fullscreenImage.labels.length > 0 && fullscreenImage.labels.map((label, idx) => {
              const bbox = label.bbox;
              if (bbox.length < 4) return null;
              const [x, y, w, h] = bbox;
              return (
                <div
                  key={idx}
                  style={{
                    position: 'absolute',
                    left: `${(x - w/2) * 100}%`,
                    top: `${(y - h/2) * 100}%`,
                    width: `${w * 100}%`,
                    height: `${h * 100}%`,
                    border: '2px solid red',
                    pointerEvents: 'none'
                  }}
                />
              );
            })}
          </div>
        </div>

        {/* 底部信息栏 */}
        <div style={{ padding: 'var(--space-4)', background: 'rgba(0,0,0,0.8)', color: 'white' }}>
          <div style={{ display: 'flex', gap: 'var(--space-6)', marginBottom: 'var(--space-3)', flexWrap: 'wrap' }}>
            <span>📐 尺寸: {fullscreenImage.width} x {fullscreenImage.height}</span>
            <span>📁 拆分: {fullscreenImage.split}</span>
            🏷️ 标注数: {fullscreenImage.label_count > 0 ? (
              <span style={{ color: 'var(--success-400)' }}>{fullscreenImage.label_count}</span>
            ) : (
              <span style={{ color: 'var(--text-secondary)' }}>无</span>
            )}
            <span>💾 大小: {formatFileSize(fullscreenImage.size)}</span>
          </div>

          {Object.keys(classCounts).length > 0 && (
            <div style={{ marginBottom: 'var(--space-3)' }}>
              <span style={{ marginRight: 'var(--space-2)' }}>类别统计:</span>
              {Object.entries(classCounts).map(([clsId, count]) => (
                <span key={clsId} style={{
                  display: 'inline-block', marginRight: 'var(--space-2)', padding: '2px 8px',
                  background: 'var(--primary-600)', borderRadius: '4px', fontSize: '0.875rem'
                }}>
                  class_{clsId}: {count}
                </span>
              ))}
            </div>
          )}

          <div style={{ display: 'flex', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
            <Button
              variant={showLabels ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => setShowLabels(!showLabels)}
            >
              👁️ {showLabels ? '隐藏标注' : '显示标注'}
            </Button>
            <Button variant="secondary" size="sm" onClick={() => setPixelView(!pixelView)}>
              🔍 {pixelView ? '关闭像素视图' : '像素视图'}
            </Button>
            <Button variant="secondary" size="sm" onClick={() => {
              const link = document.createElement('a');
              link.href = fullscreenImage.path;
              link.download = fullscreenImage.filename;
              link.click();
            }}>
              📥 下载
            </Button>
            <Button variant="danger" size="sm" onClick={() => handleDeleteImage(fullscreenImage.filename)}>
              🗑️ 删除
            </Button>
          </div>
        </div>
      </div>
    );
  };

  // 导出弹窗
  const renderExportModal = () => {
    if (!showExportModal) return null;

    return (
      <div style={{
        position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
        background: 'rgba(0,0,0,0.5)', zIndex: 1000,
        display: 'flex', alignItems: 'center', justifyContent: 'center'
      }}>
        <div style={{
          background: 'var(--bg-card)', borderRadius: 'var(--radius-lg)',
          padding: 'var(--space-6)', minWidth: '400px'
        }}>
          <h3 style={{ marginBottom: 'var(--space-4)' }}>导出数据集</h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-4)' }}>
            选择导出格式：
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
            <Button
              variant="primary"
              onClick={() => handleExport('yolo')}
              disabled={exporting}
            >
              YOLO 格式 (images + labels + data.yaml)
            </Button>
            <Button
              variant="secondary"
              onClick={() => handleExport('coco')}
              disabled={exporting}
            >
              COCO 格式 (images + annotations/*.json)
            </Button>
            <Button
              variant="secondary"
              onClick={() => handleExport('ndjson')}
              disabled={exporting}
            >
              NDJSON 格式 (每行一个JSON对象)
            </Button>
          </div>
          <Button
            variant="ghost"
            onClick={() => setShowExportModal(false)}
            style={{ marginTop: 'var(--space-4)', width: '100%' }}
          >
            取消
          </Button>
        </div>
      </div>
    );
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-6)' }}>
        <h1 style={{ fontFamily: 'DM Sans', fontWeight: 700 }}>数据集管理</h1>
        <div style={{ display: 'flex', gap: 'var(--space-3)' }}>
          <select
            value={selectedDataset || ''}
            onChange={(e) => setSelectedDataset(e.target.value)}
            style={{ padding: 'var(--space-2)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', minWidth: '200px' }}
          >
            {datasets.map((ds) => (
              <option key={ds.name} value={ds.name}>{ds.name}</option>
            ))}
          </select>
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => handleUpload(Array.from(e.target.files || []))}
            accept=".zip"
            style={{ display: 'none' }}
          />
          <Button variant="primary" loading={uploading} onClick={() => fileInputRef.current?.click()}>
            上传数据集
          </Button>
          <Button variant="secondary" disabled={!selectedDataset} onClick={handleDeleteDataset}>
            删除数据集
          </Button>
        </div>
      </div>

      {loading ? (
        <p>加载中...</p>
      ) : datasets.length === 0 ? (
        <Card>
          <div style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
            <p style={{ fontSize: '2rem', marginBottom: 'var(--space-4)' }}>📁</p>
            <p style={{ color: 'var(--text-secondary)' }}>暂无数据集</p>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>上传数据集开始使用</p>
          </div>
        </Card>
      ) : (
        <>
          <div style={{ display: 'flex', gap: 'var(--space-1)', marginBottom: 'var(--space-4)', borderBottom: '1px solid var(--border)' }}>
            {[
              { key: 'overview', label: '概览', icon: '📈' },
              { key: 'images', label: '图片浏览', icon: '🖼️' },
              { key: 'statistics', label: '详细统计', icon: '📊' },
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key as any)}
                style={{
                  padding: 'var(--space-2) var(--space-4)',
                  border: 'none',
                  background: activeTab === tab.key ? 'var(--primary-500)' : 'transparent',
                  color: activeTab === tab.key ? 'white' : 'var(--text-secondary)',
                  borderRadius: 'var(--radius-md) var(--radius-md) 0 0',
                  cursor: 'pointer',
                  fontWeight: 500,
                }}
              >
                {tab.icon} {tab.label}
              </button>
            ))}
          </div>

          <div style={{ minHeight: '400px' }}>
            {activeTab === 'overview' && renderOverview()}
            {activeTab === 'images' && renderImages()}
            {activeTab === 'statistics' && (
              <Card>
                <CardHeader icon="📊" title="详细统计" />
                <p style={{ color: 'var(--text-secondary)' }}>详细统计功能开发中...</p>
              </Card>
            )}
          </div>
        </>
      )}

      {renderFullscreenViewer()}
      {renderExportModal()}
    </div>
  );
};
