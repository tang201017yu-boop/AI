import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Card, CardHeader, Button, StatCard } from '../../components/common';
import { systemApi } from '../../services/api';
import type { SystemInfo } from '../../types';
import { SolutionFeatureIcon } from '../Solutions/solutionIcons';
import {
  IconHomeAnnotation,
  IconHomeChart,
  IconHomeSolutions,
  IconHomeTrain,
  IconHomeWorkflow,
} from './homeIcons';
import styles from './Home.module.css';

const solutions: { id: string; title: string; desc: string }[] = [
  { id: 'object-counting', title: '对象计数', desc: '统计区域对象数量' },
  { id: 'heatmap', title: '热图分析', desc: '可视化检测密度' },
  { id: 'speed-estimation', title: '速度估算', desc: '计算移动对象速度' },
  { id: 'distance-calculation', title: '距离计算', desc: '测量对象间距离' },
  { id: 'object-blur', title: '隐私保护', desc: '对象模糊处理' },
  { id: 'object-crop', title: '对象裁剪', desc: '自动提取检测对象' },
  { id: 'queue-management', title: '队列管理', desc: '监控队列长度' },
  { id: 'vision-eye', title: '虚拟围栏', desc: '区域入侵检测' },
];

export const Home: React.FC = () => {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);

  useEffect(() => {
    systemApi.getInfo()
      .then((res) => setSystemInfo(((res.data as any)?.data || res.data) as SystemInfo | null))
      .catch(console.error);
  }, []);

  return (
    <div>
      {/* Hero Section */}
      <div className={styles.hero}>
        <h1 className={styles.heroTitle}>
          AI <span className={styles.heroTitleAccent}>Vision</span> Platform
        </h1>
        <p className={styles.heroSubtitle}>
          基于 Ultralytics YOLO 与 Supervision 的新一代智能视觉平台
          <br />
          融合 AI 驱动的数据标注、模型训练与部署解决方案
        </p>
        <div className={styles.heroButtons}>
          <Link to="/annotation">
            <Button variant="primary" size="lg">
              <span className={styles.heroBtnInner}>
                <IconHomeAnnotation size={20} />
                智能标注
              </span>
            </Button>
          </Link>
          <Link to="/training">
            <Button variant="secondary" size="lg">
              <span className={styles.heroBtnInner}>
                <IconHomeTrain size={20} />
                开始训练
              </span>
            </Button>
          </Link>
        </div>
      </div>

      {/* System Stats */}
      <div className={styles.section}>
        <Card>
          <CardHeader title="系统状态" icon={<IconHomeChart />} />
          <div className={styles.grid4}>
            <StatCard value={systemInfo?.total_models || '-'} label="模型数量" />
            <StatCard value={systemInfo?.total_datasets || '-'} label="数据集数量" variant="accent" />
            <StatCard value={systemInfo?.gpu_available ? 'GPU' : 'CPU'} label="计算资源" variant="success" />
            <StatCard value={systemInfo?.ultralytics_version || 'YOLO'} label="YOLO 版本" />
          </div>
        </Card>
      </div>

      {/* AI Workflow */}
      <div className={styles.section}>
        <Card>
          <CardHeader title="AI 工作流" icon={<IconHomeWorkflow />} />
          <div className={styles.grid3}>
            <div className={styles.workflowStep} data-step="1">
              <h3 className={styles.workflowTitle}>
                <span style={{ color: 'var(--primary-500)' }}>01.</span> Supervision 智能标注
              </h3>
              <p className={styles.workflowDesc}>
                集成 Supervision 智能标注系统，支持 SAM 自动分割、YOLO 预标注，一键生成高质量训练数据
              </p>
              <Link to="/annotation">
                <Button variant="primary" size="sm" style={{ marginTop: 'var(--space-4)' }}>
                  开始标注 →
                </Button>
              </Link>
            </div>
            <div className={styles.workflowStep} data-step="2">
              <h3 className={styles.workflowTitle}>
                <span style={{ color: 'var(--primary-500)' }}>02.</span> 模型训练
              </h3>
              <p className={styles.workflowDesc}>
                基于 Ultralytics YOLO 框架，支持多模型训练、超参数调优、实时监控训练进度
              </p>
              <Link to="/training">
                <Button variant="secondary" size="sm" style={{ marginTop: 'var(--space-4)' }}>
                  开始训练 →
                </Button>
              </Link>
            </div>
            <div className={styles.workflowStep} data-step="3">
              <h3 className={styles.workflowTitle}>
                <span style={{ color: 'var(--primary-500)' }}>03.</span> 智能部署
              </h3>
              <p className={styles.workflowDesc}>
                一键部署训练模型，提供 RESTful API，支持 TensorRT 加速、批量推理
              </p>
              <Link to="/inference">
                <Button variant="secondary" size="sm" style={{ marginTop: 'var(--space-4)' }}>
                  开始推理 →
                </Button>
              </Link>
            </div>
          </div>
        </Card>
      </div>

      {/* Solutions */}
      <div className={styles.section}>
        <Card>
          <CardHeader title="Ultralytics 智能解决方案" icon={<IconHomeSolutions />} />
          <div className={styles.grid4}>
            {solutions.map((item) => (
              <div key={item.id} className={styles.solutionCard}>
                <div className={styles.solutionIcon} aria-hidden>
                  <SolutionFeatureIcon name={item.id} size={32} />
                </div>
                <div className={styles.solutionTitle}>{item.title}</div>
                <div className={styles.solutionDesc}>{item.desc}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
};
