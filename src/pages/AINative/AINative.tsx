import React, { useState } from 'react';
import { Card, CardHeader, Button, Input } from '../../components/common';
import styles from './AINative.module.css';

interface Scene {
  name: string;
  description: string;
  keywords: string[];
  default_threshold: number;
}

interface ParsedResult {
  success: boolean;
  scenes: string[];
  threshold: number;
  threshold_unit: string | null;
  alarm: boolean;
  alarm_type: string | null;
  models: string[];
  config: Record<string, any>;
  explanation: string;
}

export const AINative: React.FC = () => {
  const [requirement, setRequirement] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ParsedResult | null>(null);
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [projects, setProjects] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState<'create' | 'projects'>('create');

  // 加载支持的场景
  React.useEffect(() => {
    fetch('/api/v1/ai-native/scenes')
      .then(res => res.json())
      .then(data => {
        if (data.scenes) {
          setScenes(data.scenes);
        }
      })
      .catch(console.error);

    // 加载已有项目
    fetch('/api/v1/ai-native/projects')
      .then(res => res.json())
      .then(data => {
        if (data.projects) {
          setProjects(data.projects);
        }
      })
      .catch(console.error);
  }, []);

  const handleAnalyze = async () => {
    if (!requirement.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/v1/ai-native/understand?text=${encodeURIComponent(requirement)}`);
      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateProject = async () => {
    if (!result) return;
    setLoading(true);
    try {
      const res = await fetch('/api/v1/ai-native/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          requirement: requirement,
          project_name: requirement.slice(0, 20)
        })
      });
      const data = await res.json();
      if (data.success) {
        alert('检测系统创建成功！');
        // 刷新项目列表
        const res = await fetch('/api/v1/ai-native/projects');
        const data = await res.json();
        if (data.projects) {
          setProjects(data.projects);
        }
      }
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const examplePrompts = [
    "帮我检测工厂里的安全帽",
    "检测隧道裂缝，超过5mm要报警",
    "检测流水线上的产品缺陷",
    "检测违规占道经营行为",
    "检测烟雾和火焰，有火情时立即报警"
  ];

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>🔮 AI 智能检测系统</h1>
        <p className={styles.subtitle}>
          用自然语言描述需求，AI自动为您生成完整的检测系统
        </p>
      </div>

      {/* 标签页 */}
      <div className={styles.tabs}>
        <button
          className={`${styles.tab} ${activeTab === 'create' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('create')}
        >
          🆕 创建检测系统
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'projects' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('projects')}
        >
          📁 我的项目 ({projects.length})
        </button>
      </div>

      {activeTab === 'create' ? (
        <div className={styles.content}>
          {/* 输入区域 */}
          <Card>
            <CardHeader icon="💬" title="描述您的需求" />
            <div className={styles.inputArea}>
              <textarea
                className={styles.textarea}
                placeholder="用自然语言描述您想检测什么...
例如：帮我检测工厂里的安全帽，检测到未佩戴安全帽的人员时立即报警"
                value={requirement}
                onChange={(e) => setRequirement(e.target.value)}
                rows={4}
              />
              <div className={styles.inputActions}>
                <Button
                  variant="primary"
                  size="lg"
                  onClick={handleAnalyze}
                  loading={loading}
                  disabled={!requirement.trim()}
                >
                  🚀 开始生成
                </Button>
              </div>
            </div>

            {/* 示例提示 */}
            <div className={styles.examples}>
              <span className={styles.examplesLabel}>试试这样描述：</span>
              {examplePrompts.map((example, idx) => (
                <button
                  key={idx}
                  className={styles.exampleBtn}
                  onClick={() => setRequirement(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </Card>

          {/* 分析结果 */}
          {result && (
            <Card>
              <CardHeader icon="📊" title="AI 分析结果" />

              <div className={styles.resultSection}>
                <div className={styles.explanation}>
                  {result.explanation}
                </div>

                {/* 识别到的场景 */}
                <div className={styles.resultItem}>
                  <span className={styles.resultLabel}>🎯 检测目标：</span>
                  {result.scenes.length > 0 ? (
                    <div className={styles.tagGroup}>
                      {result.scenes.map((scene, idx) => (
                        <span key={idx} className={styles.tag}>
                          {scene}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <span className={styles.tagDefault}>通用目标检测</span>
                  )}
                </div>

                {/* 阈值 */}
                <div className={styles.resultItem}>
                  <span className={styles.resultLabel}>📐 置信度阈值：</span>
                  <span className={styles.tag}>{(result.threshold * 100).toFixed(0)}%</span>
                </div>

                {/* 报警 */}
                <div className={styles.resultItem}>
                  <span className={styles.resultLabel}>🔔 报警设置：</span>
                  {result.alarm ? (
                    <span className={styles.tagAlarm}>
                      {result.alarm_type === 'sound' ? '📢 声音报警' :
                        result.alarm_type === 'sms' ? '📱 短信通知' :
                          result.alarm_type === 'email' ? '📧 邮件通知' : '🔔 弹窗报警'}
                    </span>
                  ) : (
                    <span className={styles.tagDefault}>不报警</span>
                  )}
                </div>

                {/* 推荐模型 */}
                <div className={styles.resultItem}>
                  <span className={styles.resultLabel}>🤖 推荐模型：</span>
                  <div className={styles.tagGroup}>
                    {result.models.slice(0, 5).map((model, idx) => (
                      <span key={idx} className={styles.tagModel}>{model}</span>
                    ))}
                  </div>
                </div>
              </div>

              {/* 创建按钮 */}
              <div className={styles.createActions}>
                <Button
                  variant="primary"
                  size="lg"
                  onClick={handleCreateProject}
                  loading={loading}
                >
                  ✅ 确认并创建检测系统
                </Button>
              </div>
            </Card>
          )}

          {/* 支持的场景 */}
          <Card>
            <CardHeader icon="📚" title="支持的检测场景" />
            <div className={styles.scenesGrid}>
              {scenes.map((scene, idx) => (
                <div key={idx} className={styles.sceneCard}>
                  <div className={styles.sceneName}>{scene.name}</div>
                  <div className={styles.sceneDesc}>{scene.description}</div>
                  <div className={styles.sceneKeywords}>
                    关键词：{scene.keywords.slice(0, 3).join('、')}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      ) : (
        <div className={styles.content}>
          {projects.length === 0 ? (
            <Card>
              <div className={styles.emptyState}>
                <p className={styles.emptyIcon}>📁</p>
                <p>暂无项目</p>
                <p className={styles.emptyHint}>创建一个AI检测系统开始使用</p>
              </div>
            </Card>
          ) : (
            <div className={styles.projectsGrid}>
              {projects.map((project, idx) => (
                <Card key={idx} className={styles.projectCard}>
                  <div className={styles.projectName}>{project.name}</div>
                  <div className={styles.projectDesc}>{project.description}</div>
                  <div className={styles.projectMeta}>
                    <span>状态：{project.status}</span>
                    <span>创建于：{new Date(project.created_at).toLocaleDateString()}</span>
                  </div>
                  <div className={styles.projectActions}>
                    <Button variant="secondary" size="sm">查看详情</Button>
                    <Button variant="primary" size="sm">开始检测</Button>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
