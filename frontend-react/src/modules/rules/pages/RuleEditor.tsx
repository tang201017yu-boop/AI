import React, { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { Card, CardHeader } from '../../../components/common';
import { RuleForm } from '../components/RuleForm';
import { rulesApi } from '../api/rulesApi';
import type { RuleFormValues } from '../../../types';
import styles from './RuleEditor.module.css';

const empty: RuleFormValues = { name: '', description: '', content: '', tags: '' };

export const RuleEditor: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const isEdit = Boolean(id);
  const [initial, setInitial] = useState<RuleFormValues>(empty);
  const [loadErr, setLoadErr] = useState(false);

  useEffect(() => {
    if (!id) {
      setInitial(empty);
      return;
    }
    let c = false;
    (async () => {
      try {
        const res = await rulesApi.get(id);
        const r = (res.data as { data?: { name: string; description?: string; content: string; tags?: string[] } })?.data;
        if (!c && r) {
          setInitial({
            name: r.name,
            description: r.description || '',
            content: r.content,
            tags: (r.tags || []).join(', '),
          });
        }
      } catch {
        if (!c) {
          setLoadErr(true);
          setInitial({ ...empty, name: '未加载', content: '无法从服务器读取规则' });
        }
      }
    })();
    return () => {
      c = true;
    };
  }, [id]);

  const onSubmit = async (values: RuleFormValues) => {
    const tags = values.tags
      .split(/[,，]/)
      .map((s) => s.trim())
      .filter(Boolean);
    try {
      if (isEdit && id) {
        await rulesApi.update(id, {
          name: values.name,
          description: values.description,
          content: values.content,
          tags,
        });
        navigate(`/governance/rules/${id}`);
        return;
      }
      const res = await rulesApi.create({
        name: values.name,
        description: values.description,
        content: values.content,
        tags,
      });
      const created = (res.data as { data?: { id: string } })?.data;
      if (created?.id) navigate(`/governance/rules/${created.id}`);
      else navigate('/governance/rules');
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.top}>
        <Link to={isEdit && id ? `/governance/rules/${id}` : '/governance/rules'} className={styles.back}>
          ← 返回
        </Link>
      </div>
      {loadErr ? <p className={styles.warn}>无法加载远程规则，仍可编辑后保存（需后端支持）。</p> : null}
      <Card>
        <CardHeader title={isEdit ? '编辑规则' : '新建规则'} />
        <RuleForm
          key={`${isEdit ? id : 'new'}`}
          initial={initial}
          submitLabel={isEdit ? '保存' : '创建'}
          onSubmit={onSubmit}
          onCancel={() => navigate(isEdit && id ? `/governance/rules/${id}` : '/governance/rules')}
        />
      </Card>
    </div>
  );
};
