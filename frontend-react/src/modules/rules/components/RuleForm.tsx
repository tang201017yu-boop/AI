import React, { useState } from 'react';
import { Button, Input } from '../../../components/common';
import type { RuleFormValues } from '../../../types';
import styles from './RuleForm.module.css';

export interface RuleFormProps {
  initial: RuleFormValues;
  submitLabel: string;
  onSubmit: (values: RuleFormValues) => void | Promise<void>;
  onCancel?: () => void;
}

export const RuleForm: React.FC<RuleFormProps> = ({ initial, submitLabel, onSubmit, onCancel }) => {
  const [name, setName] = useState(initial.name);
  const [description, setDescription] = useState(initial.description);
  const [content, setContent] = useState(initial.content);
  const [tags, setTags] = useState(initial.tags);
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      await onSubmit({ name, description, content, tags });
    } finally {
      setSaving(false);
    }
  };

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <Input label="规则名称" value={name} onChange={(e) => setName(e.target.value)} required />
      <div className={styles.field}>
        <label className={styles.label}>说明</label>
        <input
          className={styles.input}
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="可选"
        />
      </div>
      <div className={styles.field}>
        <label className={styles.label}>规则内容</label>
        <textarea
          className={styles.textarea}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          required
          rows={10}
        />
      </div>
      <div className={styles.field}>
        <label className={styles.label}>标签（逗号分隔）</label>
        <input
          className={styles.input}
          value={tags}
          onChange={(e) => setTags(e.target.value)}
        />
      </div>
      <div className={styles.actions}>
        {onCancel ? (
          <Button type="button" variant="secondary" onClick={onCancel}>
            取消
          </Button>
        ) : null}
        <Button type="submit" variant="primary" disabled={saving}>
          {saving ? '保存中…' : submitLabel}
        </Button>
      </div>
    </form>
  );
};
