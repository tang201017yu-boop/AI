import React, { useCallback } from 'react';
import styles from './Upload.module.css';

interface UploadProps {
  accept?: string;
  multiple?: boolean;
  onFilesSelected: (files: File[]) => void;
  hint?: string;
}

export const Upload: React.FC<UploadProps> = ({
  accept,
  multiple = false,
  onFilesSelected,
  hint = '点击或拖拽文件到此处上传',
}) => {
  const [dragOver, setDragOver] = React.useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      onFilesSelected(Array.from(e.target.files));
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files) {
      onFilesSelected(Array.from(e.dataTransfer.files));
    }
  }, [onFilesSelected]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  return (
    <div
      className={`${styles.uploadArea} ${dragOver ? styles.dragOver : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={() => document.getElementById('file-input')?.click()}
    >
      <input
        type="file"
        id="file-input"
        className={styles.input}
        accept={accept}
        multiple={multiple}
        onChange={handleChange}
      />
      <div className={styles.icon}>📁</div>
      <div className={styles.text}>{hint}</div>
      {accept && <div className={styles.hint}>支持格式: {accept}</div>}
    </div>
  );
};
