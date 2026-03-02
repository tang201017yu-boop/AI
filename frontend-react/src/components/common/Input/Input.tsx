import React from 'react';
import styles from './Input.module.css';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helper?: string;
}

export const Input: React.FC<InputProps> = ({
  label,
  error,
  helper,
  required,
  className = '',
  ...props
}) => {
  return (
    <div className={styles.wrapper}>
      {label && (
        <label className={`${styles.label} ${required ? styles.required : ''}`}>
          {label}
        </label>
      )}
      <input
        className={`${styles.input} ${error ? styles.error : ''} ${className}`}
        required={required}
        {...props}
      />
      {error && <span className={styles.errorText}>{error}</span>}
      {!error && helper && <span className={styles.helper}>{helper}</span>}
    </div>
  );
};
