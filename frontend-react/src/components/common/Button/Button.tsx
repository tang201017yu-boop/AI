import React from 'react';
import styles from './Button.module.css';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'accent' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
  iconOnly?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  icon,
  iconOnly = false,
  className = '',
  disabled,
  ...props
}) => {
  const classes = [
    styles.button,
    styles[variant],
    styles[size],
    iconOnly && styles.iconOnly,
    className,
  ].filter(Boolean).join(' ');

  return (
    <button className={classes} disabled={disabled || loading} {...props}>
      {loading ? (
        <span className={styles.spinner} />
      ) : (
        <>
          {icon}
          {!iconOnly && children}
        </>
      )}
    </button>
  );
};
