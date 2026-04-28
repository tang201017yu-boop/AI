import React from 'react';
import { NavLink } from 'react-router-dom';
import styles from './Navbar.module.css';

const navItems = [
  { path: '/', label: '首页' },
  { path: '/ai-native', label: 'AI原生' },
  { path: '/inference', label: '模型推理' },
  { path: '/training', label: '模型训练' },
  { path: '/solutions', label: '智能方案' },
  { path: '/models', label: '模型管理' },
  { path: '/datasets', label: '数据集' },
  { path: '/annotation', label: '智能标注' },
  { path: '/augmentation', label: '数据增强' },
  { path: '/governance/rules', label: '规则库' },
  { path: '/governance/arbitration', label: '仲裁' },
  { path: '/governance/annotation', label: '标注增强' },
];

export const Navbar: React.FC = () => {
  return (
    <nav className={styles.navbar}>
      <div className={styles.container}>
        <div className={styles.logoSection}>
          <img src="/logo.svg" alt="Logo" className={styles.logo} />
          <span className={styles.logoText}>AI Vision</span>
        </div>
        <div className={styles.nav}>
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `${styles.navLink} ${isActive ? styles.navLinkActive : ''}`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  );
};
