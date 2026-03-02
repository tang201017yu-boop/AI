import React from 'react';
import { Outlet } from 'react-router-dom';
import { Navbar } from '../Navbar/Navbar';
import styles from './Layout.module.css';

export const Layout: React.FC = () => {
  return (
    <div className={styles.layout}>
      <Navbar />
      <main className={styles.main}>
        <div className="container">
          <Outlet />
        </div>
      </main>
      <footer className={styles.footer}>
        <div className={styles.footerContent}>
          <p className={styles.footerText}>
            © 2024 AI Vision Platform. 基于 Ultralytics YOLO 与 Supervision.
          </p>
        </div>
      </footer>
    </div>
  );
};
