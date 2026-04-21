import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendUrl = env.VITE_BACKEND_URL || 'http://localhost:8000'

  if (mode === 'development') {
    // 启动时确认代理目标；跨机部署请在 .env.development 设置 VITE_BACKEND_URL
    console.info(`[vite] API 代理目标: ${backendUrl} (/api, /uploads, /annotation-images)`)
  }

  /** dev 与 preview 共用，避免 `npm run preview` 时 /uploads 落到静态服务器 404 */
  const proxy = {
    '/api': {
      target: backendUrl,
      changeOrigin: true,
      secure: false,
      ws: true,
    },
    '/uploads': {
      target: backendUrl,
      changeOrigin: true,
      secure: false,
      timeout: 0,
      proxyTimeout: 0,
    },
    '/annotation-images': {
      target: backendUrl,
      changeOrigin: true,
      secure: false,
    },
  }

  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0',
      port: 3000,
      proxy,
    },
    preview: {
      host: '0.0.0.0',
      port: 4173,
      proxy,
    },
  }
})
