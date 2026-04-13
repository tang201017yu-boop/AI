import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendUrl = env.VITE_BACKEND_URL || 'http://localhost:8000'

  if (mode === 'development') {
    // 启动时确认代理目标；跨机部署请在 .env.development 设置 VITE_BACKEND_URL
    console.info(`[vite] API 代理目标: ${backendUrl} (/api, /uploads, /annotation-images)`)
  }

  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0',
      port: 3000,
      proxy: {
        '/api': {
          target: backendUrl,
          changeOrigin: true,
          secure: false,
          ws: true
        },
        '/uploads': {
          target: backendUrl,
          changeOrigin: true,
          secure: false
        },
        '/annotation-images': {
          target: backendUrl,
          changeOrigin: true,
          secure: false
        }
      }
    }
  }
})
