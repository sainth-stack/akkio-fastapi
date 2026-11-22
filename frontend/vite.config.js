import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [
      react({
        include: '**/*.{jsx,js}',
        babel: {
          plugins: [],
        },
      }),
    ],
    optimizeDeps: {
      include: ['plotly.js/dist/plotly', 'react-plotly.js'],
      extensions: ['.js', '.jsx'],
    },
    server: {
      port: Number(env.VITE_DEV_PORT) || 3002,
      host: true,
      open: false,
    },
    preview: {
      port: Number(env.VITE_DEV_PORT) || 3002,
      host: true,
    },
    build: {
      outDir: 'build',
      emptyOutDir: true,
      sourcemap: true,
    },
  };
});
