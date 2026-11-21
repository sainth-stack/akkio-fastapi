import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { transformWithEsbuild } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [
      {
        name: 'treat-js-files-as-jsx',
        async transform(code, id) {
          if (!id.includes('/src/') || !id.endsWith('.js')) {
            return null;
          }
          return transformWithEsbuild(code, id, {
            loader: 'jsx',
            jsx: 'automatic',
          });
        },
      },
      react({
        include: '**/*.{jsx,js}',
      }),
    ],
    optimizeDeps: {
      include: ['plotly.js/dist/plotly', 'react-plotly.js'],
      esbuildOptions: {
        loader: {
          '.js': 'jsx',
        },
      },
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
