import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({ plugins: [react()], test: { projects: [
	{ extends: true, test: { name: 'backend', include: ['tests/unit/**/*.test.ts', 'tests/integration/**/*.test.ts'], environment: 'node', testTimeout: 60000 } },
	{ extends: true, test: { name: 'frontend', include: ['tests/unit/**/*.test.tsx'], environment: 'jsdom' } },
] } });