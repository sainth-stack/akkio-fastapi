// Centralized user-friendly error messages for the App Builder

export const getFriendlyError = (rawError) => {
  if (!rawError) return 'Something went wrong. Please try again.';

  const msg = typeof rawError === 'string' ? rawError : rawError.message || JSON.stringify(rawError);

  // Strip technical details
  if (msg.includes('API key') || msg.includes('.env') || msg.includes('OpenAI') || msg.includes('akkio-fastapi')) {
    return 'Generation service is unavailable. Please try again later.';
  }
  if (msg.includes('ECONNREFUSED') || msg.includes('fetch') || msg.includes('network') || msg.toLowerCase().includes('connection')) {
    return 'Connection error. Please check your internet and try again.';
  }
  if (msg.includes('timeout') || msg.includes('Timeout')) {
    return 'Request timed out. Please try again.';
  }
  if (msg.includes('500') || msg.includes('Internal Server Error')) {
    return 'Server error. Please try again in a moment.';
  }
  if (msg.includes('401') || msg.includes('403') || msg.includes('Unauthorized')) {
    return 'Session expired. Please refresh the page.';
  }
  if (msg.includes('404')) {
    return 'Resource not found. Please try again.';
  }
  if (msg.includes('npm') || msg.includes('build failed') || msg.includes('Build failed') || msg.includes('webpack') || msg.includes('vite')) {
    return 'App build encountered an error. Try regenerating or simplifying your request.';
  }

  // If short and doesn't look too technical, show it
  if (msg.length < 100 && !msg.includes('/') && !msg.includes('Error:')) {
    return msg;
  }

  return 'Something went wrong. Please try again.';
};

export const PIPELINE_STATE_LABELS = {
  PRD_RUNNING: 'Generating your plan\u2026',
  PRD_COMPLETE: 'Plan ready',
  UIUX_RUNNING: 'Designing UI/UX\u2026',
  UIUX_COMPLETE: 'UI/UX design ready',
  STYLE_RUNNING: 'Building design system\u2026',
  STYLE_COMPLETE: 'Design system ready',
  ARCH_RUNNING: 'Designing architecture\u2026',
  ARCH_COMPLETE: 'Architecture ready',
  ARCHITECTURE_RUNNING: 'Designing architecture\u2026',
  ARCHITECTURE_COMPLETE: 'Architecture ready',
  ARCHITECTURE_FAILED: 'Architecture failed \u2014 try again',
  CODEGEN_RUNNING: 'Building your app\u2026',
  CODEGEN_COMPLETE: 'App built successfully',
  CODEGEN_FAILED: 'Build failed \u2014 try again',
  AGENTS_RUNNING: 'Agents working\u2026',
  AGENTS_COMPLETE: 'Agents done',
  PLAN_RUNNING: 'Planning your app\u2026',
  PLAN_COMPLETE: 'Plan ready',
  PLAN_FAILED: 'Planning failed \u2014 try again',
  PRD_FAILED: 'Plan generation failed \u2014 try again',
  UIUX_FAILED: 'UI/UX design failed \u2014 try again',
  STYLE_FAILED: 'Design system failed \u2014 try again',
  IDLE: 'Ready',
  ERROR: 'Something went wrong',
};

export const getFriendlyPipelineState = (state) => {
  return PIPELINE_STATE_LABELS[state] || (state ? state.replace(/_/g, ' ').toLowerCase() : 'Ready');
};
