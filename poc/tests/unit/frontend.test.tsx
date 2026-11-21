import React from 'react';
import { afterEach, describe, it, expect } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom/vitest';
import { PromptInput } from '../../frontend/src/components/PromptInput';
import { GenerationStatus } from '../../frontend/src/components/GenerationStatus';
import { GeneratedImage } from '../../frontend/src/components/GeneratedImage';
import { statuses, type GenerationJob } from '../../shared/types';
import { GeminiSignIn } from '../../frontend/src/components/GeminiSignIn';

afterEach(cleanup);
describe('frontend states', () => {
  it('opens Google sign-in without collecting passwords', async () => {
    let action = '';
    render(<GeminiSignIn session={{ status: 'IDLE', error: null }} pending={false} disabled={false} onAction={value => { action = value; }}/>);
    await userEvent.click(screen.getByRole('button', { name: 'Sign in to Gemini' }));
    expect(action).toBe('start');
    expect(screen.queryByRole('textbox')).toBeNull();
  });
  it('keeps sign-in waiting on verification failure and allows confirmation or cancellation', async () => {
    let action = '';
    render(<GeminiSignIn session={{ status: 'WAITING_FOR_LOGIN', error: { code: 'AUTH_REQUIRED', message: 'Finish Google sign-in.' } }} pending={false} disabled={false} onAction={value => { action = value; }}/>);
    expect(screen.getByRole('status')).toHaveTextContent('browser stays open');
    expect(screen.getByRole('alert')).toHaveTextContent('Finish Google sign-in.');
    await userEvent.click(screen.getByRole('button', { name: 'I have signed in' }));
    expect(action).toBe('confirm');
    await userEvent.click(screen.getByRole('button', { name: 'Cancel sign-in' }));
    expect(action).toBe('cancel');
  });
  it('disables empty and busy submissions', () => {
    const { rerender } = render(<PromptInput prompt=" " setPrompt={() => {}} busy={false} onGenerate={() => {}}/>);
    expect(screen.getByRole('button', { name: /generate image/i })).toBeDisabled();
    rerender(<PromptInput prompt="An image" setPrompt={() => {}} busy={true} onGenerate={() => {}}/>);
    expect(screen.getByRole('textbox')).toBeDisabled();
  });
  it('clears the prompt', async () => {
    let cleared = false;
    render(<PromptInput prompt="Image" setPrompt={value => { cleared = value === ''; }} busy={false} onGenerate={() => {}}/>);
    await userEvent.click(screen.getByRole('button', { name: /clear prompt/i }));
    expect(cleared).toBe(true);
  });
  it.each(['IDLE', ...statuses] as const)('announces %s', status => {
    render(<GenerationStatus status={status}/>);
    expect(screen.getByRole('status')).not.toBeEmptyDOMElement();
  });
  it('has no fake output and displays only a completed actual image URL', () => {
    const { rerender } = render(<GeneratedImage job={null} status="IDLE"/>);
    expect(screen.queryByRole('img')).toBeNull();
    const job = { jobId: 'test', status: 'COMPLETED', prompt: 'Test prompt', imageUrl: '/api/generated-images/test.png', metadata: { width: 512, height: 512, bytes: 2000, mime: 'image/png' } } as GenerationJob;
    rerender(<GeneratedImage job={job} status="COMPLETED"/>);
    expect(screen.getByRole('img')).toHaveAttribute('src', job.imageUrl);
    expect(screen.getByRole('link', { name: /download image/i })).toHaveAttribute('href', `${job.imageUrl}?download=1`);
  });
});