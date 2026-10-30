import { useCallback, useRef, useState } from 'react';
import {
  alpha,
  Box,
  Button,
  CircularProgress,
  Drawer,
  Fab,
  IconButton,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import ChatIcon from '@mui/icons-material/Chat';
import CloseIcon from '@mui/icons-material/Close';
import SendIcon from '@mui/icons-material/Send';
import DownloadIcon from '@mui/icons-material/Download';
import { useNavigate } from 'react-router-dom';
import type { AppDefinition } from '../../renderer/useAppDefinition';
import { getAkkioAccessToken, getAkkioApiBase } from '../../renderer/akkioApi';
import { downloadRagDataPdf } from '../../utils/ragDownload';

type ChatMessage = { role: 'user' | 'assistant'; content: string };

function formatApiDetail(detail: unknown): string {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === 'object' && 'msg' in item) {
          return String((item as { msg?: string }).msg);
        }
        return JSON.stringify(item);
      })
      .filter(Boolean)
      .join('; ');
  }
  if (detail && typeof detail === 'object') return JSON.stringify(detail);
  return 'Request failed';
}

function wantsRagDownload(text: string): boolean {
  const low = text.toLowerCase();
  return (
    /\b(download|export)\b.*\b(rag|context)\b/.test(low) ||
    /\brag data\b/.test(low) ||
    /\bexport\b.*\bdata\b/.test(low)
  );
}

function wantsPredictNav(text: string): boolean {
  const low = text.toLowerCase();
  return /\b(open|go to)\b.*\bpredict\b/.test(low) || /\bpredict page\b/.test(low);
}

export default function AppChatAssistant({
  definition,
  pageId,
  pageTitle,
}: {
  definition: AppDefinition;
  pageId?: string;
  pageTitle?: string;
}) {
  const apiBase = getAkkioApiBase();
  const navigate = useNavigate();
  const caps = definition.meta?.capabilities || [];
  const showRagExport =
    !caps.length || caps.includes('rag_export') || caps.includes('rag_chat');
  const hasPredictNav = definition.navigation?.some((n) => n.pageId === 'predict');
  const [open, setOpen] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: 'How can I help you?',
    },
  ]);
  const listRef = useRef<HTMLDivElement>(null);

  const scrollToEnd = () => {
    requestAnimationFrame(() => {
      listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
    });
  };

  const runRagExport = useCallback(async () => {
    setExporting(true);
    try {
      await downloadRagDataPdf(definition, pageId);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Export failed';
      setMessages((prev) => [...prev, { role: 'assistant', content: msg }]);
      scrollToEnd();
    } finally {
      setExporting(false);
    }
  }, [definition, pageId]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    if (wantsRagDownload(text)) {
      const nextUser: ChatMessage = { role: 'user', content: text };
      setMessages((prev) => [
        ...prev,
        nextUser,
        {
          role: 'assistant',
          content: 'Opening the RAG data export — use Print → Save as PDF in the new window.',
        },
      ]);
      setInput('');
      scrollToEnd();
      await runRagExport();
      return;
    }

    if (wantsPredictNav(text) && hasPredictNav) {
      const nextUser: ChatMessage = { role: 'user', content: text };
      setMessages((prev) => [
        ...prev,
        nextUser,
        { role: 'assistant', content: 'Opening the Predict page for manual inputs.' },
      ]);
      setInput('');
      navigate('/predict');
      scrollToEnd();
      return;
    }

    if (!apiBase) {
      setMessages((prev) => [
        ...prev,
        { role: 'user', content: text },
        {
          role: 'assistant',
          content: 'Chat is not available — open this app through the Akkio host so the API can load your dashboard data.',
        },
      ]);
      setInput('');
      scrollToEnd();
      return;
    }

    const nextUser: ChatMessage = { role: 'user', content: text };
    const history = [...messages.filter((m) => m.role === 'user' || m.role === 'assistant'), nextUser];
    setMessages((prev) => [...prev, nextUser]);
    setInput('');
    setLoading(true);
    scrollToEnd();

    try {
      const token = getAkkioAccessToken();
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (token) headers.Authorization = `Bearer ${token}`;

      const res = await fetch(`${apiBase}/chat`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          messages: history.map((m) => ({ role: m.role, content: m.content })),
          page_id: pageId,
          app_definition: {
            meta: definition.meta,
            pages: definition.pages,
            data: definition.data || definition.mock,
          },
        }),
      });
      const data = (await res.json()) as { content?: string; detail?: unknown };
      if (!res.ok) {
        throw new Error(formatApiDetail(data.detail) || `Request failed (${res.status})`);
      }
      setMessages((prev) => [...prev, { role: 'assistant', content: data.content || 'No response.' }]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Something went wrong.';
      const needsKeyHint = /api key|openai_api_key|not configured|503/i.test(msg);
      const suffix = needsKeyHint
        ? ' Add OPENAI_API_KEY to akkio-fastapi/.env on the server and restart the API.'
        : '';
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `I could not reach the assistant. ${msg}${suffix}`,
        },
      ]);
    } finally {
      setLoading(false);
      scrollToEnd();
    }
  }, [apiBase, definition, hasPredictNav, input, loading, messages, navigate, pageId, runRagExport]);

  return (
    <>
      <Fab
        color="primary"
        aria-label="Open data assistant"
        onClick={() => setOpen(true)}
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          zIndex: (t) => t.zIndex.drawer + 2,
        }}
      >
        <ChatIcon />
      </Fab>

      <Drawer
        anchor="right"
        open={open}
        onClose={() => setOpen(false)}
        PaperProps={{
          sx: {
            width: { xs: '100%', sm: 400 },
            display: 'flex',
            flexDirection: 'column',
            bgcolor: 'background.paper',
            color: 'text.primary',
          },
        }}
      >
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ p: 2, pb: 1 }}>
          <Typography variant="subtitle1" fontWeight={700} color="text.primary">
            Assistant
          </Typography>
          <IconButton onClick={() => setOpen(false)} aria-label="Close chat">
            <CloseIcon />
          </IconButton>
        </Stack>
        <Typography variant="caption" color="text.secondary" sx={{ px: 2, pb: 1 }}>
          Answers use this app&apos;s data — not the open web.
        </Typography>
        {showRagExport ? (
          <Box sx={{ px: 2, pb: 1 }}>
            <Button
              size="small"
              variant="outlined"
              startIcon={exporting ? <CircularProgress size={14} /> : <DownloadIcon />}
              onClick={runRagExport}
              disabled={exporting}
              fullWidth
            >
              Download RAG data (PDF)
            </Button>
          </Box>
        ) : null}

        <Box
          ref={listRef}
          sx={{
            flex: 1,
            overflow: 'auto',
            px: 2,
            py: 1,
            display: 'flex',
            flexDirection: 'column',
            gap: 1.5,
          }}
        >
          {messages.map((m, i) => (
            <Paper
              key={i}
              elevation={0}
              sx={(theme) => ({
                p: 1.5,
                alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                maxWidth: '92%',
                bgcolor:
                  m.role === 'user'
                    ? 'primary.main'
                    : alpha(theme.palette.text.primary, theme.palette.mode === 'dark' ? 0.12 : 0.06),
                color: m.role === 'user' ? 'primary.contrastText' : 'text.primary',
                borderRadius: 2,
                border: m.role === 'user' ? 'none' : `1px solid ${theme.palette.divider}`,
              })}
            >
              <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                {m.content}
              </Typography>
            </Paper>
          ))}
          {loading ? (
            <Stack direction="row" spacing={1} alignItems="center" sx={{ py: 1 }}>
              <CircularProgress size={18} />
              <Typography variant="caption" color="text.secondary">Thinking…</Typography>
            </Stack>
          ) : null}
        </Box>

        <Stack direction="row" spacing={1} sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
          <TextField
            fullWidth
            size="small"
            placeholder="Ask a question…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            disabled={loading}
            multiline
            maxRows={4}
          />
          <IconButton color="primary" onClick={send} disabled={loading || !input.trim()} aria-label="Send">
            <SendIcon />
          </IconButton>
        </Stack>
      </Drawer>
    </>
  );
}
