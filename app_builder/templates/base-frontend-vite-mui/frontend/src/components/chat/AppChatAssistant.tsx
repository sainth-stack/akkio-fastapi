import { useCallback, useRef, useState } from 'react';
import {
  Box,
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
import type { AppDefinition } from '../../renderer/useAppDefinition';
import { getAkkioAccessToken, getAkkioApiBase } from '../../renderer/akkioApi';

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
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: `Hi — ask me about KPIs, charts, and data on ${definition.meta?.appName || 'this dashboard'}${pageTitle ? ` (${pageTitle})` : ''}.`,
    },
  ]);
  const listRef = useRef<HTMLDivElement>(null);

  const scrollToEnd = () => {
    requestAnimationFrame(() => {
      listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
    });
  };

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;
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
  }, [apiBase, definition, input, loading, messages, pageId]);

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
          sx: { width: { xs: '100%', sm: 400 }, display: 'flex', flexDirection: 'column' },
        }}
      >
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ p: 2, pb: 1 }}>
          <Typography variant="subtitle1" fontWeight={700}>
            Data assistant
          </Typography>
          <IconButton onClick={() => setOpen(false)} aria-label="Close chat">
            <CloseIcon />
          </IconButton>
        </Stack>
        <Typography variant="caption" color="text.secondary" sx={{ px: 2, pb: 1 }}>
          Answers use your app KPIs, charts, and tables — not the open web.
        </Typography>

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
              sx={{
                p: 1.5,
                alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                maxWidth: '92%',
                bgcolor: m.role === 'user' ? 'primary.main' : 'action.hover',
                color: m.role === 'user' ? 'primary.contrastText' : 'text.primary',
                borderRadius: 2,
              }}
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
            placeholder="Ask about MTBF, downtime, trends…"
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
