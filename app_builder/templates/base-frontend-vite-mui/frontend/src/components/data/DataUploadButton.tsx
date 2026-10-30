import { useRef, useState } from 'react';
import {
  Button,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Snackbar,
  Alert,
  Typography,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { getAkkioAccessToken, getAkkioApiBase } from '../../renderer/akkioApi';
import { useAppDefinition } from '../../renderer/useAppDefinition';

type ImportTarget = { dataKey: string; blockType: string; pageId: string; label: string };

export default function DataUploadButton() {
  const { definition, refreshDefinition, importTargets } = useAppDefinition();
  const apiBase = getAkkioApiBase();
  const fileRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [targetKey, setTargetKey] = useState('');
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [snack, setSnack] = useState<{ severity: 'success' | 'error'; message: string } | null>(
    null,
  );

  if (!apiBase || !definition) {
    return null;
  }

  const targets = importTargets.length ? importTargets : [];

  const runUpload = async (file: File, key?: string) => {
    setLoading(true);
    try {
      const token = getAkkioAccessToken();
      const headers: Record<string, string> = {};
      if (token) headers.Authorization = `Bearer ${token}`;
      const form = new FormData();
      form.append('file', file);
      if (key) form.append('target_key', key);
      const res = await fetch(`${apiBase}/data/import`, {
        method: 'POST',
        headers,
        body: form,
      });
      const data = (await res.json()) as {
        detail?: unknown;
        message?: string;
        updatedKeys?: string[];
        summary?: { updatedKeys?: string[]; rowCount?: number };
      };
      if (!res.ok) {
        const detail =
          typeof data.detail === 'string'
            ? data.detail
            : Array.isArray(data.detail)
              ? data.detail.map((d) => JSON.stringify(d)).join('; ')
              : 'Upload failed';
        throw new Error(detail);
      }
      await refreshDefinition();
      const keys = data.updatedKeys || data.summary?.updatedKeys || [];
      setSnack({
        severity: 'success',
        message: keys.length
          ? `Updated ${keys.join(', ')}. Dashboard refreshed.`
          : data.message || 'Data uploaded.',
      });
      setOpen(false);
      setPendingFile(null);
      setTargetKey('');
    } catch (e) {
      setSnack({
        severity: 'error',
        message: e instanceof Error ? e.message : 'Upload failed',
      });
    } finally {
      setLoading(false);
    }
  };

  const onFileChosen = (file: File | null) => {
    if (!file) return;
    if (targets.length > 1 && !targetKey) {
      setPendingFile(file);
      setOpen(true);
      return;
    }
    void runUpload(file, targetKey || targets[0]?.dataKey);
  };

  return (
    <>
      <input
        ref={fileRef}
        type="file"
        hidden
        accept=".csv,.json,.xlsx,.xls,application/json,text/csv"
        onChange={(e) => {
          onFileChosen(e.target.files?.[0] ?? null);
          e.target.value = '';
        }}
      />
      <Button
        size="small"
        variant="outlined"
        startIcon={loading ? <CircularProgress size={14} /> : <UploadFileIcon />}
        disabled={loading}
        onClick={() => fileRef.current?.click()}
        sx={{ ml: 'auto', whiteSpace: 'nowrap' }}
      >
        Upload data
      </Button>

      <Dialog open={open} onClose={() => !loading && setOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>Import data</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Choose which part of the app to update. CSV column headers should match table fields when
            possible.
          </Typography>
          <FormControl fullWidth size="small">
            <InputLabel id="import-target-label">Dataset</InputLabel>
            <Select
              labelId="import-target-label"
              label="Dataset"
              value={targetKey || targets[0]?.dataKey || ''}
              onChange={(e) => setTargetKey(String(e.target.value))}
            >
              {targets.map((t) => (
                <MenuItem key={t.dataKey} value={t.dataKey}>
                  {t.label} ({t.dataKey})
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          {pendingFile ? (
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              File: {pendingFile.name}
            </Typography>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)} disabled={loading}>
            Cancel
          </Button>
          <Button
            variant="contained"
            disabled={loading || !pendingFile}
            onClick={() => pendingFile && runUpload(pendingFile, targetKey || targets[0]?.dataKey)}
          >
            Import
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={!!snack}
        autoHideDuration={6000}
        onClose={() => setSnack(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        {snack ? (
          <Alert severity={snack.severity} onClose={() => setSnack(null)} sx={{ width: '100%' }}>
            {snack.message}
          </Alert>
        ) : null}
      </Snackbar>
    </>
  );
}

/** Compact hint for dashboard first page */
export function DataUploadHint() {
  const { importTargets } = useAppDefinition();
  const apiBase = getAkkioApiBase();
  if (!apiBase || !importTargets.length) return null;
  return (
    <List dense sx={{ mb: 2, bgcolor: 'action.hover', borderRadius: 2 }}>
      <ListItem>
        <ListItemText
          primary="Use your own data"
          secondary="Upload CSV, Excel, or JSON from the toolbar — KPIs, charts, and tables update immediately."
        />
      </ListItem>
    </List>
  );
}
