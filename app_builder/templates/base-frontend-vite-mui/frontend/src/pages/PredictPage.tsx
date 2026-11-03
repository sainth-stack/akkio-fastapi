import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import SyncIcon from '@mui/icons-material/Sync';
import { useAppDefinition } from '../renderer/useAppDefinition';
import { buildAppDefinitionPayload } from '../utils/ragDownload';
import { getAkkioAccessToken, getAkkioApiBase } from '../renderer/akkioApi';
import PredictionIndicators, { type LivePrediction } from '../components/predict/PredictionIndicators';

type PredictField = { key: string; label: string; unit?: string };
type PredictConfig = {
  targetField?: string;
  fields?: PredictField[];
  baseline?: Record<string, number>;
  description?: string;
};

function fieldLabel(f: PredictField): string {
  const { label, unit } = f;
  if (!unit) return label;
  const low = label.toLowerCase();
  if (low.includes(`(${unit})`) || low.includes(unit)) return label;
  return `${label} (${unit})`;
}

const LEGACY_FIELDS: PredictField[] = [
  { key: 'mtbf_hours', label: 'MTBF (hours)', unit: 'h' },
  { key: 'mttr_hours', label: 'MTTR (hours)', unit: 'h' },
  { key: 'downtime_hours', label: 'Downtime (hours)', unit: 'h' },
  { key: 'open_work_orders', label: 'Open work orders' },
  { key: 'machine_load_pct', label: 'Machine load / OEE (%)', unit: '%' },
  { key: 'production_output', label: 'Production output' },
];

function baselineToForm(fields: PredictField[], baseline: Record<string, number>): Record<string, string> {
  const out: Record<string, string> = {};
  fields.forEach(({ key }) => {
    out[key] = baseline[key] != null ? String(baseline[key]) : '';
  });
  return out;
}

export default function PredictPage() {
  const { definition } = useAppDefinition();
  const apiBase = getAkkioApiBase();
  const predictConfig = (definition?.meta?.predictConfig as PredictConfig | undefined) ?? undefined;

  const fields = useMemo<PredictField[]>(() => {
    if (predictConfig?.fields?.length) {
      return predictConfig.fields.map((f) => ({
        key: f.key,
        label: f.label || f.key.replace(/_/g, ' '),
        unit: f.unit,
      }));
    }
    return LEGACY_FIELDS;
  }, [predictConfig]);

  const [targetField, setTargetField] = useState('');
  const [values, setValues] = useState<Record<string, string>>({});
  const [scenario, setScenario] = useState('');
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [result, setResult] = useState<LivePrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [baselineLoaded, setBaselineLoaded] = useState(false);

  useEffect(() => {
    const initialTarget =
      predictConfig?.targetField || fields[0]?.key || 'production_output';
    setTargetField(initialTarget);
    if (predictConfig?.baseline) {
      setValues(baselineToForm(fields, predictConfig.baseline));
      setBaselineLoaded(true);
    }
  }, [fields, predictConfig?.baseline, predictConfig?.targetField]);

  const inputFields = useMemo(
    () => fields.filter((f) => f.key !== targetField),
    [fields, targetField],
  );

  const postPredict = useCallback(
    async (body: Record<string, unknown>) => {
      if (!apiBase) throw new Error('Open this app from the Akkio host to run predictions.');
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      const token = getAkkioAccessToken();
      if (token) headers.Authorization = `Bearer ${token}`;
      const res = await fetch(`${apiBase}/predict`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          app_definition: buildAppDefinitionPayload(definition),
          ...body,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(String(data.detail || 'Prediction failed'));
      return data as LivePrediction & { baseline?: Record<string, number>; predictConfig?: PredictConfig };
    },
    [apiBase, definition],
  );

  const syncFromLiveData = useCallback(async () => {
    setSyncing(true);
    setError(null);
    try {
      const data = await postPredict({ refresh_baseline_only: true });
      const cfg = data.predictConfig || predictConfig;
      const syncFields = cfg?.fields?.length
        ? cfg.fields.map((f) => ({ key: f.key, label: f.label, unit: f.unit }))
        : fields;
      const baseline = data.baseline || cfg?.baseline || {};
      setValues(baselineToForm(syncFields, baseline));
      if (cfg?.targetField) setTargetField(cfg.targetField);
      setBaselineLoaded(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load live KPIs');
    } finally {
      setSyncing(false);
    }
  }, [fields, postPredict, predictConfig]);

  const runPredict = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const features: Record<string, number> = {};
      fields.forEach(({ key }) => {
        const raw = values[key];
        if (raw !== '' && raw != null) {
          features[key] = Number(raw) || 0;
        }
      });
      const data = await postPredict({
        features,
        prompt: scenario.trim(),
        target_field: targetField,
      });
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  }, [fields, postPredict, scenario, targetField, values]);

  if (!definition) return null;

  return (
    <Box sx={{ maxWidth: 1000, mx: 'auto', pb: 10 }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
        <TrendingUpIcon color="primary" />
        <Typography variant="h5" fontWeight={700}>Predict</Typography>
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        {predictConfig?.description ||
          'Forecasts use averages from your uploaded operational data. Choose the column to predict, adjust the other inputs, then run.'}
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={5}>
          <Card variant="outlined">
            <CardContent>
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                <Typography variant="subtitle1" fontWeight={600}>Inputs</Typography>
                <Button
                  size="small"
                  startIcon={<SyncIcon />}
                  onClick={syncFromLiveData}
                  disabled={syncing}
                >
                  {syncing ? 'Syncing…' : 'Sync from data'}
                </Button>
              </Stack>
              <Stack spacing={2}>
                <FormControl size="small" fullWidth>
                  <InputLabel id="predict-target-label">Column to predict</InputLabel>
                  <Select
                    labelId="predict-target-label"
                    label="Column to predict"
                    value={targetField}
                    onChange={(e) => setTargetField(String(e.target.value))}
                  >
                    {fields.map((f) => (
                      <MenuItem key={f.key} value={f.key}>
                        {f.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <TextField
                  label="Scenario prompt (optional)"
                  placeholder="What if defect rate rises 10%?"
                  size="small"
                  multiline
                  minRows={2}
                  value={scenario}
                  onChange={(e) => setScenario(e.target.value)}
                />
                {inputFields.map((f) => (
                  <TextField
                    key={f.key}
                    label={fieldLabel(f)}
                    type="number"
                    size="small"
                    placeholder={baselineLoaded ? undefined : 'Sync from data first'}
                    value={values[f.key] ?? ''}
                    onChange={(e) => setValues((v) => ({ ...v, [f.key]: e.target.value }))}
                  />
                ))}
                <Button variant="contained" onClick={runPredict} disabled={loading || !baselineLoaded || !targetField}>
                  {loading ? 'Running forecast…' : 'Predict'}
                </Button>
                {!baselineLoaded ? (
                  <Typography variant="caption" color="text.secondary">
                    Sync from data first so inputs match your app&apos;s operational columns.
                  </Typography>
                ) : null}
                {error ? <Typography color="error" variant="body2">{error}</Typography> : null}
              </Stack>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={7}>
          <Card variant="outlined" sx={{ minHeight: 320 }}>
            <CardContent>
              <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>Forecast indicators</Typography>
              {loading ? (
                <Typography variant="body2" color="text.secondary">Computing forecast…</Typography>
              ) : !result ? (
                <Typography variant="body2" color="text.secondary">
                  Choose a target column, fill the other fields, and click Predict.
                </Typography>
              ) : (
                <PredictionIndicators result={result} />
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
