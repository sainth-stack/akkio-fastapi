import {
  Alert,
  Box,
  Grid,
  LinearProgress,
  Paper,
  Stack,
  Typography,
} from '@mui/material';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import StatCard from '../ui/StatCard';

export type PredictionIndicator = {
  id: string;
  kind: 'metric' | 'status' | 'gauge';
  label: string;
  value: string | number;
  unit?: string;
  baseline?: number;
  delta_pct?: number | null;
  trend?: 'up' | 'down' | 'flat';
  severity?: 'error' | 'warning' | 'success' | 'info';
  confidence?: number;
};

export type LivePrediction = {
  predicted_downtime_hours_next_period: number;
  predicted_mtbf_hours: number;
  predicted_production_output: number;
  risk_level: string;
  confidence: number;
  drivers: string[];
  recommendation: string;
  indicators?: PredictionIndicator[];
  scenario_notes?: string[];
  prompt?: string;
  computed_at?: string;
  baseline?: Record<string, number>;
  scenario_inputs?: Record<string, number>;
  inputs_used?: Record<string, number>;
};

function severityToColor(sev?: string): 'error' | 'warning' | 'success' | 'info' | 'primary' {
  if (sev === 'error') return 'error';
  if (sev === 'warning') return 'warning';
  if (sev === 'success') return 'success';
  if (sev === 'info') return 'info';
  return 'primary';
}

export default function PredictionIndicators({ result }: { result: LivePrediction }) {
  const metrics = (result.indicators || []).filter((i) => i.kind === 'metric');
  const status = (result.indicators || []).find((i) => i.kind === 'status');
  const gauge = (result.indicators || []).find((i) => i.kind === 'gauge');

  return (
    <Stack spacing={2}>
      {status ? (
        <Alert severity={severityToColor(status.severity)} icon={false}>
          <Typography variant="subtitle2" fontWeight={700}>
            {status.label}: {status.value}
          </Typography>
          {status.confidence != null ? (
            <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
              Forecast confidence {(status.confidence * 100).toFixed(0)}%
            </Typography>
          ) : null}
        </Alert>
      ) : null}

      {gauge ? (
        <Box>
          <Stack direction="row" justifyContent="space-between" sx={{ mb: 0.5 }}>
            <Typography variant="caption" color="text.secondary" fontWeight={600}>
              {gauge.label}
            </Typography>
            <Typography variant="caption" fontWeight={700}>
              {gauge.value}{gauge.unit || ''}
            </Typography>
          </Stack>
          <LinearProgress
            variant="determinate"
            value={Math.min(100, Number(gauge.value) || 0)}
            color={severityToColor(gauge.severity)}
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Box>
      ) : null}

      <Grid container spacing={2}>
        {metrics.map((m) => {
          const delta = m.delta_pct;
          const trend =
            delta != null
              ? { value: delta, label: `${delta > 0 ? '+' : ''}${delta}% vs your inputs` }
              : undefined;
          const display =
            typeof m.value === 'number'
              ? `${m.value}${m.unit ? ` ${m.unit}` : ''}`
              : String(m.value);
          return (
            <Grid item xs={12} sm={6} key={m.id}>
              <StatCard
                label={m.label}
                value={display}
                trend={trend}
                color={severityToColor(m.severity)}
              />
            </Grid>
          );
        })}
      </Grid>

      <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
        <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1 }}>Drivers</Typography>
        <Stack spacing={0.5}>
          {result.drivers.map((d) => (
            <Stack key={d} direction="row" spacing={1} alignItems="center">
              {d.toLowerCase().includes('high') || d.toLowerCase().includes('below') ? (
                <TrendingDownIcon fontSize="small" color="warning" />
              ) : (
                <TrendingUpIcon fontSize="small" color="success" />
              )}
              <Typography variant="body2">{d}</Typography>
            </Stack>
          ))}
        </Stack>
        <Typography variant="body2" sx={{ mt: 1.5 }}>{result.recommendation}</Typography>
      </Paper>

      {result.scenario_notes && result.scenario_notes.length > 0 ? (
        <Typography variant="caption" color="text.secondary">
          Scenario: {result.scenario_notes.join(' · ')}
        </Typography>
      ) : null}
      {result.inputs_used ? (
        <Typography variant="caption" color="text.secondary" display="block">
          Inputs used: MTBF {result.inputs_used.mtbf_hours}h · downtime {result.inputs_used.downtime_hours}h ·
          load {result.inputs_used.machine_load_pct}% · WOs {result.inputs_used.open_work_orders}
        </Typography>
      ) : null}
      {result.computed_at ? (
        <Typography variant="caption" color="text.secondary">
          Computed {new Date(result.computed_at).toLocaleString()}
        </Typography>
      ) : null}
    </Stack>
  );
}
