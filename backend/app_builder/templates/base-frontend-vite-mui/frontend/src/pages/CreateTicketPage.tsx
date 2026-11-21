import { useState } from 'react';
import {
  Alert,
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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material';
import AddCircleIcon from '@mui/icons-material/AddCircle';
import { useAppDefinition } from '../renderer/useAppDefinition';
import { getAkkioAccessToken, getAkkioApiBase } from '../renderer/akkioApi';

type SimilarRow = Record<string, unknown>;

const CATEGORIES = ['Materials Management', 'Sales & Distribution', 'Finance', 'PP', 'General'];
const COMPONENTS = ['IDoc', 'BAPI', 'RFC', 'User exit', 'Workflow', 'Authorization', 'Other'];
const PRIORITIES = ['Low', 'Medium', 'High'];

export default function CreateTicketPage() {
  const { definition, applyDefinition } = useAppDefinition();
  const apiBase = getAkkioApiBase();
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('Materials Management');
  const [component, setComponent] = useState('IDoc');
  const [priority, setPriority] = useState('Medium');
  const [reportedBy, setReportedBy] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [similar, setSimilar] = useState<SimilarRow[]>([]);
  const [createdId, setCreatedId] = useState<string | null>(null);

  const submit = async () => {
    if (!apiBase) {
      setError('Open this app from the Akkio host to save tickets.');
      return;
    }
    if (!title.trim()) {
      setError('Title is required.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      const token = getAkkioAccessToken();
      if (token) headers.Authorization = `Bearer ${token}`;
      const res = await fetch(`${apiBase}/tickets`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          title: title.trim(),
          description: description.trim(),
          category,
          component,
          priority,
          reported_by: reportedBy.trim() || 'App user',
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(String(data.detail || 'Could not create ticket'));
      setCreatedId(String(data.ticket?.ticket_id || ''));
      setSimilar(Array.isArray(data.similar) ? data.similar : []);
      if (data.definition?.data) {
        applyDefinition({ ...definition!, ...data.definition });
      }
      setTitle('');
      setDescription('');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Create failed');
    } finally {
      setLoading(false);
    }
  };

  if (!definition) return null;

  return (
    <Box sx={{ maxWidth: 960, mx: 'auto', pb: 8 }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
        <AddCircleIcon color="primary" />
        <Typography variant="h5" fontWeight={700}>
          Create Ticket
        </Typography>
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Log a new SAP service incident. It is saved at the top of the ticket list and matched against
        historical tickets.
      </Typography>

      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <Stack spacing={2}>
            <TextField
              label="Title"
              required
              fullWidth
              size="small"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
            <TextField
              label="Description"
              fullWidth
              multiline
              minRows={3}
              size="small"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Category</InputLabel>
                  <Select label="Category" value={category} onChange={(e) => setCategory(String(e.target.value))}>
                    {CATEGORIES.map((c) => (
                      <MenuItem key={c} value={c}>
                        {c}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>SAP component</InputLabel>
                  <Select label="SAP component" value={component} onChange={(e) => setComponent(String(e.target.value))}>
                    {COMPONENTS.map((c) => (
                      <MenuItem key={c} value={c}>
                        {c}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Priority</InputLabel>
                  <Select label="Priority" value={priority} onChange={(e) => setPriority(String(e.target.value))}>
                    {PRIORITIES.map((c) => (
                      <MenuItem key={c} value={c}>
                        {c}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
            <TextField
              label="Reported by"
              size="small"
              value={reportedBy}
              onChange={(e) => setReportedBy(e.target.value)}
            />
            {error ? <Alert severity="error">{error}</Alert> : null}
            {createdId ? (
              <Alert severity="success">Ticket {createdId} created — shown at the top of the ticket list.</Alert>
            ) : null}
            <Button variant="contained" onClick={() => void submit()} disabled={loading}>
              {loading ? 'Saving…' : 'Create ticket'}
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {similar.length > 0 ? (
        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>
              Similar historical tickets
            </Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Match</TableCell>
                  <TableCell>ID</TableCell>
                  <TableCell>Title</TableCell>
                  <TableCell>Category</TableCell>
                  <TableCell>Component</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {similar.map((row) => (
                  <TableRow key={String(row.ticket_id)}>
                    <TableCell>{Math.round(Number(row.similarity_score || 0) * 100)}%</TableCell>
                    <TableCell>{String(row.ticket_id)}</TableCell>
                    <TableCell>{String(row.title)}</TableCell>
                    <TableCell>{String(row.category)}</TableCell>
                    <TableCell>{String(row.component)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ) : null}
    </Box>
  );
}
