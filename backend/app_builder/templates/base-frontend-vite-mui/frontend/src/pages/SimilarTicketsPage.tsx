import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { useAppDefinition } from '../renderer/useAppDefinition';
import { getAkkioAccessToken, getAkkioApiBase } from '../renderer/akkioApi';

type TicketRow = Record<string, unknown>;

function rowsFromDefinition(defn: { data?: Record<string, unknown> }): TicketRow[] {
  const rows = defn.data?.source_sap_tickets;
  return Array.isArray(rows) ? (rows as TicketRow[]) : [];
}

export default function SimilarTicketsPage() {
  const { definition } = useAppDefinition();
  const apiBase = getAkkioApiBase();
  const [tickets, setTickets] = useState<TicketRow[]>(() => (definition ? rowsFromDefinition(definition) : []));
  const [selectedId, setSelectedId] = useState('');
  const [similar, setSimilar] = useState<TicketRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!apiBase) {
      if (definition) setTickets(rowsFromDefinition(definition));
      return;
    }
    setError(null);
    try {
      const headers: Record<string, string> = {};
      const token = getAkkioAccessToken();
      if (token) headers.Authorization = `Bearer ${token}`;
      const res = await fetch(`${apiBase}/tickets`, { headers });
      const data = await res.json();
      if (!res.ok) throw new Error(String(data.detail || 'Could not load tickets'));
      const list = Array.isArray(data.tickets) ? data.tickets : [];
      setTickets(list);
      if (!selectedId && list[0]?.ticket_id) {
        setSelectedId(String(list[0].ticket_id));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Load failed');
      if (definition) setTickets(rowsFromDefinition(definition));
    }
  }, [apiBase, definition, selectedId]);

  const loadSimilar = useCallback(async () => {
    if (!apiBase || !selectedId) {
      setSimilar([]);
      return;
    }
    try {
      const headers: Record<string, string> = {};
      const token = getAkkioAccessToken();
      if (token) headers.Authorization = `Bearer ${token}`;
      const res = await fetch(
        `${apiBase}/tickets?ticket_id=${encodeURIComponent(selectedId)}`,
        { headers },
      );
      const data = await res.json();
      if (!res.ok) throw new Error(String(data.detail || 'Similar search failed'));
      setSimilar(Array.isArray(data.similar) ? data.similar : []);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Similar search failed');
    }
  }, [apiBase, selectedId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    void loadSimilar();
  }, [loadSimilar]);

  const allRows = useMemo(() => tickets, [tickets]);

  if (!definition) return null;

  return (
    <Box sx={{ maxWidth: 1100, mx: 'auto', pb: 8 }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
        <SearchIcon color="primary" />
        <Typography variant="h5" fontWeight={700}>
          Similar Tickets
        </Typography>
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        New tickets appear in the first row. Pick a reference ticket to see the closest historical matches.
      </Typography>

      {error ? <Alert severity="warning" sx={{ mb: 2 }}>{error}</Alert> : null}

      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <FormControl size="small" fullWidth sx={{ mb: 2, maxWidth: 480 }}>
            <InputLabel id="ref-ticket-label">Reference ticket</InputLabel>
            <Select
              labelId="ref-ticket-label"
              label="Reference ticket"
              value={selectedId}
              onChange={(e) => setSelectedId(String(e.target.value))}
            >
              {allRows.map((t) => (
                <MenuItem key={String(t.ticket_id)} value={String(t.ticket_id)}>
                  {String(t.ticket_id)} — {String(t.title)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1 }}>
            All tickets (newest first)
          </Typography>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>Title</TableCell>
                <TableCell>Category</TableCell>
                <TableCell>Component</TableCell>
                <TableCell>Priority</TableCell>
                <TableCell>Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {allRows.map((row, idx) => (
                <TableRow
                  key={String(row.ticket_id)}
                  selected={String(row.ticket_id) === selectedId}
                  sx={idx === 0 ? { bgcolor: 'action.hover' } : undefined}
                >
                  <TableCell>
                    {String(row.ticket_id)}
                    {idx === 0 ? <Chip size="small" label="Latest" sx={{ ml: 1 }} /> : null}
                  </TableCell>
                  <TableCell>{String(row.title)}</TableCell>
                  <TableCell>{String(row.category)}</TableCell>
                  <TableCell>{String(row.component)}</TableCell>
                  <TableCell>{String(row.priority)}</TableCell>
                  <TableCell>{String(row.status)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>
            Similar tickets
          </Typography>
          {!similar.length ? (
            <Typography variant="body2" color="text.secondary">
              Select a reference ticket to see matches.
            </Typography>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Match</TableCell>
                  <TableCell>ID</TableCell>
                  <TableCell>Title</TableCell>
                  <TableCell>Category</TableCell>
                  <TableCell>Component</TableCell>
                  <TableCell>Status</TableCell>
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
                    <TableCell>{String(row.status)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}
