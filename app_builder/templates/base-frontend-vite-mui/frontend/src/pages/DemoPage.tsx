/**
 * DemoPage — acceptance test page that renders every component from the kit.
 * This page is included in the default template App.tsx so that:
 *   • `tsc --noEmit` validates all typed props
 *   • `vite build` succeeds with zero errors
 *   • the browser renders a live preview of the entire component library
 */
import React, { useState } from 'react';
import { Box, Grid, Stack, Typography } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import DashboardIcon from '@mui/icons-material/Dashboard';
import PeopleIcon from '@mui/icons-material/People';
import {
  AppShell,
  BarChart,
  Button,
  Card,
  ConfirmDialog,
  DataTable,
  EmptyState,
  ErrorState,
  FormField,
  LineChart,
  LoadingState,
  Modal,
  PageHeader,
  PieChart,
  StatCard,
  StatusChip,
  Tabs,
  Toast,
} from '../components/ui';

// ---------------------------------------------------------------------------
// Sample data
// ---------------------------------------------------------------------------

interface Row {
  id: string;
  name: string;
  status: string;
  revenue: number;
  date: string;
  [key: string]: unknown;
}

const ROWS: Row[] = [
  { id: '1', name: 'Acme Corp', status: 'active', revenue: 45200, date: '2026-01-10' },
  { id: '2', name: 'Globex Inc', status: 'pending', revenue: 31800, date: '2026-02-14' },
  { id: '3', name: 'Umbrella Ltd', status: 'inactive', revenue: 19500, date: '2026-03-05' },
  { id: '4', name: 'Initech', status: 'active', revenue: 67100, date: '2026-04-22' },
  { id: '5', name: 'Massive Dynamic', status: 'failed', revenue: 8200, date: '2026-05-30' },
];

const CHART_DATA = [
  { name: 'Jan', revenue: 4000, cost: 2400 },
  { name: 'Feb', revenue: 3000, cost: 1398 },
  { name: 'Mar', revenue: 6000, cost: 9800 },
  { name: 'Apr', revenue: 8000, cost: 3908 },
  { name: 'May', revenue: 5000, cost: 4800 },
  { name: 'Jun', revenue: 9000, cost: 3800 },
];

const PIE_DATA = [
  { name: 'Enterprise', value: 400 },
  { name: 'SMB', value: 300 },
  { name: 'Consumer', value: 200 },
  { name: 'Partner', value: 100 },
];

// ---------------------------------------------------------------------------
// DemoPage
// ---------------------------------------------------------------------------

export default function DemoPage() {
  const [modalOpen, setModalOpen] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [toastOpen, setToastOpen] = useState(false);
  const [name, setName] = useState('');
  const [nameError, setNameError] = useState('');

  const handleValidate = () => {
    if (!name.trim()) {
      setNameError('Name is required');
    } else {
      setNameError('');
      setToastOpen(true);
    }
  };

  return (
    <Box>
      {/* PageHeader */}
      <PageHeader
        title="Component Kit Demo"
        subtitle="Every component from the kit rendered on one page."
        actions={
          <>
            <Button variant="outlined" onClick={() => setConfirmOpen(true)} startIcon={<DeleteIcon />}>
              Delete
            </Button>
            <Button variant="contained" onClick={() => setModalOpen(true)} startIcon={<AddIcon />}>
              New Item
            </Button>
          </>
        }
        breadcrumbs={[{ label: 'Home', href: '/' }, { label: 'Demo' }]}
      />

      {/* StatCards */}
      <Grid container spacing={2} mb={3}>
        {[
          { label: 'Total Revenue', value: '$174,800', color: 'primary', trend: { value: 12, label: 'vs last month' } },
          { label: 'Active Clients', value: '2,048', color: 'success', trend: { value: 5 } },
          { label: 'Pending Tasks', value: '47', color: 'warning', trend: { value: -8, label: 'vs last week' } },
          { label: 'Errors', value: '3', color: 'error', trend: { value: -50 } },
        ].map((s) => (
          <Grid item xs={12} sm={6} md={3} key={s.label}>
            <StatCard
              label={s.label}
              value={s.value}
              color={s.color as 'primary' | 'success' | 'warning' | 'error'}
              trend={s.trend}
              icon={<DashboardIcon />}
            />
          </Grid>
        ))}
      </Grid>

      {/* Charts in Tabs */}
      <Card title="Analytics" sx={{ mb: 3 }}>
        <Tabs
          tabs={[
            {
              label: 'Line',
              content: (
                <LineChart
                  data={CHART_DATA}
                  series={[
                    { dataKey: 'revenue', name: 'Revenue' },
                    { dataKey: 'cost', name: 'Cost' },
                  ]}
                />
              ),
            },
            {
              label: 'Bar',
              content: (
                <BarChart
                  data={CHART_DATA}
                  series={[
                    { dataKey: 'revenue', name: 'Revenue' },
                    { dataKey: 'cost', name: 'Cost' },
                  ]}
                />
              ),
            },
            {
              label: 'Pie',
              content: (
                <PieChart data={PIE_DATA} innerRadius={50} outerRadius={100} showLabels />
              ),
            },
          ]}
        />
      </Card>

      {/* DataTable */}
      <Card title="Clients" sx={{ mb: 3 }}>
        <DataTable<Row>
          columns={[
            { field: 'name', header: 'Name', sortable: true },
            { field: 'status', header: 'Status', renderCell: (v) => <StatusChip status={String(v)} /> },
            { field: 'revenue', header: 'Revenue', sortable: true, align: 'right',
              renderCell: (v) => `$${Number(v).toLocaleString()}` },
            { field: 'date', header: 'Date', sortable: true },
          ]}
          rows={ROWS}
          rowKey="id"
        />
      </Card>

      {/* Feedback states */}
      <Grid container spacing={2} mb={3}>
        <Grid item xs={12} md={4}>
          <Card title="Empty State">
            <EmptyState
              icon={<PeopleIcon sx={{ fontSize: 48, color: 'text.disabled' }} />}
              title="No contacts yet"
              description="Add your first contact to get started."
              action={<Button variant="contained" size="small">Add Contact</Button>}
            />
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card title="Loading State">
            <LoadingState message="Fetching data…" />
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card title="Error State">
            <ErrorState
              title="Failed to load"
              message="Check your connection and try again."
              onRetry={() => alert('retry')}
            />
          </Card>
        </Grid>
      </Grid>

      {/* FormField */}
      <Card title="Form Field" sx={{ mb: 3, maxWidth: 480 }}>
        <Stack spacing={2}>
          <FormField
            label="Name"
            name="name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            error={nameError}
            placeholder="Enter your name"
            required
          />
          <FormField label="Email" name="email" value="" onChange={() => {}} type="email" />
          <Button variant="contained" onClick={handleValidate}>
            Validate &amp; Show Toast
          </Button>
        </Stack>
      </Card>

      {/* StatusChips */}
      <Card title="Status Chips" sx={{ mb: 3 }}>
        <Stack direction="row" flexWrap="wrap" gap={1}>
          {['active', 'inactive', 'pending', 'processing', 'failed', 'approved', 'rejected', 'draft'].map(
            (s) => <StatusChip key={s} status={s} />,
          )}
        </Stack>
      </Card>

      {/* Typography note */}
      <Typography variant="caption" color="text.secondary">
        All components above are from src/components/ui/index.ts
      </Typography>

      {/* ── Overlays ─────────────────────────────────────────────────────── */}

      <Modal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        title="Create New Item"
        actions={
          <>
            <Button variant="outlined" onClick={() => setModalOpen(false)}>Cancel</Button>
            <Button variant="contained" onClick={() => setModalOpen(false)}>Save</Button>
          </>
        }
      >
        <Stack spacing={2} pt={1}>
          <FormField label="Item Name" name="itemName" value="" onChange={() => {}} />
          <FormField label="Description" name="desc" value="" onChange={() => {}} multiline rows={3} />
        </Stack>
      </Modal>

      <ConfirmDialog
        open={confirmOpen}
        onClose={() => setConfirmOpen(false)}
        onConfirm={() => { setConfirmOpen(false); setToastOpen(true); }}
        title="Delete Item"
        message="This action cannot be undone. Are you sure you want to delete this item?"
        confirmLabel="Delete"
        dangerous
      />

      <Toast
        open={toastOpen}
        onClose={() => setToastOpen(false)}
        message="Action completed successfully!"
        severity="success"
      />
    </Box>
  );
}
