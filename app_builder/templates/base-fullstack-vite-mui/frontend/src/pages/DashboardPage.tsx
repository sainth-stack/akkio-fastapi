import { useQuery } from '@tanstack/react-query';
import { Box, Typography } from '@mui/material';
import KPICard from '../components/KPICard';
import { apiFetch } from '../api/client';

type DashboardKpis = {
  total_lots?: number;
  pending_inspections?: number;
  released?: number;
  held?: number;
  rejected?: number;
  open_capa?: number;
};

export default function DashboardPage() {
  const { data, isLoading } = useQuery({
    queryKey: ['dashboard'],
    queryFn: () => apiFetch<DashboardKpis>('/api/dashboard/kpis'),
  });

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 2 }}>Dashboard</Typography>
      <Box
        sx={{
          display: 'grid',
          gap: 2,
          gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: 'repeat(3, 1fr)', lg: 'repeat(6, 1fr)' },
        }}
      >
        <KPICard label="Incoming lots" value={data?.total_lots ?? (isLoading ? '—' : 0)} />
        <KPICard label="Pending inspections" value={data?.pending_inspections ?? 0} />
        <KPICard label="Released" value={data?.released ?? 0} color="success.main" />
        <KPICard label="Held" value={data?.held ?? 0} color="warning.main" />
        <KPICard label="Rejected" value={data?.rejected ?? 0} color="error.main" />
        <KPICard label="Open CAPA" value={data?.open_capa ?? 0} />
      </Box>
    </Box>
  );
}
