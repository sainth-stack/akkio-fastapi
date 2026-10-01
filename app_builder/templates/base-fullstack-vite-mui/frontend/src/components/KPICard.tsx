import { Card, CardContent, Typography } from '@mui/material';

type Props = {
  label: string;
  value: string | number;
  color?: string;
};

export default function KPICard({ label, value, color }: Props) {
  return (
    <Card>
      <CardContent>
        <Typography variant="caption" color="text.secondary">{label}</Typography>
        <Typography variant="h5" sx={{ fontWeight: 700, color: color || 'text.primary' }}>{value}</Typography>
      </CardContent>
    </Card>
  );
}
