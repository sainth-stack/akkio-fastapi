/**
 * Component Kit — barrel export.
 * Import everything from here: import { AppShell, DataTable, StatCard } from '../components/ui';
 */

// Layout & Navigation
export { default as AppShell } from './AppShell';
export type { AppShellProps, NavItem, BreadcrumbItem } from './AppShell';

// Content Layout
export { default as PageHeader } from './PageHeader';
export type { PageHeaderProps, PageHeaderBreadcrumb } from './PageHeader';

// Primitives
export { default as Button } from './Button';
export type { ButtonProps } from './Button';

export { default as Card } from './Card';
export type { CardProps } from './Card';

// Data Display
export { default as StatCard } from './StatCard';
export type { StatCardProps, StatCardTrend } from './StatCard';

export { default as DataTable } from './DataTable';
export type { DataTableProps, Column } from './DataTable';

export { default as StatusChip } from './StatusChip';
export type { StatusChipProps } from './StatusChip';

// Forms
export { default as FormField } from './FormField';
export type { FormFieldProps } from './FormField';

// Overlays
export { default as Modal } from './Modal';
export type { ModalProps, ModalMaxWidth } from './Modal';

export { default as ConfirmDialog } from './ConfirmDialog';
export type { ConfirmDialogProps } from './ConfirmDialog';

export { default as Toast } from './Toast';
export type { ToastProps, ToastSeverity } from './Toast';

// Feedback States
export { default as EmptyState } from './EmptyState';
export type { EmptyStateProps } from './EmptyState';

export { default as LoadingState } from './LoadingState';
export type { LoadingStateProps } from './LoadingState';

export { default as ErrorState } from './ErrorState';
export type { ErrorStateProps } from './ErrorState';

// Navigation Tabs
export { default as Tabs } from './Tabs';
export type { TabsProps, TabItem } from './Tabs';

// Charts
export { default as LineChart } from './charts/LineChart';
export type { LineChartProps, LineChartSeries } from './charts/LineChart';

export { default as BarChart } from './charts/BarChart';
export type { BarChartProps, BarChartSeries } from './charts/BarChart';

export { default as PieChart } from './charts/PieChart';
export type { PieChartProps, PieChartDataItem } from './charts/PieChart';
