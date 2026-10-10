import React, { useMemo, useState } from 'react';
import {
  Box,
  InputAdornment,
  Paper,
  Skeleton,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow,
  TableSortLabel,
  TextField,
  Typography,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Column<T extends Record<string, unknown>> {
  field: keyof T;
  header: string;
  sortable?: boolean;
  width?: string | number;
  align?: 'left' | 'center' | 'right';
  renderCell?: (value: T[keyof T], row: T) => React.ReactNode;
}

export interface DataTableProps<T extends Record<string, unknown>> {
  columns: Column<T>[];
  rows: T[];
  rowKey?: keyof T;
  searchable?: boolean;
  pageSizeOptions?: number[];
  defaultPageSize?: number;
  emptyMessage?: string;
  loading?: boolean;
  onRowClick?: (row: T) => void;
}

type Order = 'asc' | 'desc';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function stableSort<T>(arr: T[], compareFn: (a: T, b: T) => number): T[] {
  return arr
    .map((item, index) => ({ item, index }))
    .sort((a, b) => {
      const cmp = compareFn(a.item, b.item);
      return cmp !== 0 ? cmp : a.index - b.index;
    })
    .map(({ item }) => item);
}

function getComparator<T extends Record<string, unknown>>(
  order: Order,
  orderBy: keyof T,
): (a: T, b: T) => number {
  return (a, b) => {
    const av = a[orderBy] ?? '';
    const bv = b[orderBy] ?? '';
    const result = String(av).localeCompare(String(bv), undefined, { numeric: true });
    return order === 'asc' ? result : -result;
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function DataTable<T extends Record<string, unknown>>({
  columns,
  rows,
  rowKey,
  searchable = true,
  pageSizeOptions = [10, 25, 50],
  defaultPageSize = 10,
  emptyMessage = 'No data available',
  loading = false,
  onRowClick,
}: DataTableProps<T>) {
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(defaultPageSize);
  const [orderBy, setOrderBy] = useState<keyof T | ''>('');
  const [order, setOrder] = useState<Order>('asc');

  const handleSort = (field: keyof T) => {
    if (orderBy === field) {
      setOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setOrderBy(field);
      setOrder('asc');
    }
    setPage(0);
  };

  const filtered = useMemo(() => {
    if (!search.trim()) return rows;
    const q = search.toLowerCase();
    return rows.filter((row) =>
      columns.some((col) => String(row[col.field] ?? '').toLowerCase().includes(q)),
    );
  }, [rows, search, columns]);

  const sorted = useMemo(() => {
    if (!orderBy) return filtered;
    return stableSort(filtered, getComparator(order, orderBy));
  }, [filtered, order, orderBy]);

  const paginated = sorted.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);

  return (
    <Paper sx={{ borderRadius: 3, overflow: 'hidden' }}>
      {searchable && (
        <Box px={2} pt={2} pb={1}>
          <TextField
            size="small"
            placeholder="Search…"
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(0); }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" color="action" />
                </InputAdornment>
              ),
            }}
            sx={{ width: { xs: '100%', sm: 300 } }}
          />
        </Box>
      )}

      <TableContainer>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              {columns.map((col) => (
                <TableCell
                  key={String(col.field)}
                  align={col.align ?? 'left'}
                  style={{ width: col.width, whiteSpace: 'nowrap' }}
                >
                  {col.sortable !== false ? (
                    <TableSortLabel
                      active={orderBy === col.field}
                      direction={orderBy === col.field ? order : 'asc'}
                      onClick={() => handleSort(col.field)}
                    >
                      {col.header}
                    </TableSortLabel>
                  ) : (
                    col.header
                  )}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {loading ? (
              Array.from({ length: rowsPerPage }).map((_, i) => (
                <TableRow key={i}>
                  {columns.map((col) => (
                    <TableCell key={String(col.field)}>
                      <Skeleton variant="text" />
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : paginated.length === 0 ? (
              <TableRow>
                <TableCell colSpan={columns.length} align="center" sx={{ py: 6 }}>
                  <Stack alignItems="center" spacing={1}>
                    <Typography variant="body2" color="text.secondary">
                      {emptyMessage}
                    </Typography>
                  </Stack>
                </TableCell>
              </TableRow>
            ) : (
              paginated.map((row, ri) => {
                const key = rowKey ? String(row[rowKey]) : ri;
                return (
                  <TableRow
                    key={key}
                    hover
                    onClick={() => onRowClick?.(row)}
                    sx={{ cursor: onRowClick ? 'pointer' : 'default' }}
                  >
                    {columns.map((col) => (
                      <TableCell key={String(col.field)} align={col.align ?? 'left'}>
                        {col.renderCell
                          ? col.renderCell(row[col.field], row)
                          : String(row[col.field] ?? '')}
                      </TableCell>
                    ))}
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <TablePagination
        component="div"
        count={filtered.length}
        page={page}
        onPageChange={(_, p) => setPage(p)}
        rowsPerPage={rowsPerPage}
        onRowsPerPageChange={(e) => { setRowsPerPage(Number(e.target.value)); setPage(0); }}
        rowsPerPageOptions={pageSizeOptions}
      />
    </Paper>
  );
}
