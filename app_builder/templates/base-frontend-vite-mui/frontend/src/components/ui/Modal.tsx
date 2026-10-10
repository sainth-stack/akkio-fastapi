import React from 'react';
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  IconButton,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

export type ModalMaxWidth = 'xs' | 'sm' | 'md' | 'lg' | 'xl';

export interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  maxWidth?: ModalMaxWidth;
  actions?: React.ReactNode;
  children: React.ReactNode;
  disableClose?: boolean;
}

/**
 * Modal — Dialog wrapper with title bar, close button, scrollable content,
 * and a sticky action footer.
 */
export default function Modal({
  open,
  onClose,
  title,
  maxWidth = 'sm',
  actions,
  children,
  disableClose = false,
}: ModalProps) {
  return (
    <Dialog
      open={open}
      onClose={disableClose ? undefined : onClose}
      maxWidth={maxWidth}
      fullWidth
      PaperProps={{ sx: { borderRadius: 3 } }}
    >
      <DialogTitle
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          pb: 1,
        }}
      >
        {title}
        {!disableClose && (
          <IconButton size="small" onClick={onClose} aria-label="close">
            <CloseIcon fontSize="small" />
          </IconButton>
        )}
      </DialogTitle>
      <Divider />
      <DialogContent sx={{ pt: 2 }}>{children}</DialogContent>
      {actions && (
        <>
          <Divider />
          <DialogActions sx={{ px: 3, py: 1.5 }}>{actions}</DialogActions>
        </>
      )}
    </Dialog>
  );
}
