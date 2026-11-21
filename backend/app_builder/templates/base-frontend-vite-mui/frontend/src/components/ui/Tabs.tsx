import React, { useState } from 'react';
import {
  Box,
  Tab,
  Tabs as MuiTabs,
} from '@mui/material';

export interface TabItem {
  label: string;
  content: React.ReactNode;
  disabled?: boolean;
  icon?: React.ReactElement;
}

export interface TabsProps {
  tabs: TabItem[];
  defaultTab?: number;
  onChange?: (index: number) => void;
  variant?: 'standard' | 'scrollable' | 'fullWidth';
}

/**
 * Tabs — MUI Tabs with automatic panel switching.
 * Named `AppTabs` internally to avoid conflict with MUI's `Tabs`.
 */
export default function Tabs({
  tabs,
  defaultTab = 0,
  onChange,
  variant = 'standard',
}: TabsProps) {
  const [active, setActive] = useState(defaultTab);

  const handleChange = (_: React.SyntheticEvent, value: number) => {
    setActive(value);
    onChange?.(value);
  };

  return (
    <Box>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <MuiTabs
          value={active}
          onChange={handleChange}
          variant={variant}
          allowScrollButtonsMobile
        >
          {tabs.map((tab, i) => (
            <Tab
              key={i}
              label={tab.label}
              icon={tab.icon}
              iconPosition="start"
              disabled={tab.disabled}
              sx={{ minHeight: 48, textTransform: 'none', fontWeight: 500 }}
            />
          ))}
        </MuiTabs>
      </Box>
      <Box pt={2}>{tabs[active]?.content}</Box>
    </Box>
  );
}
