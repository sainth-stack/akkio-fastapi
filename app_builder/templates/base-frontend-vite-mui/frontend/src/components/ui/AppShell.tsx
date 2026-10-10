import React, { useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import {
  AppBar,
  Box,
  Breadcrumbs,
  CssBaseline,
  Divider,
  Drawer,
  IconButton,
  Link,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import * as MuiIcons from '@mui/icons-material';

const DRAWER_WIDTH = 256;

export interface NavItem {
  label: string;
  path: string;
  icon?: string;
}

export interface BreadcrumbItem {
  label: string;
  href?: string;
}

export interface AppShellProps {
  navItems?: NavItem[];
  appName?: string;
  logo?: React.ReactNode;
  breadcrumbs?: BreadcrumbItem[];
}

function DynamicIcon({ name }: { name?: string }) {
  if (!name) return null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const IconComp = (MuiIcons as Record<string, any>)[name] as React.ElementType | undefined;
  return IconComp ? <IconComp fontSize="small" /> : null;
}

export default function AppShell({
  navItems = [],
  appName = 'Application',
  logo,
  breadcrumbs,
}: AppShellProps) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  const handleDrawerToggle = () => setMobileOpen((prev) => !prev);

  const drawerContent = (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Toolbar sx={{ px: 2 }}>
        {logo ?? (
          <Typography variant="h6" fontWeight={700} color="primary" noWrap>
            {appName}
          </Typography>
        )}
      </Toolbar>
      <Divider />
      <List sx={{ flex: 1, py: 1 }}>
        {navItems.map((item) => {
          const isActive =
            item.path === '/'
              ? location.pathname === '/'
              : location.pathname.startsWith(item.path);
          return (
            <ListItemButton
              key={item.path}
              component={NavLink}
              to={item.path}
              selected={isActive}
              onClick={() => isMobile && setMobileOpen(false)}
              sx={{ mx: 1, borderRadius: 2 }}
            >
              {item.icon && (
                <ListItemIcon sx={{ minWidth: 36 }}>
                  <DynamicIcon name={item.icon} />
                </ListItemIcon>
              )}
              <ListItemText
                primary={item.label}
                primaryTypographyProps={{ fontSize: 14, fontWeight: isActive ? 600 : 400 }}
              />
            </ListItemButton>
          );
        })}
      </List>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      <CssBaseline />

      {/* Top bar */}
      <AppBar
        position="fixed"
        color="inherit"
        elevation={0}
        sx={{
          width: { md: `calc(100% - ${DRAWER_WIDTH}px)` },
          ml: { md: `${DRAWER_WIDTH}px` },
          borderBottom: '1px solid',
          borderColor: 'divider',
          zIndex: theme.zIndex.drawer + 1,
        }}
      >
        <Toolbar>
          {isMobile && (
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}

          {breadcrumbs && breadcrumbs.length > 0 ? (
            <Breadcrumbs aria-label="breadcrumb">
              {breadcrumbs.map((bc, i) =>
                bc.href && i < breadcrumbs.length - 1 ? (
                  <Link key={i} underline="hover" color="inherit" href={bc.href} sx={{ fontSize: 14 }}>
                    {bc.label}
                  </Link>
                ) : (
                  <Typography key={i} color="text.primary" sx={{ fontSize: 14 }}>
                    {bc.label}
                  </Typography>
                ),
              )}
            </Breadcrumbs>
          ) : (
            <Typography variant="subtitle1" fontWeight={600}>
              {appName}
            </Typography>
          )}
        </Toolbar>
      </AppBar>

      {/* Sidebar — permanent on desktop, temporary drawer on mobile */}
      <Box
        component="nav"
        sx={{ width: { md: DRAWER_WIDTH }, flexShrink: { md: 0 } }}
      >
        {isMobile ? (
          <Drawer
            variant="temporary"
            open={mobileOpen}
            onClose={handleDrawerToggle}
            ModalProps={{ keepMounted: true }}
            sx={{
              '& .MuiDrawer-paper': {
                boxSizing: 'border-box',
                width: DRAWER_WIDTH,
              },
            }}
          >
            {drawerContent}
          </Drawer>
        ) : (
          <Drawer
            variant="permanent"
            sx={{
              '& .MuiDrawer-paper': {
                boxSizing: 'border-box',
                width: DRAWER_WIDTH,
                borderRight: '1px solid',
                borderColor: 'divider',
              },
            }}
            open
          >
            {drawerContent}
          </Drawer>
        )}
      </Box>

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: { xs: 2, sm: 3 },
          mt: 8,
          maxWidth: '100%',
          overflow: 'auto',
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
}
