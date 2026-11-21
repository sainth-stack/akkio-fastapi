export function getStoredUser() {
  try {
    return JSON.parse(localStorage.getItem('user') || 'null');
  } catch {
    return null;
  }
}

export function isAuthenticated() {
  return Boolean(localStorage.getItem('access_token'));
}

export function getUserPermissions() {
  const user = getStoredUser();
  if (!user?.roles) return new Set();
  const perms = new Set();
  user.roles.forEach((role) => {
    (role.permissions || []).forEach((p) => perms.add(p));
  });
  return perms;
}

export function hasPermission(feature) {
  const perms = getUserPermissions();
  if (perms.size === 0) return true;
  return (
    perms.has(`${feature}_read`) ||
    perms.has(`${feature}_write`) ||
    [...perms].some((p) => p.startsWith(`${feature}_`))
  );
}

export function hasAdminAccess() {
  return hasPermission('admin');
}

export function hasWritePermission(feature) {
  const perms = getUserPermissions();
  if (perms.size === 0) return true;
  return perms.has(`${feature}_write`);
}

export function clearAuthStorage() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('user');
  localStorage.removeItem('email');
  localStorage.removeItem('username');
  localStorage.removeItem('_id');
}
