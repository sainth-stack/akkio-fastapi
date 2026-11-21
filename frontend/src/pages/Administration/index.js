import React from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import UsersTab from './UsersTab';
import OrganizationsTab from './OrganizationsTab';
import RolesTab from './RolesTab';
import './administration.css';

export default function Administration() {
  return (
    <div className="administration-page">
      <Routes>
        <Route index element={<Navigate to="users" replace />} />
        <Route path="users" element={<UsersTab />} />
        <Route path="organizations" element={<OrganizationsTab />} />
        <Route path="roles" element={<RolesTab />} />
        <Route path="*" element={<Navigate to="users" replace />} />
      </Routes>
    </div>
  );
}
