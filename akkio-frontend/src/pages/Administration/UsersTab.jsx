import React, { useCallback, useEffect, useState } from 'react';
import { Button, Card, Form, Input, Modal, Select, Space, Table, Tag, message } from 'antd';
import { FaPlus, FaEdit, FaTrashAlt } from 'react-icons/fa';
import api from '../../utils/api';
import { hasWritePermission } from '../../utils/auth';

export default function UsersTab() {
  const canWrite = hasWritePermission('admin');
  const [users, setUsers] = useState([]);
  const [roles, setRoles] = useState([]);
  const [organizations, setOrganizations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState(null);
  const [form] = Form.useForm();

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [usersRes, rolesRes, orgsRes] = await Promise.all([
        api.get('/admin/users'),
        api.get('/admin/roles'),
        api.get('/admin/organizations'),
      ]);
      setUsers(usersRes.data || []);
      setRoles(rolesRes.data || []);
      setOrganizations(orgsRes.data || []);
    } catch (e) {
      message.error('Failed to load users');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const openCreate = () => {
    setEditing(null);
    form.resetFields();
    form.setFieldsValue({ app: 'akkio', role_ids: [] });
    setModalOpen(true);
  };

  const openEdit = (record) => {
    setEditing(record);
    form.setFieldsValue({
      name: record.name,
      email: record.email,
      username: record.username,
      app: record.app,
      organization_id: record.organization?.id,
      role_ids: (record.roles || []).map((r) => r.id),
    });
    setModalOpen(true);
  };

  const handleDelete = async (record) => {
    if (!window.confirm(`Delete user "${record.email}"?`)) return;
    try {
      await api.delete(`/admin/users/${record.id}`);
      message.success('User deleted');
      loadData();
    } catch (e) {
      message.error(e?.response?.data?.detail || 'Delete failed');
    }
  };

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      const payload = { ...values };
      if (editing && !payload.password) {
        delete payload.password;
      }
      if (editing) {
        await api.put(`/admin/users/${editing.id}`, payload);
        message.success('User updated');
      } else {
        await api.post('/admin/users', payload);
        message.success('User created');
      }
      setModalOpen(false);
      loadData();
    } catch (e) {
      if (e?.response?.data?.detail) {
        message.error(typeof e.response.data.detail === 'string' ? e.response.data.detail : 'Save failed');
      }
    }
  };

  const columns = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Email', dataIndex: 'email', key: 'email' },
    {
      title: 'Organization',
      key: 'organization',
      render: (_, row) => row.organization?.name || '—',
    },
    {
      title: 'Roles',
      key: 'roles',
      render: (_, row) => (
        <Space wrap size={[4, 4]}>
          {(row.roles || []).length
            ? row.roles.map((r) => <Tag key={r.id}>{r.name}</Tag>)
            : '—'}
        </Space>
      ),
    },
    ...(canWrite
      ? [{
          title: 'Actions',
          key: 'actions',
          width: 100,
          render: (_, row) => (
            <Space>
              <Button type="text" icon={<FaEdit />} onClick={() => openEdit(row)} />
              <Button type="text" danger icon={<FaTrashAlt />} onClick={() => handleDelete(row)} />
            </Space>
          ),
        }]
      : []),
  ];

  return (
    <div>
      <div className="admin-page-header">
        <div>
          <h1 className="admin-page-header__title">Users</h1>
          <p className="admin-page-header__subtitle">
            Manage workspace members, assign organizations, and control role-based access.
          </p>
        </div>
        {canWrite && (
          <Button type="primary" icon={<FaPlus />} onClick={openCreate} size="large">
            Add user
          </Button>
        )}
      </div>

      <Card className="admin-data-card" bordered={false}>
        <Table
          rowKey="id"
          loading={loading}
          dataSource={users}
          columns={columns}
          pagination={{ pageSize: 10, showSizeChanger: true, showTotal: (t) => `${t} users` }}
        />
      </Card>

      <Modal
        title={editing ? 'Edit user' : 'Create user'}
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        onOk={handleSubmit}
        okText={editing ? 'Save' : 'Create'}
        destroyOnClose
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="Name" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="email" label="Email" rules={[{ required: true, type: 'email' }]}>
            <Input disabled={!!editing} />
          </Form.Item>
          <Form.Item
            name="password"
            label={editing ? 'Password (leave blank to keep)' : 'Password'}
            rules={editing ? [] : [{ required: true, min: 8 }]}
          >
            <Input.Password />
          </Form.Item>
          <Form.Item name="username" label="Username">
            <Input />
          </Form.Item>
          <Form.Item name="organization_id" label="Organization">
            <Select allowClear placeholder="Select organization">
              {organizations.map((org) => (
                <Select.Option key={org.id} value={org.id}>{org.name}</Select.Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item name="role_ids" label="Roles">
            <Select mode="multiple" placeholder="Assign roles">
              {roles.map((role) => (
                <Select.Option key={role.id} value={role.id}>{role.name}</Select.Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item name="app" hidden>
            <Input />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
