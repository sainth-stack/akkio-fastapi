import React, { useCallback, useEffect, useState } from 'react';
import { Button, Card, Form, Input, Modal, Select, Space, Table, Tag, message } from 'antd';
import { FaPlus, FaEdit, FaTrashAlt } from 'react-icons/fa';
import api from '../../utils/api';
import { hasWritePermission } from '../../utils/auth';

const PERMISSION_OPTIONS = [
  'admin_read',
  'admin_write',
  'home_read',
  'reports_read',
  'reports_write',
];

const PERMISSION_COLORS = {
  admin_read: 'purple',
  admin_write: 'magenta',
  home_read: 'blue',
  reports_read: 'cyan',
  reports_write: 'geekblue',
};

export default function RolesTab() {
  const canWrite = hasWritePermission('admin');
  const [roles, setRoles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState(null);
  const [form] = Form.useForm();

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get('/admin/roles');
      setRoles(res.data || []);
    } catch (e) {
      message.error('Failed to load roles');
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
    form.setFieldsValue({ permissions: [] });
    setModalOpen(true);
  };

  const openEdit = (record) => {
    setEditing(record);
    form.setFieldsValue({ name: record.name, permissions: record.permissions || [] });
    setModalOpen(true);
  };

  const handleDelete = async (record) => {
    if (!window.confirm(`Delete role "${record.name}"?`)) return;
    try {
      await api.delete(`/admin/roles/${record.id}`);
      message.success('Role deleted');
      loadData();
    } catch (e) {
      message.error(e?.response?.data?.detail || 'Delete failed');
    }
  };

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      if (editing) {
        await api.put(`/admin/roles/${editing.id}`, values);
        message.success('Role updated');
      } else {
        await api.post('/admin/roles', values);
        message.success('Role created');
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
    {
      title: 'Permissions',
      dataIndex: 'permissions',
      key: 'permissions',
      render: (perms) => (
        <Space wrap size={[4, 4]}>
          {(perms || []).map((p) => (
            <Tag key={p} color={PERMISSION_COLORS[p] || 'default'}>{p}</Tag>
          ))}
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
          <h1 className="admin-page-header__title">User Roles</h1>
          <p className="admin-page-header__subtitle">
            Create roles and assign granular read/write permissions across the platform.
          </p>
        </div>
        {canWrite && (
          <Button type="primary" icon={<FaPlus />} onClick={openCreate} size="large">
            Add role
          </Button>
        )}
      </div>

      <Card className="admin-data-card" bordered={false}>
        <Table
          rowKey="id"
          loading={loading}
          dataSource={roles}
          columns={columns}
          pagination={{ pageSize: 10, showSizeChanger: true, showTotal: (t) => `${t} roles` }}
        />
      </Card>

      <Modal
        title={editing ? 'Edit role' : 'Create role'}
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
          <Form.Item name="permissions" label="Permissions">
            <Select mode="tags" placeholder="Select or type permissions">
              {PERMISSION_OPTIONS.map((p) => (
                <Select.Option key={p} value={p}>{p}</Select.Option>
              ))}
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
