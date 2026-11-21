import React, { useCallback, useEffect, useState } from 'react';
import { Button, Card, Form, Input, Modal, Space, Table, message } from 'antd';
import { FaPlus, FaEdit, FaTrashAlt } from 'react-icons/fa';
import api from '../../utils/api';
import { hasWritePermission } from '../../utils/auth';

export default function OrganizationsTab() {
  const canWrite = hasWritePermission('admin');
  const [organizations, setOrganizations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState(null);
  const [form] = Form.useForm();

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get('/admin/organizations');
      setOrganizations(res.data || []);
    } catch (e) {
      message.error('Failed to load organizations');
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
    setModalOpen(true);
  };

  const openEdit = (record) => {
    setEditing(record);
    form.setFieldsValue({ name: record.name, description: record.description });
    setModalOpen(true);
  };

  const handleDelete = async (record) => {
    if (!window.confirm(`Delete organization "${record.name}"?`)) return;
    try {
      await api.delete(`/admin/organizations/${record.id}`);
      message.success('Organization deleted');
      loadData();
    } catch (e) {
      message.error(e?.response?.data?.detail || 'Delete failed');
    }
  };

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      if (editing) {
        await api.put(`/admin/organizations/${editing.id}`, values);
        message.success('Organization updated');
      } else {
        await api.post('/admin/organizations', values);
        message.success('Organization created');
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
    { title: 'Description', dataIndex: 'description', key: 'description', render: (v) => v || '—' },
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
          <h1 className="admin-page-header__title">Organizations</h1>
          <p className="admin-page-header__subtitle">
            Define tenant boundaries and group users by team or customer account.
          </p>
        </div>
        {canWrite && (
          <Button type="primary" icon={<FaPlus />} onClick={openCreate} size="large">
            Add organization
          </Button>
        )}
      </div>

      <Card className="admin-data-card" bordered={false}>
        <Table
          rowKey="id"
          loading={loading}
          dataSource={organizations}
          columns={columns}
          pagination={{ pageSize: 10, showSizeChanger: true, showTotal: (t) => `${t} organizations` }}
        />
      </Card>

      <Modal
        title={editing ? 'Edit organization' : 'Create organization'}
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
          <Form.Item name="description" label="Description">
            <Input.TextArea rows={3} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
