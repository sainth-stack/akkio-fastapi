import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, Input, message as antdMessage } from 'antd';
import { FaEdit, FaTrashAlt, FaSearch } from 'react-icons/fa';
import { FaPlus } from 'react-icons/fa6';
import Spinner from 'react-bootstrap/Spinner';

import api from '../utils/api';
import { IconBadge, IoAppsOutline } from './AppBuilderIcons';
import { BUILDER_KIND_APP, getBuilderConfig } from './builderConfig';
import './AppBuilderList.css';

const AppBuilderList = ({ builderKind = BUILDER_KIND_APP }) => {
  const config = getBuilderConfig(builderKind);
  const navigate = useNavigate();
  const [apps, setApps] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    const fetchApps = async () => {
      setLoading(true);
      try {
        const res = await api.get('/app-builder/apps', { params: { kind: config.kind } });
        const data = res.data;
        if (data.status === 'success' && Array.isArray(data.apps)) {
          setApps(data.apps);
        }
      } catch (err) {
        console.error('Error fetching apps:', err);
        antdMessage.error('Failed to load apps');
      } finally {
        setLoading(false);
      }
    };
    fetchApps();
  }, [config.kind]);

  const filteredApps = useMemo(() => {
    if (!searchQuery.trim()) return apps;
    const q = searchQuery.toLowerCase();
    return apps.filter(
      (app) =>
        (app.app_name && app.app_name.toLowerCase().includes(q)) ||
        (app.prompt && app.prompt.toLowerCase().includes(q)) ||
        (app.project_name && app.project_name.toLowerCase().includes(q))
    );
  }, [apps, searchQuery]);

  const handleDelete = async (app, e) => {
    e.stopPropagation();
    if (!window.confirm(`Delete "${app.app_name}"? This cannot be undone.`)) return;
    try {
      const res = await api.delete(`/app-builder/apps/${app.id}`);
      const data = res.data;
      if (data.status === 'success') {
        antdMessage.success('App deleted');
        setApps((prev) => prev.filter((a) => a.id !== app.id));
      } else {
        antdMessage.error(data.detail || 'Failed to delete');
      }
    } catch (err) {
      console.error(err);
      antdMessage.error('Failed to delete app');
    }
  };

  const handleEdit = (app) => {
    navigate(config.editPath(app.id));
  };

  return (
    <div className="app-builder-list">
      <div className="app-builder-list__header">
        <h1 className="app-builder-list__title">All</h1>
      </div>

      <div className="app-builder-list__search">
        <Input
          prefix={<FaSearch style={{ color: '#94a3b8' }} />}
          placeholder={config.searchPlaceholder}
          size="large"
          allowClear
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="app-builder-list__search-input"
        />
      </div>

      <div>
        <h3 className="app-builder-list__section-title">{config.listTitle}</h3>

        {loading ? (
          <div className="app-builder-list__loading">
            <Spinner animation="border" />
          </div>
        ) : (
          <div className="app-builder-list__grid">
            <div
              className="app-builder-list__new-card"
              onClick={() => navigate(config.newPath)}
            >
              <div className="app-builder-list__new-icon">
                <FaPlus size={22} />
              </div>
              <div className="app-builder-list__new-label">{config.newCardLabel}</div>
            </div>

            {filteredApps.map((app) => (
              <div
                key={app.id}
                className="app-builder-list__card"
                onClick={() => handleEdit(app)}
              >
                <div className="app-builder-list__card-header">
                  <div className="app-builder-list__card-icon">
                    <IconBadge icon={IoAppsOutline} variant="slate" size={20} className="app-builder-list__card-icon-badge" />
                  </div>
                  <div className="app-builder-list__card-actions">
                    <Button
                      type="text"
                      size="small"
                      icon={<FaEdit style={{ color: '#64748b' }} />}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleEdit(app);
                      }}
                      title="Edit"
                    />
                    <Button
                      type="text"
                      size="small"
                      danger
                      icon={<FaTrashAlt />}
                      onClick={(e) => handleDelete(app, e)}
                      title="Delete"
                    />
                  </div>
                </div>

                <h3 className="app-builder-list__card-title">
                  {app.app_name || app.project_name || 'Untitled app'}
                </h3>

                <div className="app-builder-list__card-desc">
                  {app.prompt || 'No description.'}
                </div>

                <div className="app-builder-list__card-cta">
                  <Button
                    type="primary"
                    block
                    style={{ borderRadius: '8px', background: '#eef2ff', color: '#4f46e5', border: 'none', fontWeight: '600' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleEdit(app);
                    }}
                  >
                    Open / Edit
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default AppBuilderList;
