/* eslint-disable react-hooks/exhaustive-deps */
import React, { useEffect, useMemo, useState } from "react";
import "./akkioSidebar.scss";
import "./akkioSidebarLayout.scss";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { IoHome, IoChevronForward } from "react-icons/io5";
import { FaPuzzlePiece, FaLayerGroup, FaUserGear } from "react-icons/fa6";
import WorkspaceUsageCard from "./WorkspaceUsageCard";
import { hasPermission } from "../../utils/auth";

const MENU_ITEMS = [
  { name: 'Home', icon: IoHome, path: '/welcome', id: 1, permission: 'home' },
  // { name: 'Multi Agent', icon: FaBrain, path: '/multi-agent', permission: 'reports' },
  { name: 'App Builder', icon: FaPuzzlePiece, path: '/app-builder', permission: 'reports' },
  { name: 'Agentic Builder', icon: FaLayerGroup, path: '/agentic-builder', permission: 'reports' },
];

const ADMIN_CHILDREN = [
  { name: 'Users', path: '/administration/users' },
  { name: 'Organizations', path: '/administration/organizations' },
  { name: 'User Roles', path: '/administration/roles' },
];

export default function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();
  const isAdminRoute = location.pathname.startsWith('/administration');
  const [adminExpanded, setAdminExpanded] = useState(isAdminRoute);
  const showAdmin = hasPermission('admin');

  useEffect(() => {
    if (isAdminRoute) {
      setAdminExpanded(true);
    }
  }, [isAdminRoute]);

  const menuItems = useMemo(() => {
    return MENU_ITEMS.filter((item) => hasPermission(item.permission));
  }, [location.pathname]);

  const isActive = (item) => {
    if (item.name === "App Builder" && (location.pathname === "/app-builder" || location.pathname.startsWith("/app-builder/"))) {
      return true;
    }
    if (
      item.name === "Agentic Builder" &&
      (location.pathname.startsWith("/agentic-builder") || location.pathname.startsWith("/fullstack-builder"))
    ) {
      return true;
    }
    return location.pathname === item.path;
  };

  const toggleAdmin = () => {
    const next = !adminExpanded;
    setAdminExpanded(next);
    if (next && !isAdminRoute) {
      navigate('/administration/users');
    }
  };

  return (
    <div className="akkioSidebar akkioSidebarLayout akkioSidebarLayout--main">
      <div className="akkioSidebar__header">
      </div>
      <hr className="akkioSidebar__divider" />

      <ul className="akkioSidebar__list" id="menu">
        {menuItems.map((item) => (
          <li
            key={item.name}
            className={`akkioSidebar__item ${isActive(item) ? "akkioSidebar__item--active" : ""}`}
          >
            <Link to={item.path} className="akkioSidebar__link">
              <item.icon size={20} className="akkioSidebar__icon" />
              <span className="akkioSidebar__text">{item.name}</span>
            </Link>
          </li>
        ))}

        {showAdmin && (
          <li
            className={`akkioSidebar__item akkioSidebar__item--group ${
              isAdminRoute ? "akkioSidebar__item--active" : ""
            }`}
          >
            <button
              type="button"
              className="akkioSidebar__link akkioSidebar__link--toggle"
              onClick={toggleAdmin}
              aria-expanded={adminExpanded}
            >
              <FaUserGear size={20} className="akkioSidebar__icon" />
              <span className="akkioSidebar__text">Administration</span>
              <IoChevronForward
                size={16}
                className={`akkioSidebar__chev ${adminExpanded ? "akkioSidebar__chev--open" : ""}`}
              />
            </button>

            {adminExpanded && (
              <ul className="akkioSidebar__sublist">
                {ADMIN_CHILDREN.map((child) => (
                  <li
                    key={child.path}
                    className={`akkioSidebar__item akkioSidebar__item--sub ${
                      location.pathname === child.path ? "akkioSidebar__item--active" : ""
                    }`}
                  >
                    <Link to={child.path} className="akkioSidebar__link">
                      <span className="akkioSidebar__text">{child.name}</span>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </li>
        )}
      </ul>

      <div className="akkioSidebar__footer">
        <WorkspaceUsageCard />
      </div>
    </div>
  );
}
