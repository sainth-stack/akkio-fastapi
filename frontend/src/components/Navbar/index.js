import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Logo from '../../assets/images/OtamatLogo.svg'
import { useLocation } from "react-router-dom";
import api from '../../utils/api';
import { clearAuthStorage, getStoredUser } from '../../utils/auth';

const TITLE_BY_PATH = {
  '/welcome': 'Home',
  '/multi-agent': 'Multi Agent',
  '/app-builder': 'App Builder',
  '/agentic-builder': 'Agentic Builder',
  '/fullstack-builder': 'Agentic Builder',
  '/administration': 'Administration',
};

function Navbar() {
  const navigate = useNavigate()
  const [name, setName] = useState("Home")
  const handleLogout = async () => {
    try {
      await api.post('/auth/logout');
    } catch (e) {
      // ignore — still clear local session
    }
    clearAuthStorage();
    navigate('/login')
  }
  let location = useLocation();
  useEffect(() => {
    if (location.pathname.startsWith('/app-builder')) {
      setName('App Builder')
      return
    }
    if (location.pathname.startsWith('/agentic-builder') || location.pathname.startsWith('/fullstack-builder')) {
      setName('Agentic Builder')
      return
    }
    if (location.pathname.startsWith('/administration')) {
      const sub = location.pathname.replace(/^\/administration\/?/, '');
      const subTitles = {
        users: 'Users',
        organizations: 'Organizations',
        roles: 'User Roles',
      };
      setName(sub && subTitles[sub] ? `Administration · ${subTitles[sub]}` : 'Administration');
      return;
    }
    setName(TITLE_BY_PATH[location.pathname] || 'Home')
  }, [location.pathname])

  const user = getStoredUser();
  const fileName = localStorage.getItem("filename") || ""
  return (
    <>
      <nav className="navbar navbar-expand-lg navbar-light bg-white shadow-sm sticky-top bg-white-fixed" style={{ zIndex: 10, marginLeft: '0px' }}>
        <div className="collapse navbar-collapse" style={{ marginLeft: '0px' }} id="navbarNav">
          <img
            src={Logo}
            style={{ width: '100px',height:'60px',marginLeft:'50px' }}
            id="logo_RL"
            alt="Otamat"
          />
          <div style={{
            marginLeft: '80px',
            marginTop: '10px',
            fontWeight: 700,
            fontSize: '23px'
          }}>
            {name}
          </div>

        </div>
        {!!fileName && (
          <div className="me-2 d-flex align-items-center">
            <span
              className="d-inline-flex align-items-center"
              title={fileName}
              style={{
                fontFamily: "poppins",
                fontSize: "12px",
                fontWeight: 500,
                color: "#212529",
                background: "#f8f9fa",
                border: "1px solid rgba(0,0,0,0.08)",
                borderRadius: "999px",
                padding: "6px 10px",
                maxWidth: "340px",
                boxShadow: "0 1px 1px rgba(0,0,0,0.03)",
              }}
            >
              <i className="bi bi-file-earmark-text me-2" style={{ fontSize: "14px", color: "#6c757d" }} />
              <span
                style={{
                  maxWidth: "290px",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {fileName}
              </span>
            </span>
          </div>
        )}
        <div className="nav-item ms-1 dropdown d-flex align-items-center mr-0 pr-0" style={{ color: 'black' }}>
          <a
            className="nav-link dropdown-toggle p-0 m-0 pe-5"
            href="/#"
            id="navbarDropdown"
            role="button"
            data-bs-toggle="dropdown"
            aria-haspopup="true"
            aria-expanded="false"
            style={{ textDecoration: 'none', color: 'black' }}
          >
            <span className="ml-2 fs14 text-dark" title={"Admin"}>
              {user?.name || user?.email || "User"}
            </span>
            <i className="bi bi-caret-down-fill"></i>
          </a>
          <div className="dropdown-menu" aria-labelledby="navbarDropdown" style={{ position: "absolute", left: "-60px", top: "30px" }}>
            <div className="dropdown-item user-info" style={{
              padding: "10px 15px",
              borderBottom: "1px solid #eee"
            }}>
              <div style={{ fontWeight: "500" }}>
                {user?.name || user?.email || "User"}
              </div>
            </div>
            <span className="dropdown-item" style={{ cursor: 'pointer' }} onClick={() => handleLogout()}>Logout</span>
          </div>
        </div>
      </nav>
    </>

  );

}
export default Navbar;
