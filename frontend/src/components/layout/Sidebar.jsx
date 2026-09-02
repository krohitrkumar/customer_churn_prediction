import { NavLink, useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import Logo from '../ui/Logo';
import './Sidebar.css';

const NAV_ITEMS = [
  {
    path: '/dashboard',
    label: 'Dashboard',
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
        <rect x="2" y="2" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth="1.6"/>
        <rect x="10" y="2" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth="1.6"/>
        <rect x="2" y="10" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth="1.6"/>
        <rect x="10" y="10" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth="1.6"/>
      </svg>
    ),
  },
  {
    path: '/customers',
    label: 'Customers',
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
        <circle cx="9" cy="6" r="3" stroke="currentColor" strokeWidth="1.6"/>
        <path d="M3 16c0-3.314 2.686-6 6-6s6 2.686 6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
      </svg>
    ),
  },
  {
    path: '/predict',
    label: 'AI Predict',
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
        <path d="M2 14 L6 8 L10 11 L13 5 L16 7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
        <circle cx="16" cy="7" r="2" fill="currentColor" opacity="0.8"/>
      </svg>
    ),
    highlight: true,
  },
];

const BOTTOM_ITEMS = [
  {
    path: '/settings',
    label: 'Settings',
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
        <circle cx="9" cy="9" r="2.5" stroke="currentColor" strokeWidth="1.6"/>
        <path d="M9 1.5v2M9 14.5v2M1.5 9h2M14.5 9h2M3.58 3.58l1.41 1.41M13.01 13.01l1.41 1.41M3.58 14.42l1.41-1.41M13.01 4.99l1.41-1.41" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
      </svg>
    ),
  },
];

export default function Sidebar({ collapsed, onToggle, mobileOpen = false }) {
  const { user, logout, isAdmin, isCSM } = useAuth();
  const navigate = useNavigate();

  function handleLogout() {
    logout();
    navigate('/login');
  }

  const roleBadge = isAdmin ? 'Admin' : isCSM ? 'CSM' : 'Analyst';
  const roleClass = isAdmin ? 'role--admin' : isCSM ? 'role--csm' : 'role--user';

  return (
    <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''} ${mobileOpen ? 'mobile-open' : ''}`} aria-label="Main navigation">
      {/* Header */}
      <div className="sidebar-header">
        {!collapsed && (
          <NavLink to="/dashboard" style={{ pointerEvents: 'none', textDecoration: 'none' }}>
            <Logo size="sm" showText={true} />
          </NavLink>
        )}
        <button
          className="sidebar-toggle"
          onClick={onToggle}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            {collapsed
              ? <path d="M5 3l6 5-6 5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
              : <path d="M11 3L5 8l6 5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
            }
          </svg>
        </button>
      </div>

      {/* Divider */}
      <div className="sidebar-divider" />

      {/* Nav items */}
      <nav className="sidebar-nav" aria-label="Primary navigation">
        <ul>
          {NAV_ITEMS.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) =>
                  `sidebar-item ${isActive ? 'sidebar-item--active' : ''} ${item.highlight ? 'sidebar-item--highlight' : ''}`
                }
              >
                <span className="sidebar-item-icon">{item.icon}</span>
                {!collapsed && <span className="sidebar-item-label">{item.label}</span>}
                {!collapsed && item.highlight && (
                  <span className="sidebar-item-badge">AI</span>
                )}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Spacer */}
      <div className="sidebar-spacer" />

      {/* Bottom: settings + user */}
      <div className="sidebar-bottom">
        <div className="sidebar-divider" />
        {BOTTOM_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `sidebar-item ${isActive ? 'sidebar-item--active' : ''}`
            }
          >
            <span className="sidebar-item-icon">{item.icon}</span>
            {!collapsed && <span className="sidebar-item-label">{item.label}</span>}
          </NavLink>
        ))}

        {/* User card */}
        <div className={`sidebar-user ${collapsed ? 'sidebar-user--collapsed' : ''}`}>
          <div className="sidebar-avatar" aria-hidden="true">
            {user?.first_name?.[0]?.toUpperCase() ?? 'U'}
          </div>
          {!collapsed && (
            <div className="sidebar-user-info">
              <span className="sidebar-user-name">
                {user?.first_name} {user?.last_name}
              </span>
              <span className={`sidebar-user-role ${roleClass}`}>{roleBadge}</span>
            </div>
          )}
          {!collapsed && (
            <button
              className="sidebar-logout"
              onClick={handleLogout}
              aria-label="Logout"
              title="Logout"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M6 14H3a1 1 0 01-1-1V3a1 1 0 011-1h3M10 11l3-3-3-3M13 8H6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          )}
        </div>
      </div>
    </aside>
  );
}
