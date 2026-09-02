import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import Logo from '../ui/Logo';
import './TopBar.css';

export default function TopBar({ onMenuToggle, sidebarCollapsed = false }) {
  const { user } = useAuth();
  const navigate = useNavigate();

  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('retentrix_theme') || 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('retentrix_theme', theme);
  }, [theme]);

  function toggleTheme() {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }

  return (
    <header className={`topbar ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`} role="banner">
      {/* Left: hamburger (mobile) */}
      <button
        className="topbar-menu-btn"
        onClick={onMenuToggle}
        aria-label="Toggle navigation menu"
      >
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <path d="M3 5h14M3 10h14M3 15h14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
      </button>

      {/* Mobile logo */}
      <div className="topbar-mobile-logo">
        <Logo size="xs" showText />
      </div>

      {/* Right actions */}
      <div className="topbar-right">
        {/* Light/Dark Theme Switcher */}
        <button
          className="topbar-theme-btn"
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          {theme === 'dark' ? (
            /* Sun Icon */
            <svg width="17" height="17" viewBox="0 0 20 20" fill="none">
              <circle cx="10" cy="10" r="4" stroke="currentColor" strokeWidth="1.8" />
              <path d="M10 2v2M10 16v2M2 10h2M16 10h2M4.34 4.34l1.42 1.42M14.24 14.24l1.42 1.42M4.34 15.66l1.42-1.42M14.24 5.76l1.42-1.42" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          ) : (
            /* Moon Icon */
            <svg width="17" height="17" viewBox="0 0 20 20" fill="none">
              <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </button>

        {/* Predict CTA */}
        <button
          className="topbar-predict-btn"
          onClick={() => navigate('/predict')}
          aria-label="Run AI prediction"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.5" />
            <path d="M5 3.5l4 3.5-4 3.5" fill="currentColor" />
          </svg>
          <span>Run Predict</span>
        </button>

        {/* Avatar */}
        <button
          className="topbar-avatar"
          onClick={() => navigate('/settings')}
          aria-label={`Profile — ${user?.first_name}`}
          title={user?.email}
        >
          {user?.first_name?.[0]?.toUpperCase() ?? 'U'}
        </button>
      </div>
    </header>
  );
}
