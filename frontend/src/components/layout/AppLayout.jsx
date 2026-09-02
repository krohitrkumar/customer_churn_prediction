import { useState } from 'react';
import Sidebar from './Sidebar';
import TopBar from './TopBar';
import './AppLayout.css';

export default function AppLayout({ children }) {
  const [collapsed, setCollapsed]   = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="app-layout">
      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <Sidebar
        collapsed={collapsed}
        onToggle={() => setCollapsed((c) => !c)}
        mobileOpen={mobileOpen}
      />

      {/* Main */}
      <div className={`app-main ${collapsed ? 'sidebar-collapsed' : ''}`}>
        <TopBar
          onMenuToggle={() => setMobileOpen((o) => !o)}
          sidebarCollapsed={collapsed}
        />
        <main className="page-content" id="main-content">
          {children}
        </main>
      </div>
    </div>
  );
}
