import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { jwtDecode } from 'jwt-decode';
import { getMe } from '../api/auth';

const TOKEN_KEY = 'retentrix_token';
const USER_KEY  = 'retentrix_user';
const ACTIVE_KEY = 'retentrix_last_active';

// Session expires after 7 days of inactivity
const SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1000;

const AuthContext = createContext(null);

function isTokenExpired(token) {
  try {
    const { exp } = jwtDecode(token);
    return Date.now() >= exp * 1000;
  } catch {
    return true;
  }
}

function isSessionExpired() {
  const lastActive = localStorage.getItem(ACTIVE_KEY);
  if (!lastActive) return false;
  return Date.now() - parseInt(lastActive, 10) > SESSION_TTL_MS;
}

export function AuthProvider({ children }) {
  const [user, setUser]       = useState(null);
  const [token, setToken]     = useState(null);
  const [loading, setLoading] = useState(true);

  // ── Bootstrap: restore session on mount ──
  useEffect(() => {
    const storedToken = localStorage.getItem(TOKEN_KEY);
    if (!storedToken || isTokenExpired(storedToken) || isSessionExpired()) {
      clearSession();
      setLoading(false);
      return;
    }
    const storedUser = localStorage.getItem(USER_KEY);
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser));
        setToken(storedToken);
      } catch {
        clearSession();
      }
    }
    // Refresh user from backend in background
    getMe()
      .then((fresh) => {
        setUser(fresh);
        localStorage.setItem(USER_KEY, JSON.stringify(fresh));
      })
      .catch(() => clearSession())
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function saveSession(accessToken, userObj) {
    localStorage.setItem(TOKEN_KEY, accessToken);
    localStorage.setItem(USER_KEY, JSON.stringify(userObj));
    localStorage.setItem(ACTIVE_KEY, Date.now().toString());
    setToken(accessToken);
    setUser(userObj);
  }

  function clearSession() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    localStorage.removeItem(ACTIVE_KEY);
    setToken(null);
    setUser(null);
  }

  const logout = useCallback(() => {
    clearSession();
  }, []);

  const isAuthenticated = !!token && !!user;
  const role = user?.role ?? null;
  const isAdmin = role === 'admin';
  const isCSM   = role === 'csm';
  const canWrite = isAdmin || isCSM;

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        loading,
        isAuthenticated,
        role,
        isAdmin,
        isCSM,
        canWrite,
        saveSession,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
