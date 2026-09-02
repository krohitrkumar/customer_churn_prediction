import { useState, useCallback, useRef } from 'react';

let toastIdCounter = 0;

export function useToastState() {
  const [toasts, setToasts] = useState([]);
  const timers = useRef({});

  const dismiss = useCallback((id) => {
    clearTimeout(timers.current[id]);
    setToasts((prev) => prev.map((t) => (t.id === id ? { ...t, leaving: true } : t)));
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 300);
  }, []);

  const toast = useCallback(
    ({ type = 'info', title, message, duration = 4000 }) => {
      const id = ++toastIdCounter;
      setToasts((prev) => [...prev, { id, type, title, message, leaving: false }]);
      if (duration > 0) {
        timers.current[id] = setTimeout(() => dismiss(id), duration);
      }
      return id;
    },
    [dismiss]
  );

  const success = useCallback((title, message) => toast({ type: 'success', title, message }), [toast]);
  const error   = useCallback((title, message) => toast({ type: 'error',   title, message, duration: 6000 }), [toast]);
  const warning = useCallback((title, message) => toast({ type: 'warning', title, message }), [toast]);
  const info    = useCallback((title, message) => toast({ type: 'info',    title, message }), [toast]);

  return { toasts, toast, success, error, warning, info, dismiss };
}
