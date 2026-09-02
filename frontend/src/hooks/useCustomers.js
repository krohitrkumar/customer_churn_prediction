import { useState, useEffect, useCallback, useRef } from 'react';
import { getCustomers } from '../api/customers';

// Module-level cache so all components share the same data
// and we don't re-fetch on every mount
let _cache = null;
let _cacheTs = 0;
const CACHE_TTL_MS = 30_000; // 30 seconds stale-while-revalidate

const listeners = new Set();

function notifyListeners(data) {
  listeners.forEach((fn) => fn(data));
}

export function useCustomers() {
  const [customers, setCustomers] = useState(_cache ?? []);
  const [loading, setLoading] = useState(_cache === null);
  const [error, setError] = useState(null);

  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    // Subscribe to updates from other hook instances
    listeners.add(setCustomers);
    return () => {
      mounted.current = false;
      listeners.delete(setCustomers);
    };
  }, []);

  const fetch = useCallback(async (force = false) => {
    const now = Date.now();
    // Return cache if fresh and not forced
    if (!force && _cache !== null && now - _cacheTs < CACHE_TTL_MS) {
      if (mounted.current) setCustomers(_cache);
      return;
    }

    if (mounted.current) {
      setLoading(true);
      setError(null);
    }
    try {
      const data = await getCustomers({ limit: 500 });
      _cache = data;
      _cacheTs = Date.now();
      notifyListeners(data); // Update all mounted instances
      if (mounted.current) setCustomers(data);
    } catch (err) {
      if (mounted.current) setError(err);
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, []);

  // Only fetch on mount if cache is stale or empty
  useEffect(() => {
    fetch();
  }, [fetch]);

  const refetch = useCallback(() => fetch(true), [fetch]);

  return { customers, loading, error, refetch };
}

// Helper to invalidate cache from outside (e.g. after create/delete)
export function invalidateCustomerCache() {
  _cache = null;
  _cacheTs = 0;
}
