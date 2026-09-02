import { useState } from 'react';
import { predictSingle, getPredictionHistory } from '../api/predictions';

export function usePrediction() {
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  async function predict(data) {
    setLoading(true);
    setError(null);
    try {
      const res = await predictSingle(data);
      setResult(res);
      return res;
    } catch (err) {
      setError(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }

  function reset() { setResult(null); setError(null); }

  return { result, loading, error, predict, reset };
}

export function useHistory(customerId) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  async function load() {
    if (!customerId) return;
    setLoading(true);
    try {
      const data = await getPredictionHistory(customerId);
      setHistory(data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }

  return { history, loading, error, load };
}
