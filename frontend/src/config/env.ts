/**
 * Environment configuration helper
 * Centralizes access to import.meta.env for easier testing
 */
export const getApiUrl = (): string => {
  // In test environment (Node.js), use process.env
  if (typeof process !== 'undefined' && process.env.NODE_ENV === 'test') {
    return process.env.VITE_API_URL || '/api';
  }
  // In browser/Vite environment, check if import.meta is available
  if (typeof window !== 'undefined') {
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore - import.meta is available in Vite but not in Jest
    return import.meta?.env?.VITE_API_URL || '/api';
  }
  // Fallback
  return '/api';
};
