/**
 * Mock implementation for testing
 */
export const getApiUrl = (): string => {
  return process.env.VITE_API_URL || 'http://localhost:8000/api';
};
