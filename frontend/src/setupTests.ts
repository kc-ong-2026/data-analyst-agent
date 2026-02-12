import '@testing-library/jest-dom';

// Mock env module to handle import.meta
jest.mock('./config/env', () => ({
  getApiUrl: () => 'http://localhost:8000/api',
}));
