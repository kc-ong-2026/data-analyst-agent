import '@testing-library/jest-dom';
import { TextEncoder, TextDecoder } from 'util';

// Mock environment variables
process.env.VITE_API_URL = 'http://localhost:8000/api';

// Polyfill TextEncoder/TextDecoder for Node.js environment
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder as any;

// Polyfill ReadableStream for Node.js environment (Node 18+ has it built-in)
if (typeof ReadableStream === 'undefined') {
  // @ts-ignore
  global.ReadableStream = class ReadableStream {
    private reader: any;

    constructor(underlyingSource?: any) {
      const chunks: any[] = [];
      if (underlyingSource && underlyingSource.start) {
        const controller = {
          enqueue: (chunk: any) => chunks.push(chunk),
          close: () => {},
        };
        underlyingSource.start(controller);
      }

      this.reader = {
        read: async () => {
          if (chunks.length === 0) {
            return { done: true, value: undefined };
          }
          return { done: false, value: chunks.shift() };
        },
      };
    }

    getReader() {
      return this.reader;
    }
  };
}

// Mock import.meta.env for Vite
Object.defineProperty(globalThis, 'import', {
  value: {
    meta: {
      env: {
        VITE_API_URL: 'http://localhost:8000/api',
      },
    },
  },
  writable: true,
});

// Mock fetch for SSE testing
global.fetch = jest.fn();
