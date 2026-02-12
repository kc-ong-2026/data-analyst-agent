import '@testing-library/jest-dom';
import { TextEncoder, TextDecoder } from 'util';

// Suppress console logs in tests (except for actual errors)
const originalError = console.error;
const originalWarn = console.warn;
const originalLog = console.log;

beforeAll(() => {
  console.error = (...args: unknown[]) => {
    // Suppress expected error messages in tests
    const message = args[0]?.toString() || '';
    if (
      message.includes('Failed to parse SSE event') ||
      message.includes('[Stream] Error:')
    ) {
      return;
    }
    originalError(...args);
  };

  console.warn = jest.fn();
  console.log = jest.fn();
});

afterAll(() => {
  console.error = originalError;
  console.warn = originalWarn;
  console.log = originalLog;
});

// Mock environment variables
process.env.VITE_API_URL = 'http://localhost:8000/api';

// Polyfill TextEncoder/TextDecoder for Node.js environment
global.TextEncoder = TextEncoder;
// @ts-expect-error - TextDecoder type mismatch between Node and DOM
global.TextDecoder = TextDecoder;

// Polyfill ReadableStream for Node.js environment (Node 18+ has it built-in)
if (typeof ReadableStream === 'undefined') {
  interface MockUnderlyingSource {
    start?: (controller: MockController) => void;
  }

  interface MockController {
    enqueue: (chunk: unknown) => void;
    close: () => void;
  }

  interface MockReader {
    read: () => Promise<{ done: boolean; value: unknown }>;
  }

  // @ts-expect-error - Polyfill for testing environment
  global.ReadableStream = class ReadableStream {
    private reader: MockReader;

    constructor(underlyingSource?: MockUnderlyingSource) {
      const chunks: unknown[] = [];
      if (underlyingSource?.start) {
        const controller: MockController = {
          enqueue: (chunk: unknown) => chunks.push(chunk),
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
