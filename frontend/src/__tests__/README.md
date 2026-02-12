# Frontend Testing Guide

This directory contains all frontend tests using **Jest** and **React Testing Library**.

## Test Structure

```
src/
├── api/
│   ├── __tests__/
│   │   └── client.test.ts          # API client tests
│   └── client.ts
├── store/
│   ├── __tests__/
│   │   └── chatStore.test.ts       # Zustand store tests
│   └── chatStore.ts
├── types/
│   ├── __tests__/
│   │   └── index.test.ts           # Type definition tests
│   └── index.ts
└── components/
    ├── __tests__/
    │   ├── ChatMessage.test.tsx     # Component tests (future)
    │   ├── ChatInput.test.tsx
    │   └── VisualizationPanel.test.tsx
    └── ...
```

## Running Tests

```bash
# Run all tests
npm test

# Run in watch mode (recommended for development)
npm run test:watch

# Run with coverage
npm run test:coverage

# Run in CI mode
npm run test:ci

# Run specific test file
npm test -- api/__tests__/client.test.ts

# Run tests matching a pattern
npm test -- --testNamePattern="streaming"
```

## Writing Tests

### 1. Testing Zustand Stores

```typescript
import { renderHook, act } from '@testing-library/react';
import { useChatStore } from '../chatStore';

describe('chatStore', () => {
  it('should update state correctly', () => {
    const { result } = renderHook(() => useChatStore());

    expect(result.current.messages).toEqual([]);

    act(() => {
      result.current.sendMessage('Hello');
    });

    expect(result.current.messages).toHaveLength(1);
  });
});
```

### 2. Testing API Calls

```typescript
import { chatApi } from '../client';

jest.mock('axios');

describe('chatApi', () => {
  it('should send message', async () => {
    const mockResponse = { message: 'Hello', conversation_id: 'conv-123' };
    (chatApi.sendMessage as jest.Mock).mockResolvedValue(mockResponse);

    const result = await chatApi.sendMessage({ message: 'Test' });

    expect(result.conversation_id).toBe('conv-123');
  });
});
```

### 3. Testing SSE Streaming

```typescript
it('should handle SSE events', async () => {
  const mockEvents = [
    'data: {"type":"start","conversation_id":"conv-123"}\n\n',
    'data: {"type":"token","content":"Hello"}\n\n',
    'data: {"type":"done"}\n\n',
  ];

  const mockReadableStream = new ReadableStream({
    start(controller) {
      mockEvents.forEach(event => {
        controller.enqueue(new TextEncoder().encode(event));
      });
      controller.close();
    },
  });

  (global.fetch as jest.Mock).mockResolvedValue({
    ok: true,
    body: mockReadableStream,
  });

  const callbacks = {
    onStart: jest.fn(),
    onToken: jest.fn(),
    onComplete: jest.fn(),
  };

  await chatApi.streamMessage({ message: 'Test' }, callbacks);

  expect(callbacks.onStart).toHaveBeenCalledWith('conv-123');
  expect(callbacks.onToken).toHaveBeenCalledWith('Hello');
});
```

### 4. Testing React Components (Example)

```typescript
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ChatInput from '../ChatInput';

describe('ChatInput', () => {
  it('should render input field', () => {
    render(<ChatInput onSend={jest.fn()} />);

    expect(screen.getByPlaceholderText(/type a message/i)).toBeInTheDocument();
  });

  it('should call onSend when submitted', async () => {
    const onSend = jest.fn();
    render(<ChatInput onSend={onSend} />);

    const input = screen.getByPlaceholderText(/type a message/i);
    await userEvent.type(input, 'Hello{enter}');

    expect(onSend).toHaveBeenCalledWith('Hello');
  });
});
```

## Best Practices

### 1. Always Clean Up Mocks

```typescript
beforeEach(() => {
  jest.clearAllMocks();
});
```

### 2. Test Both Success and Error Cases

```typescript
it('should handle success', async () => {
  // Test happy path
});

it('should handle errors', async () => {
  // Test error path
});
```

### 3. Use `waitFor` for Async Operations

```typescript
import { waitFor } from '@testing-library/react';

await waitFor(() => {
  expect(result.current.isLoading).toBe(false);
});
```

### 4. Mock External Dependencies

```typescript
// Mock axios
jest.mock('axios');

// Mock fetch
global.fetch = jest.fn();

// Mock environment variables
process.env.VITE_API_URL = 'http://localhost:8000/api';
```

### 5. Use Descriptive Test Names

```typescript
// ✅ Good
it('should display error message when API call fails', () => {});

// ❌ Bad
it('should work', () => {});
```

### 6. Test User Interactions

```typescript
import userEvent from '@testing-library/user-event';

const user = userEvent.setup();
await user.click(button);
await user.type(input, 'Hello');
```

## Coverage Thresholds

The project maintains the following coverage thresholds:

- **Branches**: 70%
- **Functions**: 70%
- **Lines**: 70%
- **Statements**: 70%

View coverage report:
```bash
npm run test:coverage
open coverage/lcov-report/index.html
```

## Common Issues

### Issue 1: "Cannot find module"
**Solution**: Make sure all dependencies are installed:
```bash
npm install
```

### Issue 2: "ReferenceError: TextEncoder is not defined"
**Solution**: This is already handled in `jest.setup.ts`. Make sure the setup file is configured in `jest.config.cjs`.

### Issue 3: Tests timeout
**Solution**: Increase Jest timeout:
```typescript
jest.setTimeout(10000); // 10 seconds
```

### Issue 4: "Cannot read property of undefined"
**Solution**: Check that mocks are properly set up in `beforeEach`:
```typescript
beforeEach(() => {
  (global.fetch as jest.Mock).mockClear();
});
```

## Debugging Tests

### Run a single test
```bash
npm test -- --testNamePattern="should send message"
```

### Run with verbose output
```bash
npm test -- --verbose
```

### Run without coverage (faster)
```bash
npm test -- --no-coverage
```

### Debug in VS Code
Add this to `.vscode/launch.json`:
```json
{
  "type": "node",
  "request": "launch",
  "name": "Jest Debug",
  "program": "${workspaceFolder}/frontend/node_modules/.bin/jest",
  "args": ["--runInBand", "--no-coverage"],
  "console": "integratedTerminal",
  "internalConsoleOptions": "neverOpen"
}
```

## Resources

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)
- [Testing Zustand](https://docs.pmnd.rs/zustand/guides/testing)
- [Testing Axios](https://jestjs.io/docs/mock-functions)
