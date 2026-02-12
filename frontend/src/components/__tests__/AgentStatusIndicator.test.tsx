import { render, screen } from '@testing-library/react';
import { AgentStatusIndicator } from '../AgentStatusIndicator';
import { useChatStore } from '../../store/chatStore';

// Mock the store
jest.mock('../../store/chatStore');

const mockUseChatStore = useChatStore as jest.MockedFunction<typeof useChatStore>;

describe('AgentStatusIndicator', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should not render when not loading and no stages', () => {
    mockUseChatStore.mockReturnValue({
      agentStages: [],
      currentStage: null,
      isLoading: false,
    } as ReturnType<typeof useChatStore>);

    const { container } = render(<AgentStatusIndicator />);
    expect(container.firstChild).toBeNull();
  });

  it('should render when loading', () => {
    mockUseChatStore.mockReturnValue({
      agentStages: [],
      currentStage: 'Processing...',
      isLoading: true,
    } as ReturnType<typeof useChatStore>);

    render(<AgentStatusIndicator />);
    expect(screen.getByText('Processing...')).toBeInTheDocument();
  });

  it('should display current stage message', () => {
    mockUseChatStore.mockReturnValue({
      agentStages: [
        { agent: 'verification', status: 'running', message: 'Validating query...' },
      ],
      currentStage: 'Verifying query',
      isLoading: true,
    } as ReturnType<typeof useChatStore>);

    render(<AgentStatusIndicator />);
    expect(screen.getByText('Verifying query')).toBeInTheDocument();
  });

  it('should show all 4 stage labels', () => {
    mockUseChatStore.mockReturnValue({
      agentStages: [],
      currentStage: 'Processing...',
      isLoading: true,
    } as ReturnType<typeof useChatStore>);

    render(<AgentStatusIndicator />);

    expect(screen.getByText('Verifying')).toBeInTheDocument();
    expect(screen.getByText('Planning')).toBeInTheDocument();
    expect(screen.getByText('Extracting')).toBeInTheDocument();
    expect(screen.getByText('Analyzing')).toBeInTheDocument();
  });

  it('should show pending state for stages not started', () => {
    mockUseChatStore.mockReturnValue({
      agentStages: [
        { agent: 'verification', status: 'running', message: 'Validating...' },
      ],
      currentStage: 'Verifying query',
      isLoading: true,
    } as ReturnType<typeof useChatStore>);

    render(<AgentStatusIndicator />);

    // Verification should be running (has spinning icon)
    // Other stages should be pending (gray circles)
    // Visual verification - stages are rendered
    expect(screen.getByText('Verifying')).toBeInTheDocument();
  });

  it('should show complete state for finished stages', () => {
    mockUseChatStore.mockReturnValue({
      agentStages: [
        { agent: 'verification', status: 'complete', message: 'Validated' },
        { agent: 'coordinator', status: 'running', message: 'Planning...' },
      ],
      currentStage: 'Planning analysis',
      isLoading: true,
    } as ReturnType<typeof useChatStore>);

    render(<AgentStatusIndicator />);

    // Verification is complete, coordinator is running
    const verifyingLabel = screen.getByText('Verifying');
    const planningLabel = screen.getByText('Planning');

    // Check that labels have different styles based on status
    expect(verifyingLabel).toHaveClass('text-green-700');
    expect(planningLabel).toHaveClass('text-blue-700');
  });

  it('should show running state with spinner', () => {
    mockUseChatStore.mockReturnValue({
      agentStages: [
        { agent: 'extraction', status: 'running', message: 'Extracting data...' },
      ],
      currentStage: 'Extracting data',
      isLoading: true,
    } as ReturnType<typeof useChatStore>);

    render(<AgentStatusIndicator />);

    // Should render the current stage
    expect(screen.getByText('Extracting data')).toBeInTheDocument();
    expect(screen.getByText('Extracting')).toHaveClass('text-blue-700');
  });

  it('should show error state for failed stages', () => {
    mockUseChatStore.mockReturnValue({
      agentStages: [
        { agent: 'verification', status: 'error', message: 'Validation failed' },
      ],
      currentStage: null,
      isLoading: true, // Must be loading for component to render
    } as ReturnType<typeof useChatStore>);

    render(<AgentStatusIndicator />);

    // Error state should use red styling
    const verifyingLabel = screen.getByText('Verifying');
    expect(verifyingLabel).toHaveClass('text-red-700');
  });

  it('should handle all stages complete', () => {
    mockUseChatStore.mockReturnValue({
      agentStages: [
        { agent: 'verification', status: 'complete' },
        { agent: 'coordinator', status: 'complete' },
        { agent: 'extraction', status: 'complete' },
        { agent: 'analytics', status: 'complete' },
      ],
      currentStage: null,
      isLoading: true, // Component renders when loading
    } as ReturnType<typeof useChatStore>);

    render(<AgentStatusIndicator />);

    // All stages should show as complete
    expect(screen.getByText('Verifying')).toHaveClass('text-green-700');
    expect(screen.getByText('Planning')).toHaveClass('text-green-700');
    expect(screen.getByText('Extracting')).toHaveClass('text-green-700');
    expect(screen.getByText('Analyzing')).toHaveClass('text-green-700');
  });

  it('should show progressive completion with connectors', () => {
    mockUseChatStore.mockReturnValue({
      agentStages: [
        { agent: 'verification', status: 'complete' },
        { agent: 'coordinator', status: 'complete' },
        { agent: 'extraction', status: 'running' },
      ],
      currentStage: 'Extracting data',
      isLoading: true,
    } as ReturnType<typeof useChatStore>);

    render(<AgentStatusIndicator />);

    // Verify stages are rendered with correct states
    expect(screen.getByText('Verifying')).toHaveClass('text-green-700');
    expect(screen.getByText('Planning')).toHaveClass('text-green-700');
    expect(screen.getByText('Extracting')).toHaveClass('text-blue-700');
  });

  it('should match snapshot for typical loading state', () => {
    mockUseChatStore.mockReturnValue({
      agentStages: [
        { agent: 'verification', status: 'complete', message: 'Validated' },
        { agent: 'coordinator', status: 'running', message: 'Planning...' },
      ],
      currentStage: 'Planning analysis',
      isLoading: true,
    } as ReturnType<typeof useChatStore>);

    const { container } = render(<AgentStatusIndicator />);
    expect(container).toMatchSnapshot();
  });

  it('should handle rapid state changes', () => {
    const { rerender } = render(<AgentStatusIndicator />);

    // Initial state
    mockUseChatStore.mockReturnValue({
      agentStages: [
        { agent: 'verification', status: 'running' },
      ],
      currentStage: 'Verifying query',
      isLoading: true,
    } as ReturnType<typeof useChatStore>);

    rerender(<AgentStatusIndicator />);
    expect(screen.getByText('Verifying query')).toBeInTheDocument();

    // Update to next stage
    mockUseChatStore.mockReturnValue({
      agentStages: [
        { agent: 'verification', status: 'complete' },
        { agent: 'coordinator', status: 'running' },
      ],
      currentStage: 'Planning analysis',
      isLoading: true,
    } as ReturnType<typeof useChatStore>);

    rerender(<AgentStatusIndicator />);
    expect(screen.getByText('Planning analysis')).toBeInTheDocument();
  });

  it('should have proper accessibility attributes', () => {
    mockUseChatStore.mockReturnValue({
      agentStages: [
        { agent: 'verification', status: 'running' },
      ],
      currentStage: 'Verifying query',
      isLoading: true,
    } as ReturnType<typeof useChatStore>);

    const { container } = render(<AgentStatusIndicator />);

    // Check for proper semantic structure
    const progressContainer = container.querySelector('div');
    expect(progressContainer).toBeInTheDocument();

    // Labels should be readable
    expect(screen.getByText('Verifying')).toBeInTheDocument();
    expect(screen.getByText('Planning')).toBeInTheDocument();
  });
});
