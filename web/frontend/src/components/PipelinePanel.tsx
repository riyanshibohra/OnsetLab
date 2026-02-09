import { useState } from 'react';
import type { PlanStep, PipelineTrace } from '../types';

type StepType = 'router' | 'planner' | 'executor' | 'solver' | 'react';
type StepStatus = 'completed' | 'running' | 'waiting';

interface PipelineStepData {
  type: StepType;
  label: string;
  status: StepStatus;
  content: React.ReactNode;
}

interface PipelinePanelProps {
  trace: PipelineTrace | null;
  plan: PlanStep[];
  results: Record<string, string>;
  strategy: string;
  answer: string;
  isLoading: boolean;
}

export function PipelinePanel({
  trace,
  plan,
  results,
  strategy,
  answer,
  isLoading,
}: PipelinePanelProps) {
  const isEmpty = !trace && !isLoading;

  const steps = buildSteps(trace, plan, results, strategy, answer, isLoading);

  return (
    <div className="pipeline-panel">
      <div className="pipeline-header">Pipeline</div>
      <div className="pipeline-content">
        {isEmpty ? (
          <div className="pipeline-empty">
            <div className="pipeline-empty-icon">⚡</div>
            <div>Run a query to see the<br />agent pipeline in action</div>
          </div>
        ) : (
          <div className="pipeline-steps">
            {steps.map((step, i) => (
              <PipelineStepCard
                key={`${step.type}-${i}`}
                step={step}
                delay={i * 100}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function PipelineStepCard({ step, delay }: { step: PipelineStepData; delay: number }) {
  const [expanded, setExpanded] = useState(true);

  return (
    <div
      className="pipeline-step"
      style={{ animationDelay: `${delay}ms` }}
    >
      {/* Timeline dot */}
      <div className={`pipeline-step-dot ${step.status}`}>
        {step.status === 'completed' && (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        )}
        {step.status === 'running' && (
          <div style={{
            width: 6, height: 6,
            borderRadius: '50%',
            background: 'var(--color-executor)',
          }} />
        )}
      </div>

      {/* Step card */}
      <div className={`pipeline-step-card type-${step.type}`}>
        <div
          className="pipeline-step-card-header"
          onClick={() => setExpanded(!expanded)}
        >
          <span className="step-label">{step.label}</span>
          <span className="step-status">
            {step.status === 'running' ? '● running' :
             step.status === 'completed' ? '✓' :
             '○ waiting'}
          </span>
        </div>
        {expanded && step.content && (
          <div className="pipeline-step-body">
            {step.content}
          </div>
        )}
      </div>
    </div>
  );
}

function buildSteps(
  trace: PipelineTrace | null,
  plan: PlanStep[],
  results: Record<string, string>,
  strategy: string,
  answer: string,
  isLoading: boolean,
): PipelineStepData[] {
  const steps: PipelineStepData[] = [];

  if (isLoading && !trace) {
    // Loading state: show skeleton steps
    steps.push({
      type: 'router',
      label: 'Router',
      status: 'running',
      content: (
        <div className="step-detail">
          <div className="pipeline-step-shimmer" style={{ width: '70%' }} />
          <div className="pipeline-step-shimmer" style={{ width: '50%', marginTop: 6 }} />
        </div>
      ),
    });
    steps.push({ type: 'planner', label: 'Planner', status: 'waiting', content: null });
    steps.push({ type: 'executor', label: 'Executor', status: 'waiting', content: null });
    steps.push({ type: 'solver', label: 'Solver', status: 'waiting', content: null });
    return steps;
  }

  if (!trace) return steps;

  // 1. Router
  const isDirect = strategy === 'direct' || trace.router_decision === 'DIRECT';
  steps.push({
    type: 'router',
    label: 'Router',
    status: 'completed',
    content: (
      <div className="step-detail">
        <div className="step-detail-label">Decision</div>
        <div className="step-detail-value">
          {isDirect ? 'DIRECT' : 'TOOL'} {trace.router_reason ? `\n${trace.router_reason}` : ''}
        </div>
        {trace.tools_selected.length > 0 && (
          <>
            <div className="step-detail-label" style={{ marginTop: 10 }}>Tools matched</div>
            <div className="step-detail-value">
              {trace.tools_selected.join(', ')}
            </div>
          </>
        )}
      </div>
    ),
  });

  // If DIRECT, skip to solver
  if (isDirect) {
    steps.push({
      type: 'solver',
      label: 'Answer',
      status: answer ? 'completed' : 'running',
      content: answer ? (
        <div className="step-detail">
          <div className="step-detail-label">Response</div>
          <div className="step-detail-value">{answer}</div>
        </div>
      ) : null,
    });
    return steps;
  }

  // 2. Planner
  const hasPlanner = trace.planner_think || plan.length > 0;
  steps.push({
    type: 'planner',
    label: 'Planner',
    status: hasPlanner ? 'completed' : (isLoading ? 'running' : 'waiting'),
    content: hasPlanner ? (
      <div className="step-detail">
        {trace.planner_think && (
          <>
            <div className="step-detail-label">Reasoning</div>
            <div className="step-detail-value">{trace.planner_think}</div>
          </>
        )}
        {plan.length > 0 && (
          <>
            <div className="step-detail-label" style={{ marginTop: 10 }}>Plan</div>
            <div className="step-detail-value">
              {plan.map((step) => {
                const paramsStr = Object.entries(step.params)
                  .map(([k, v]) => `${k}="${v}"`)
                  .join(', ');
                return `${step.id} = ${step.tool}(${paramsStr})`;
              }).join('\n')}
            </div>
          </>
        )}
      </div>
    ) : null,
  });

  // 3. Executor
  const hasResults = Object.keys(results).length > 0;
  steps.push({
    type: 'executor',
    label: 'Executor',
    status: hasResults ? 'completed' : (hasPlanner && isLoading ? 'running' : 'waiting'),
    content: hasResults ? (
      <div className="step-detail">
        {plan.map((step) => (
          <div key={step.id} className="tool-call-row">
            <div className="tool-call-name">
              {step.id}: {step.tool}
            </div>
            <div className="tool-call-params">
              {Object.entries(step.params)
                .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                .join(', ')}
            </div>
            {results[step.id] && (
              <div className="tool-call-result">
                → {results[step.id]}
              </div>
            )}
          </div>
        ))}
      </div>
    ) : null,
  });

  // 4. ReAct Fallback (if triggered)
  if (trace.fallback_used) {
    steps.push({
      type: 'react',
      label: 'ReAct Fallback',
      status: answer ? 'completed' : 'running',
      content: (
        <div className="step-detail">
          <div className="step-detail-label">Reason</div>
          <div className="step-detail-value">
            {trace.fallback_reason || 'REWOO plan failed, switching to iterative reasoning'}
          </div>
        </div>
      ),
    });
  }

  // 5. Solver
  steps.push({
    type: 'solver',
    label: 'Solver',
    status: answer ? 'completed' : (hasResults && isLoading ? 'running' : 'waiting'),
    content: answer ? (
      <div className="step-detail">
        <div className="step-detail-label">Final answer</div>
        <div className="step-detail-value">{answer}</div>
      </div>
    ) : null,
  });

  return steps;
}
