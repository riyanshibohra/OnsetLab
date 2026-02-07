// Message component

import type { Message as MessageType, PlanStep } from '../types';

interface Props {
  message: MessageType;
}

export function Message({ message }: Props) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-primary-500 text-white'
            : 'bg-white border border-gray-200 shadow-sm'
        }`}
      >
        {/* Message content */}
        <p className="whitespace-pre-wrap">{message.content}</p>

        {/* Plan visualization (for assistant messages) */}
        {!isUser && message.plan && message.plan.length > 0 && (
          <PlanViewer plan={message.plan} strategy={message.strategy} />
        )}
      </div>
    </div>
  );
}

function PlanViewer({
  plan,
  strategy,
}: {
  plan: PlanStep[];
  strategy?: string;
}) {
  return (
    <div className="mt-3 pt-3 border-t border-gray-100">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-medium text-gray-500 uppercase">
          {strategy === 'direct' ? 'Direct' : 'Plan'}
        </span>
        {strategy && strategy !== 'direct' && (
          <span className="text-xs px-2 py-0.5 bg-primary-100 text-primary-700 rounded-full">
            {strategy.toUpperCase()}
          </span>
        )}
      </div>

      <div className="space-y-2">
        {plan.map((step, idx) => (
          <div
            key={step.id}
            className="flex items-start gap-2 text-sm"
          >
            <span className="font-mono text-gray-400 text-xs mt-0.5">
              {step.id}
            </span>
            <div className="flex-1">
              <span className="font-medium text-primary-600">{step.tool}</span>
              <span className="text-gray-400">(</span>
              <span className="text-gray-600">
                {Object.entries(step.params)
                  .map(([k, v]) => `${k}="${v}"`)
                  .join(', ')}
              </span>
              <span className="text-gray-400">)</span>
              {step.result && (
                <span className="text-green-600 ml-2">→ {step.result}</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
