import type { ReactNode } from "react";
import { Hand, Cpu, MessageSquare, Volume2 } from "lucide-react";

function Track({
  reverse,
  fromLabel,
  fromIcon,
  toLabel,
  toIcon,
}: {
  reverse?: boolean;
  fromLabel: string;
  fromIcon: ReactNode;
  toLabel: string;
  toIcon: ReactNode;
}) {
  return (
    <div className="flex items-center gap-4">
      <div className="flex w-24 flex-col items-center gap-2 text-center">
        <div className="flex h-14 w-14 items-center justify-center rounded-full bg-white border border-ink/10 text-teal-600">
          {fromIcon}
        </div>
        <span className="text-xs font-semibold text-ink-soft">{fromLabel}</span>
      </div>

      <div className="relative h-px flex-1 bg-ink/10">
        <span
          className="absolute top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-ochre-500 motion-reduce:hidden animate-flow-dot"
          style={{ animationDirection: reverse ? "reverse" : "normal" }}
          aria-hidden="true"
        />
      </div>

      <div className="flex w-16 flex-col items-center gap-2 text-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-ink text-paper">
          <Cpu size={20} aria-hidden="true" />
        </div>
        <span className="text-xs font-semibold text-ink-soft">AI</span>
      </div>

      <div className="relative h-px flex-1 bg-ink/10">
        <span
          className="absolute top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-ochre-500 motion-reduce:hidden animate-flow-dot"
          style={{ animationDirection: reverse ? "reverse" : "normal", animationDelay: "1.1s" }}
          aria-hidden="true"
        />
      </div>

      <div className="flex w-24 flex-col items-center gap-2 text-center">
        <div className="flex h-14 w-14 items-center justify-center rounded-full bg-white border border-ink/10 text-teal-600">
          {toIcon}
        </div>
        <span className="text-xs font-semibold text-ink-soft">{toLabel}</span>
      </div>
    </div>
  );
}

/**
 * Illustrates the concept of bidirectional translation. This is a
 * static diagram, not a live demo — it must never be mistaken for an
 * actual running recognition or generation process.
 */
export function FlowVisualization() {
  return (
    <div
      className="rounded-xl2 border border-ink/10 bg-paper-dim/60 bg-topo p-6 sm:p-8"
      role="img"
      aria-label="Diagram: sign language converts to text and speech, and text or speech converts to sign, both through Fadhili AI"
    >
      <div className="flex flex-col gap-8">
        <Track
          fromLabel="Sign"
          fromIcon={<Hand size={22} aria-hidden="true" />}
          toLabel="Text + Speech"
          toIcon={<MessageSquare size={20} aria-hidden="true" />}
        />
        <Track
          reverse
          fromLabel="Text + Speech"
          fromIcon={<Volume2 size={22} aria-hidden="true" />}
          toLabel="Sign"
          toIcon={<Hand size={20} aria-hidden="true" />}
        />
      </div>
      <p className="mt-6 text-center text-xs text-ink-soft">
        A concept diagram — live recognition is not yet running in this build.
      </p>
    </div>
  );
}
