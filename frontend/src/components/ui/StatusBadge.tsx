import type { RecognitionStatus } from "../../types";

const STYLES: Record<
  RecognitionStatus,
  { dot: string; text: string; label: string }
> = {
  unavailable: { dot: "bg-ink/30", text: "text-ink-soft", label: "Not connected" },
  idle: { dot: "bg-ink/30", text: "text-ink-soft", label: "Idle" },
  loading: { dot: "bg-ochre-500 animate-pulse", text: "text-ochre-700", label: "Working" },
  success: { dot: "bg-teal-500", text: "text-teal-700", label: "Ready" },
  error: { dot: "bg-signal-red", text: "text-signal-red", label: "Error" },
};

export function StatusBadge({
  status,
  label,
}: {
  status: RecognitionStatus;
  label?: string;
}) {
  const style = STYLES[status];
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border border-ink/10 bg-ink/[0.03] px-3 py-1 text-xs font-semibold ${style.text}`}
      role="status"
    >
      <span className={`h-2 w-2 rounded-full ${style.dot}`} aria-hidden="true" />
      {label ?? style.label}
    </span>
  );
}
