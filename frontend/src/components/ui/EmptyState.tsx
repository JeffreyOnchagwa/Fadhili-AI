import type { ReactNode } from "react";

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon?: ReactNode;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-xl2 border border-dashed border-ink/15 bg-ink/[0.02] px-6 py-12 text-center">
      {icon && <div className="text-ink/30" aria-hidden="true">{icon}</div>}
      <p className="font-display text-lg text-ink">{title}</p>
      <p className="max-w-sm text-sm text-ink-soft">{description}</p>
      {action}
    </div>
  );
}
