export function Logo({ withWordmark = true }: { withWordmark?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2.5">
      <svg
        width="28"
        height="28"
        viewBox="0 0 32 32"
        aria-hidden="true"
        className="shrink-0"
      >
        <rect width="32" height="32" rx="8" fill="#14181A" />
        <path
          d="M10 20.5V11a1.5 1.5 0 0 1 3 0v5.5"
          stroke="#F2EFE7"
          strokeWidth="1.6"
          strokeLinecap="round"
          fill="none"
        />
        <path
          d="M13 16.5V9a1.5 1.5 0 0 1 3 0v7.5"
          stroke="#F2EFE7"
          strokeWidth="1.6"
          strokeLinecap="round"
          fill="none"
        />
        <path
          d="M16 16.5V9.8a1.5 1.5 0 0 1 3 0v7"
          stroke="#F2EFE7"
          strokeWidth="1.6"
          strokeLinecap="round"
          fill="none"
        />
        <path
          d="M19 17V12a1.4 1.4 0 0 1 2.8 0v8.2c0 3-2.2 5.3-5.3 5.3h-1.1c-1.8 0-3.1-.6-4.2-2l-3-3.9c-.6-.8-.4-1.9.5-2.4.7-.4 1.6-.3 2.2.3l1.8 1.7"
          stroke="#C89B3C"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
      </svg>
      {withWordmark && (
        <span className="font-display text-lg font-medium tracking-tight text-ink">
          Fadhili AI
        </span>
      )}
    </span>
  );
}
