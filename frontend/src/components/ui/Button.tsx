import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  icon?: ReactNode;
  children: ReactNode;
}

const VARIANT_CLASSES: Record<Variant, string> = {
  primary:
    "bg-teal-500 text-paper hover:bg-teal-600 disabled:bg-teal-300 disabled:text-paper/70",
  secondary:
    "bg-transparent text-ink border border-ink/20 hover:border-ink/40 hover:bg-ink/[0.03] disabled:opacity-40",
  ghost:
    "bg-transparent text-teal-700 hover:bg-teal-50 disabled:opacity-40",
  danger:
    "bg-signal-red text-paper hover:bg-[#8f342d] disabled:opacity-40",
};

const SIZE_CLASSES: Record<Size, string> = {
  md: "text-sm px-4 py-2.5 gap-2",
  lg: "text-base px-6 py-3.5 gap-2.5",
};

export function Button({
  variant = "primary",
  size = "md",
  icon,
  children,
  className = "",
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={`inline-flex items-center justify-center rounded-full font-semibold
        transition-colors duration-150 disabled:cursor-not-allowed
        min-h-[44px] ${VARIANT_CLASSES[variant]} ${SIZE_CLASSES[size]} ${className}`}
      disabled={disabled}
      {...rest}
    >
      {icon}
      {children}
    </button>
  );
}
