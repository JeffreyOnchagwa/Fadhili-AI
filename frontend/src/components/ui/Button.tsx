import { Link } from "react-router-dom";
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
    "bg-teal-600 text-paper hover:bg-teal-700 disabled:bg-teal-300 disabled:text-paper/70",
  secondary:
    "bg-transparent text-ink border border-ink/20 hover:border-ink/40 hover:bg-ink/[0.03] disabled:opacity-40",
  ghost: "bg-transparent text-teal-700 hover:bg-teal-50 disabled:opacity-40",
  danger: "bg-signal-red text-paper hover:bg-[#8f342d] disabled:opacity-40",
};

const SIZE_CLASSES: Record<Size, string> = {
  md: "text-sm px-4 py-2.5 gap-2",
  lg: "text-base px-6 py-3.5 gap-2.5",
};

// Shared shape. Deliberately a moderate radius rather than a full pill:
// capsule buttons everywhere are a template tell, and at these widths a
// pill also wastes horizontal space. A visible focus ring is included
// here rather than left to the browser default, which is easy to lose
// against the teal and ink fills.
const BASE_CLASSES = `inline-flex items-center justify-center rounded-lg font-semibold
  transition-colors duration-150 disabled:cursor-not-allowed min-h-[44px]
  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600
  focus-visible:ring-offset-2 focus-visible:ring-offset-paper`;

export function Button({
  variant = "primary",
  size = "md",
  icon,
  children,
  className = "",
  disabled,
  type = "button",
  ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      className={`${BASE_CLASSES} ${VARIANT_CLASSES[variant]} ${SIZE_CLASSES[size]} ${className}`}
      disabled={disabled}
      {...rest}
    >
      {icon}
      {children}
    </button>
  );
}

interface ButtonLinkProps {
  to: string;
  variant?: Variant;
  size?: Size;
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
}

/**
 * A link that looks like a button.
 *
 * Exists because navigation was previously written as
 * `<Link><Button/></Link>`, which nests a <button> inside an <a>. That
 * is invalid HTML and genuinely breaks assistive technology and
 * keyboard behaviour, since the two interactive elements compete. Use
 * ButtonLink when the action navigates, and Button when it performs
 * something on the page.
 */
export function ButtonLink({
  to,
  variant = "primary",
  size = "md",
  icon,
  children,
  className = "",
}: ButtonLinkProps) {
  return (
    <Link
      to={to}
      className={`${BASE_CLASSES} ${VARIANT_CLASSES[variant]} ${SIZE_CLASSES[size]} ${className}`}
    >
      {icon}
      {children}
    </Link>
  );
}
