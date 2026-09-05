import type { HTMLAttributes, ReactNode } from "react";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  padded?: boolean;
}

export function Card({ children, padded = true, className = "", ...rest }: CardProps) {
  return (
    <div
      className={`rounded-xl2 border border-ink/10 bg-white/70 ${
        padded ? "p-6" : ""
      } ${className}`}
      {...rest}
    >
      {children}
    </div>
  );
}
