import { useState } from "react";
import { NavLink } from "react-router-dom";
import { Menu, X } from "lucide-react";
import { Logo } from "../common/Logo";

const LINKS = [
  { to: "/", label: "Home" },
  { to: "/interpreter", label: "Interpreter" },
  { to: "/translate", label: "Translate" },
  { to: "/learn", label: "Learn" },
  { to: "/dictionary", label: "Dictionary" },
  { to: "/about", label: "About" },
];

export function Navbar() {
  const [open, setOpen] = useState(false);

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `rounded-full px-3.5 py-2 text-sm font-semibold transition-colors ${
      isActive ? "bg-ink text-paper" : "text-ink-soft hover:text-ink"
    }`;

  return (
    <header className="sticky top-0 z-30 border-b border-ink/10 bg-paper/90 backdrop-blur">
      <a href="#main-content" className="sr-only-focusable">
        Skip to main content
      </a>
      <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3">
        <NavLink to="/" aria-label="Fadhili AI home">
          <Logo />
        </NavLink>

        <nav
          className="hidden items-center gap-1 lg:flex"
          aria-label="Primary"
        >
          {LINKS.map((link) => (
            <NavLink key={link.to} to={link.to} className={linkClass} end={link.to === "/"}>
              {link.label}
            </NavLink>
          ))}
        </nav>

        <button
          type="button"
          className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-full border border-ink/15 lg:hidden"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-controls="mobile-nav"
          aria-label={open ? "Close menu" : "Open menu"}
        >
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {open && (
        <nav
          id="mobile-nav"
          aria-label="Primary"
          className="border-t border-ink/10 bg-paper px-5 pb-5 pt-2 lg:hidden"
        >
          <div className="flex flex-col gap-1">
            {LINKS.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.to === "/"}
                onClick={() => setOpen(false)}
                className={({ isActive }) =>
                  `min-h-[44px] rounded-lg px-3 py-3 text-base font-semibold ${
                    isActive ? "bg-ink text-paper" : "text-ink-soft"
                  }`
                }
              >
                {link.label}
              </NavLink>
            ))}
          </div>
        </nav>
      )}
    </header>
  );
}
