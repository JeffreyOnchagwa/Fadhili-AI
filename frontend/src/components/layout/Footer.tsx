import { Link } from "react-router-dom";
import { Logo } from "../common/Logo";

export function Footer() {
  return (
    <footer className="border-t border-ink/10 bg-ink/[0.02]">
      <div className="mx-auto max-w-6xl px-5 py-10">
        <div className="flex flex-col gap-8 sm:flex-row sm:items-start sm:justify-between">
          <div className="max-w-sm">
            <Logo />
            <p className="mt-3 text-sm text-ink-soft">
              Sign. Speak. Connect. Built for Kenya, designed for everyone.
            </p>
          </div>
          <nav aria-label="Footer" className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
            <Link to="/interpreter" className="text-ink-soft hover:text-ink">Interpreter</Link>
            <Link to="/translate" className="text-ink-soft hover:text-ink">Translate</Link>
            <Link to="/learn" className="text-ink-soft hover:text-ink">Learn</Link>
            <Link to="/dictionary" className="text-ink-soft hover:text-ink">Dictionary</Link>
            <Link to="/about" className="text-ink-soft hover:text-ink">About</Link>
            <Link to="/privacy" className="text-ink-soft hover:text-ink">Privacy &amp; terms</Link>
          </nav>
        </div>
        <p className="mt-8 text-xs text-ink-soft/80">
          Fadhili AI is experimental research software. KSL recognition is
          live but covers a small, verified vocabulary and can be
          confidently wrong — it is not a substitute for a qualified
          interpreter.
        </p>
      </div>
    </footer>
  );
}
