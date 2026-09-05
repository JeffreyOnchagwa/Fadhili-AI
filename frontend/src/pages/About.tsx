import { Target, Cpu, Accessibility } from "lucide-react";
import { Card } from "../components/ui/Card";

const SECTIONS = [
  {
    icon: Target,
    title: "Our mission",
    body:
      "Fadhili AI exists to reduce communication barriers between Deaf and hearing communities, using AI to make sign language more accessible in everyday life — at school, at work, and at home.",
  },
  {
    icon: Cpu,
    title: "Technology",
    body:
      "Fadhili AI is being built as a full-stack platform: a React frontend and a Python machine-learning backend for sign recognition and generation. The frontend you're using today is complete; the ML backend is being developed separately and connected in stages, with honest status indicators throughout.",
  },
  {
    icon: Accessibility,
    title: "Accessibility",
    body:
      "Accessibility isn't a feature here — it's the point. Fadhili AI is built with keyboard navigation, screen-reader support, visible focus states, and captions-ready video from the start, so the tool itself doesn't create new barriers.",
  },
];

export default function About() {
  return (
    <div className="mx-auto max-w-4xl px-5 py-12">
      <header>
        <h1 className="text-3xl font-medium sm:text-4xl">About Fadhili AI</h1>
        <p className="mt-3 max-w-2xl text-ink-soft">
          Fadhili AI is under active development. This page describes what
          we're building and why — not a finished, clinically or legally
          validated product.
        </p>
      </header>

      <div className="mt-10 flex flex-col gap-5">
        {SECTIONS.map((section) => (
          <Card key={section.title} className="flex gap-4">
            <section.icon size={22} className="mt-1 shrink-0 text-teal-600" aria-hidden="true" />
            <div>
              <h2 className="text-xl font-medium">{section.title}</h2>
              <p className="mt-2 leading-relaxed text-ink-soft">{section.body}</p>
            </div>
          </Card>
        ))}

        <Card className="bg-ink text-paper">
          <h2 className="text-xl font-medium">Why Kenyan Sign Language</h2>
          <p className="mt-2 leading-relaxed text-paper/80">
            Kenyan Sign Language is used by a large Deaf community that is
            underserved by mainstream assistive technology, most of which is
            built around ASL. Fadhili AI puts KSL first — as the default
            language and the first target for the recognition and generation
            models — while still building toward ASL and BSL support.
          </p>
        </Card>

        <Card className="border-ochre-300/60 bg-ochre-100/60">
          <p className="text-sm leading-relaxed text-ink">
            Fadhili AI is not a certified interpreter and doesn't provide
            medical, legal, or interpretation-grade reliability. It's a tool
            to support communication and learning, developed openly and
            improved over time.
          </p>
        </Card>
      </div>
    </div>
  );
}
