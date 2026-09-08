import {
  Hand,
  MessageSquareText,
  Volume2,
  Type,
  Mic,
  GraduationCap,
  Camera,
} from "lucide-react";
import { ButtonLink } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { FlowVisualization } from "../components/home/FlowVisualization";

/**
 * What Fadhili actually does today. Each description is written against
 * the shipped behaviour, not the ambition — an earlier version of this
 * list promised "structured courses from your first sign to full
 * fluency" and camera practice that "gives feedback as you go", neither
 * of which exists.
 */
const FEATURES = [
  {
    icon: Hand,
    title: "KSL → text",
    description:
      "Reads a small vocabulary of Kenyan Sign Language from your camera and writes it down.",
  },
  {
    icon: Volume2,
    title: "KSL → speech",
    description:
      "Speaks each sign aloud once the reading settles, if you switch speech on.",
  },
  {
    icon: Type,
    title: "Text → KSL",
    description:
      "Maps what you type onto the signs we hold verified recordings for, and names the words we don't.",
  },
  {
    icon: Mic,
    title: "Speech → KSL",
    description:
      "Transcribes what you say in the browser, then maps it the same way.",
  },
  {
    icon: GraduationCap,
    title: "Learn KSL",
    description:
      "Browse and search the verified vocabulary, grouped into lessons by topic.",
  },
  {
    icon: Camera,
    title: "Check a sign",
    description:
      "See whether the model reads your signing the way you intended. It cannot tell you that you are correct.",
  },
];

export default function Home() {
  return (
    <>
      <section className="mx-auto max-w-6xl px-5 pb-16 pt-14 sm:pt-20">
        <div className="grid items-center gap-12 lg:grid-cols-[1.05fr_1fr]">
          <div>
            <span className="inline-flex items-center gap-2 rounded border border-ink/10 bg-white px-3.5 py-1.5 text-xs font-semibold text-ink-soft">
              Built for Kenya. Designed for everyone.
            </span>
            <h1 className="mt-5 text-5xl font-medium leading-[1.05] tracking-tight sm:text-6xl">
              Fadhili AI
            </h1>
            <p className="mt-4 font-display text-2xl italic text-teal-700 sm:text-3xl">
              Communication without barriers.
            </p>
            <p className="mt-5 max-w-lg text-lg leading-relaxed text-ink-soft">
              AI-powered tools designed to bridge communication between Deaf
              and hearing communities through sign, text and speech.
            </p>
            <div className="mt-8 flex flex-col gap-3 sm:flex-row">
              <ButtonLink to="/interpreter" size="lg" className="w-full sm:w-auto">
                Start interpreting
              </ButtonLink>
              <ButtonLink
                to="/learn"
                size="lg"
                variant="secondary"
                className="w-full sm:w-auto"
              >
                Learn KSL
              </ButtonLink>
            </div>
            <p className="mt-3 text-sm text-ink-soft">
              Sign. Speak. Connect.
            </p>
          </div>

          <FlowVisualization />
        </div>
      </section>

      <section aria-labelledby="ksl-first-heading" className="border-y border-ink/10 bg-ink text-paper">
        <div className="mx-auto max-w-6xl px-5 py-14 sm:py-16">
          <div className="grid gap-8 lg:grid-cols-[1fr_1.1fr] lg:items-center">
            <div>
              <span className="text-sm font-semibold text-ochre-300">
                Kenyan Sign Language only
              </span>
              <h2 id="ksl-first-heading" className="mt-3 text-3xl font-medium sm:text-4xl">
                Built around KSL, from the ground up.
              </h2>
            </div>
            <p className="text-lg leading-relaxed text-paper/80">
              Most assistive technology for signed languages is built around
              American Sign Language, and Kenyan Sign Language is left as an
              afterthought. Fadhili works on KSL and nothing else. Doing one
              signed language properly is harder, and more useful, than
              claiming several.
            </p>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-5 py-16 sm:py-20">
        <h2 className="text-3xl font-medium sm:text-4xl">What Fadhili AI does</h2>
        <p className="mt-3 max-w-xl text-ink-soft">
          Interpreting, translating and learning, over the Kenyan Sign
          Language vocabulary Fadhili holds real recordings for.
        </p>
        <div className="mt-10 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((feature) => (
            <Card key={feature.title} className="transition-shadow hover:shadow-soft">
              <feature.icon size={22} className="text-teal-600" aria-hidden="true" />
              <h3 className="mt-4 text-lg font-medium">{feature.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-soft">
                {feature.description}
              </p>
            </Card>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-5 pb-20">
        <Card className="flex flex-col items-start gap-5 bg-teal-50 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <MessageSquareText size={22} className="text-teal-600 shrink-0" aria-hidden="true" />
            <p className="text-base text-ink">
              Fadhili is experimental research software. It reads a small
              vocabulary, can be confidently wrong, and is not a substitute
              for a qualified KSL interpreter.
            </p>
          </div>
          <ButtonLink to="/about" variant="secondary" className="whitespace-nowrap">
            Read our approach
          </ButtonLink>
        </Card>
      </section>
    </>
  );
}
