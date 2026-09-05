import { Link } from "react-router-dom";
import {
  Hand,
  MessageSquareText,
  Volume2,
  Type,
  Mic,
  GraduationCap,
  Camera,
} from "lucide-react";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { FlowVisualization } from "../components/home/FlowVisualization";

const FEATURES = [
  {
    icon: Hand,
    title: "Sign → Text",
    description: "Turn signed conversation into written text for anyone reading along.",
  },
  {
    icon: Volume2,
    title: "Sign → Speech",
    description: "Give signed conversation a spoken voice in real time.",
  },
  {
    icon: Type,
    title: "Text → Sign",
    description: "Type a message and see it rendered as sign language.",
  },
  {
    icon: Mic,
    title: "Speech → Sign",
    description: "Speak, and Fadhili AI carries your words into sign.",
  },
  {
    icon: GraduationCap,
    title: "Learn sign language",
    description: "Structured courses from your first sign to full fluency.",
  },
  {
    icon: Camera,
    title: "AI practice",
    description: "Practice in front of your camera and get feedback as you go.",
  },
];

export default function Home() {
  return (
    <>
      <section className="mx-auto max-w-6xl px-5 pb-16 pt-14 sm:pt-20">
        <div className="grid items-center gap-12 lg:grid-cols-[1.05fr_1fr]">
          <div>
            <span className="inline-flex items-center gap-2 rounded-full border border-ink/10 bg-white px-3.5 py-1.5 text-xs font-semibold text-ink-soft">
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
              <Link to="/interpreter">
                <Button size="lg" className="w-full sm:w-auto">Start Interpreting</Button>
              </Link>
              <Link to="/learn">
                <Button size="lg" variant="secondary" className="w-full sm:w-auto">
                  Learn Sign Language
                </Button>
              </Link>
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
              <span className="text-sm font-semibold text-ochre-300">🇰🇪 Kenyan Sign Language first</span>
              <h2 id="ksl-first-heading" className="mt-3 text-3xl font-medium sm:text-4xl">
                Built around KSL, from the ground up.
              </h2>
            </div>
            <p className="text-lg leading-relaxed text-paper/80">
              Kenyan Sign Language is Fadhili AI's primary focus and its
              default language everywhere in the app. American Sign Language
              and British Sign Language are supported alongside it, so the
              platform can grow with the communities that use it — without
              treating KSL as an afterthought.
            </p>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-5 py-16 sm:py-20">
        <h2 className="text-3xl font-medium sm:text-4xl">What Fadhili AI does</h2>
        <p className="mt-3 max-w-xl text-ink-soft">
          One platform for interpreting, translating, and learning — built to
          grow with real ML models as they're connected.
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
              Recognition and sign-generation are actively being developed —
              this build is honest about what's connected and what isn't.
            </p>
          </div>
          <Link to="/about">
            <Button variant="secondary" className="whitespace-nowrap">Read our approach</Button>
          </Link>
        </Card>
      </section>
    </>
  );
}
