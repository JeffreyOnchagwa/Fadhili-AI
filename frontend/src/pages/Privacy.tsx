import { Card } from "../components/ui/Card";

/**
 * Privacy & terms.
 *
 * Written from the actual audit in docs/PRIVACY_AND_SECURITY.md rather
 * than a generic template — every claim below is something verified in
 * this codebase (grep for cookies/localStorage/analytics, a read of the
 * request-handling code), not boilerplate. No cookie banner appears on
 * this site because nothing here requires one: see "Cookies" below.
 *
 * This is not a substitute for legal advice. Two things flagged here
 * explicitly need professional review before a public, EU-facing
 * launch: Google Fonts (loads from Google's servers on every visit)
 * and the Web Speech API (sends audio to the browser vendor's servers
 * when you use the microphone in Chrome or Edge).
 */
export default function Privacy() {
  return (
    <div className="mx-auto max-w-3xl px-5 py-12">
      <header>
        <h1 className="text-3xl font-medium sm:text-4xl">Privacy &amp; terms</h1>
        <p className="mt-3 text-ink-soft">
          What Fadhili does with your camera, microphone and data — and what
          it doesn&rsquo;t do. Last reviewed 2026-09-08.
        </p>
      </header>

      <div className="mt-10 flex flex-col gap-8">
        <section>
          <h2 className="text-xl font-medium">Your camera never leaves your device</h2>
          <p className="mt-2 leading-relaxed text-ink-soft">
            The Interpreter runs MediaPipe hand and body tracking directly in
            your browser. No video frame is ever sent anywhere. What reaches
            Fadhili&rsquo;s server is a short sequence of numeric joint
            positions — not an image, and nothing that on its own identifies
            you.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-medium">What the server stores: nothing</h2>
          <p className="mt-2 leading-relaxed text-ink-soft">
            There is no database, no request logging of what you sent, and no
            file written for any prediction. A landmark sequence exists in
            memory for the length of one request and is then gone. Your
            recordings are never used to train the model — there is no
            mechanism in this codebase by which that could happen.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-medium">Microphone (Translate page)</h2>
          <p className="mt-2 leading-relaxed text-ink-soft">
            Speech-to-text uses your browser&rsquo;s built-in speech
            recognition, only after you press &ldquo;Start listening.&rdquo;
            Fadhili receives only the transcribed text. In Chrome and Edge,
            that transcription is performed by sending your audio to the
            browser vendor&rsquo;s own servers — that happens inside the
            browser, outside Fadhili&rsquo;s control, and is worth knowing
            even though Fadhili itself never receives or stores the audio.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-medium">Cookies: none, so no banner</h2>
          <p className="mt-2 leading-relaxed text-ink-soft">
            Fadhili sets no cookies and stores nothing in your browser&rsquo;s
            local storage. There is no analytics, no advertising and no
            tracking script anywhere on this site. A cookie-consent banner
            would ask you to agree to something that doesn&rsquo;t happen, so
            none is shown. If that ever changes — if analytics or embedded
            third-party media are added — this page and that behaviour will
            change with it.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-medium">Third parties this site loads from</h2>
          <p className="mt-2 leading-relaxed text-ink-soft">
            Two typefaces (Fraunces and Manrope) load from Google Fonts on
            every visit, which exposes your IP address and browser
            information to Google the way any font CDN does. Nothing else —
            no analytics platforms, no ad networks, no social embeds, no
            session-recording tools.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-medium">Training data</h2>
          <p className="mt-2 leading-relaxed text-ink-soft">
            The recognition model was trained on the eKitabu Kenyan Sign
            Language Video Dataset, showing 15 consenting participants
            recorded for that dataset, released under a CC0-1.0 public-domain
            licence. Their footage is used to train the model; it is not
            shown anywhere in this app.
          </p>
        </section>

        <Card className="border-ochre-300/60 bg-ochre-100/60">
          <h2 className="text-xl font-medium text-ink">Terms of use</h2>
          <p className="mt-2 text-sm leading-relaxed text-ink">
            Fadhili is experimental research software, offered as-is and free
            of charge. It is not a certified interpreter and must not be
            relied on for medical, legal, safety or other situations where an
            incorrect reading could cause harm. Recognition covers a small
            vocabulary and can be confidently wrong. Nothing is sold through
            this site, so no refund or payment terms apply.
          </p>
        </Card>

        <p className="text-xs text-ink-soft">
          Fuller technical detail, including what was actually checked and
          how, lives in this project&rsquo;s public repository under
          <code className="mx-1 rounded bg-ink/5 px-1 py-0.5">docs/PRIVACY_AND_SECURITY.md</code>.
        </p>
      </div>
    </div>
  );
}
