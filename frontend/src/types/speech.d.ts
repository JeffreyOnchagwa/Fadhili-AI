export {};

declare global {
  interface SpeechRecognitionResultAlternative {
    transcript: string;
    confidence: number;
  }

  interface SpeechRecognitionResult {
    readonly length: number;
    item(index: number): SpeechRecognitionResultAlternative;
    [index: number]: SpeechRecognitionResultAlternative;
    isFinal: boolean;
  }

  interface SpeechRecognitionResultList {
    readonly length: number;
    item(index: number): SpeechRecognitionResult;
    [index: number]: SpeechRecognitionResult;
  }

  interface SpeechRecognitionEvent extends Event {
    resultIndex: number;
    results: SpeechRecognitionResultList;
  }

  interface SpeechRecognitionErrorEvent extends Event {
    error: string;
    message: string;
  }

  interface SpeechRecognition extends EventTarget {
    lang: string;
    continuous: boolean;
    interimResults: boolean;
    maxAlternatives: number;
    start(): void;
    stop(): void;
    abort(): void;
    onresult: ((event: SpeechRecognitionEvent) => void) | null;
    onend: ((event: Event) => void) | null;
    onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
    onstart: ((event: Event) => void) | null;
  }

  interface Window {
    SpeechRecognition?: new () => SpeechRecognition;
    webkitSpeechRecognition?: new () => SpeechRecognition;
  }
}
