import { useEffect, useRef } from "react";
import type { RefObject } from "react";
import type { LiveLandmarks } from "../../hooks/useKSLRecognition";

interface Props {
  videoRef: RefObject<HTMLVideoElement | null>;
  landmarksRef: RefObject<LiveLandmarks | null>;
  active: boolean;
}

// Pose connections among the 6 landmarks the model actually uses
// (shoulders, elbows, wrists) — matches POSE_INDICES in
// mediapipeFeatures.ts, drawn at their true MediaPipe indices.
const POSE_BONES: [number, number][] = [
  [11, 12], // shoulder to shoulder
  [11, 13], // left shoulder to elbow
  [13, 15], // left elbow to wrist
  [12, 14], // right shoulder to elbow
  [14, 16], // right elbow to wrist
];

// Hand skeleton: thumb, index, middle, ring, pinky, plus the palm base.
const HAND_BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

/**
 * Draws the MediaPipe skeleton over the live camera feed.
 *
 * This is not decoration. The diagnosed failure mode of the KSL model
 * is that MediaPipe silently loses a hand under certain lighting,
 * clothing and framing — signer-level hand-detection rates in the
 * training data ranged from 7% missing to 62% missing depending on
 * exactly this. A user whose hand keeps dropping out has no way to
 * know that's what's happening; a static "no sign detected" message
 * looks identical whether the camera can't see the sign or can't see
 * the hand at all. This overlay makes that failure visible and
 * actionable: an un-drawn hand means "move so MediaPipe can see it",
 * not "sign more clearly".
 *
 * Deliberately plain 2D canvas rather than WebGL/3D: the interpreter
 * already runs MediaPipe inference on a 2-core CPU with no GPU, and
 * every extra millisecond here competes with that. A skeleton is a
 * skeleton whether it's drawn in three dimensions or two; a 3D
 * renderer would add real cost for no added clarity.
 */
export function LandmarkOverlay({ videoRef, landmarksRef, active }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!active) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let rafId: number;
    let cancelled = false;

    function draw() {
      if (cancelled) return;
      rafId = requestAnimationFrame(draw);

      const video = videoRef.current;
      if (!video || !video.videoWidth || !video.videoHeight) return;
      if (!canvas || !ctx) return;

      if (
        canvas.width !== video.clientWidth ||
        canvas.height !== video.clientHeight
      ) {
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const data = landmarksRef.current;
      if (!data) return;

      // object-cover crops the video to fill the element; landmarks
      // must be mapped through the same crop or they'd drift off the
      // visible hand as soon as the aspect ratios differ.
      const videoAspect = video.videoWidth / video.videoHeight;
      const boxAspect = canvas.width / canvas.height;

      let scale: number;
      let offsetX = 0;
      let offsetY = 0;

      if (videoAspect > boxAspect) {
        scale = canvas.height / video.videoHeight;
        offsetX = (canvas.width - video.videoWidth * scale) / 2;
      } else {
        scale = canvas.width / video.videoWidth;
        offsetY = (canvas.height - video.videoHeight * scale) / 2;
      }

      const project = (nx: number, ny: number): [number, number] => {
        const px = nx * video.videoWidth * scale + offsetX;
        const py = ny * video.videoHeight * scale + offsetY;
        return [px, py];
      };

      const drawBones = (
        points: { x: number; y: number }[] | null,
        bones: [number, number][],
        color: string,
        radius: number
      ) => {
        if (!points) return;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        for (const [a, b] of bones) {
          const pa = points[a];
          const pb = points[b];
          if (!pa || !pb) continue;
          const [ax, ay] = project(pa.x, pa.y);
          const [bx, by] = project(pb.x, pb.y);
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
        }

        ctx.fillStyle = color;
        for (const point of points) {
          const [x, y] = project(point.x, point.y);
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);
          ctx.fill();
        }
      };

      // Full 33-point pose is drawn faintly for context; the 6 the
      // model uses are implied by POSE_BONES among them.
      drawBones(data.pose, POSE_BONES, "rgba(242,239,231,0.85)", 4);
      drawBones(data.leftHand, HAND_BONES, "rgba(200,155,60,0.95)", 2.5);
      drawBones(data.rightHand, HAND_BONES, "rgba(127,179,160,0.95)", 2.5);
    }

    draw();

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
    };
  }, [active, videoRef, landmarksRef]);

  if (!active) return null;

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 h-full w-full"
    />
  );
}
