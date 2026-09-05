import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";

export type CameraPermissionState =
  | "idle"
  | "requesting"
  | "granted"
  | "denied"
  | "unavailable"
  | "error";

interface UseCameraResult {
  videoRef: RefObject<HTMLVideoElement | null>;
  permission: CameraPermissionState;
  isActive: boolean;
  errorMessage: string | null;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
}

/**
 * Manages a real getUserMedia camera stream: requesting permission,
 * attaching the stream to a <video> element, and reliably stopping
 * every track on demand or on unmount so the camera light turns off.
 */
export function useCamera(): UseCameraResult {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [permission, setPermission] = useState<CameraPermissionState>("idle");
  const [isActive, setIsActive] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
  }, []);

  const startCamera = useCallback(async () => {
    setErrorMessage(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setPermission("unavailable");
      setErrorMessage(
        "This browser doesn't support camera access. Try the latest Chrome, Edge, or Safari."
      );
      return;
    }

    setPermission("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {
          // Autoplay can be blocked in some browsers until the user
          // interacts further; the stream is still live.
        });
      }
      setPermission("granted");
      setIsActive(true);
    } catch (err) {
      const domError = err as DOMException;
      if (
        domError.name === "NotAllowedError" ||
        domError.name === "PermissionDeniedError"
      ) {
        setPermission("denied");
        setErrorMessage(
          "Camera access was denied. Allow camera permission in your browser settings to use live interpretation."
        );
      } else if (
        domError.name === "NotFoundError" ||
        domError.name === "DevicesNotFoundError"
      ) {
        setPermission("unavailable");
        setErrorMessage(
          "No camera was found on this device. Connect a camera or try uploading a video instead."
        );
      } else {
        setPermission("error");
        setErrorMessage(
          "Something went wrong starting the camera. Please try again."
        );
      }
      setIsActive(false);
    }
  }, []);

  // Always release the camera when the component unmounts.
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  return { videoRef, permission, isActive, errorMessage, startCamera, stopCamera };
}
