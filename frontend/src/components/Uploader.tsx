import { useCallback, useRef, useState } from "react";

interface Props {
  accepts: string[];
  maxUploadMb: number;
  disabled: boolean;
  onPick: (file: File) => void;
}

/** Drag a capture in, or click to choose one. */
export function Uploader({ accepts, maxUploadMb, disabled, onPick }: Props) {
  const [over, setOver] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const take = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      const dot = file.name.lastIndexOf(".");
      const suffix = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
      if (!accepts.includes(suffix)) {
        setProblem(`${suffix || file.name} is not a capture. Accepted: ${accepts.join(", ")}`);
        return;
      }
      if (file.size > maxUploadMb * 1024 * 1024) {
        const mb = (file.size / 1024 / 1024).toFixed(0);
        setProblem(`That file is ${mb} MB; the limit is ${maxUploadMb} MB.`);
        return;
      }
      setProblem(null);
      onPick(file);
    },
    [accepts, maxUploadMb, onPick],
  );

  return (
    <div>
      <div
        className={`dropzone${over ? " over" : ""}${disabled ? " disabled" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          if (!disabled) take(e.dataTransfer.files?.[0]);
        }}
        onClick={() => !disabled && input.current?.click()}
        onKeyDown={(e) => {
          if (!disabled && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            input.current?.click();
          }
        }}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Choose a capture to check"
        aria-disabled={disabled}
      >
        <svg className="dropicon" viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <p className="dropmain">Drop a capture here, or click to choose</p>
        <p className="dropsub">A video, or a .zip of photos</p>
        <input
          ref={input}
          type="file"
          accept={accepts.join(",")}
          hidden
          onChange={(e) => {
            take(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>
      {problem && (
        <p className="inline-error" role="alert">
          {problem}
        </p>
      )}
    </div>
  );
}
