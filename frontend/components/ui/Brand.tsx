// frontend/components/ui/Brand.tsx
//
// Socratic-OT brand mark + wordmark.
//
// Concept: Synapse-as-dialogue. Two cell bodies, a coral neurotransmitter
// spark crossing the cleft. The mark reads first as two minds in
// conversation, second as a synapse — exactly the metaphor the product
// embodies (the tutor doesn't hand the answer over; it sets up a question
// and waits for the student's own circuit to fire).
//
// Ported from the Claude Design handoff bundle (Variant 01 in marks.jsx).
// Coral is reserved for the spark — never the cell bodies, never the
// wordmark, never UI chrome beyond a single accent. Same meaning every
// time: this is the moment of insight.

import * as React from "react";

const INK   = "#1f1d1b";
const CORAL = "#cc785c";

export interface MarkProps {
  /** Pixel size for both width and height. */
  size?:        number;
  /** Override the primary stroke / fill color. */
  ink?:         string;
  /** Override the accent color (the spark). */
  coral?:       string;
  /** When true, render single-color (ink only). For monochrome contexts. */
  mono?:        boolean;
  /** Force the simplified 16-grid variant (favicon-style). When omitted,
   *  picks automatically based on `size` — small sizes use the simplified
   *  geometry so receptor caps and the dendritic cap don't alias. */
  simplified?:  boolean;
  /** Optional aria-label override. */
  label?:       string;
  className?:   string;
}

export function MarkSynapse({
  size = 28,
  ink = INK,
  coral = CORAL,
  mono = false,
  simplified,
  label = "Socratic-OT",
  className,
}: MarkProps) {
  const c = mono ? ink : coral;
  // Auto-pick the simplified variant below 24px — receptor caps + bouton
  // are too thin to read once the SVG drops below ~24×24.
  const useSimplified = simplified ?? size < 24;

  if (useSimplified) {
    return (
      <svg
        width={size} height={size}
        viewBox="0 0 16 16"
        aria-label={label}
        role="img"
        className={className}
      >
        {/* presynaptic body */}
        <circle cx="3.5"  cy="8" r="2.5" fill={ink} />
        {/* postsynaptic body */}
        <circle cx="12.5" cy="8" r="2.5" fill={ink} />
        {/* single coral spark across the cleft */}
        <circle cx="8"    cy="8" r="1.25" fill={c} />
      </svg>
    );
  }

  return (
    <svg
      width={size} height={size}
      viewBox="0 0 64 64"
      aria-label={label}
      role="img"
      className={className}
    >
      {/* PRESYNAPTIC NEURON (left) — cell body + axon stub */}
      <g fill={ink}>
        <circle cx="13" cy="32" r="9" />
        <path d="M22 32 H24" stroke={ink} strokeWidth="2.4" strokeLinecap="round" />
      </g>
      {/* presynaptic bouton (terminal swell) */}
      <circle cx="25.5" cy="32" r="2.6" fill={ink} />

      {/* POSTSYNAPTIC NEURON (right) — cell body + receptor cap */}
      <g fill={ink}>
        <circle cx="51" cy="32" r="9" />
        <path d="M42 32 H40" stroke={ink} strokeWidth="2.4" strokeLinecap="round" />
      </g>
      {/* postsynaptic dendritic receptor cap */}
      <g stroke={ink} strokeWidth="1.6" strokeLinecap="round" fill="none">
        <path d="M38.5 28.5 L36.8 27.2" />
        <path d="M38.5 35.5 L36.8 36.8" />
      </g>

      {/* SYNAPTIC CLEFT — three neurotransmitter vesicles crossing */}
      <g fill={c}>
        <circle cx="29.5" cy="32" r="1.5" />
        <circle cx="33"   cy="32" r="1.9" />
        <circle cx="36.7" cy="32" r="1.4" opacity="0.85" />
      </g>
    </svg>
  );
}

/** Wordmark — "Socratic·OT" with a coral middle dot, set in the brand
 *  serif (Source Serif 4 → ui-serif fallback). Use `size` for the
 *  font-size; everything else (weight, tracking, line-height) is locked
 *  to keep the wordmark consistent across the site. */
export function Wordmark({
  size = 18,
  ink = INK,
  coral = CORAL,
  mono = false,
  className,
}: {
  size?:  number;
  ink?:   string;
  coral?: string;
  mono?:  boolean;
  className?: string;
}) {
  const c = mono ? ink : coral;
  return (
    <span
      className={`font-serif ${className ?? ""}`}
      style={{
        fontWeight:    500,
        fontSize:      size,
        color:         ink,
        letterSpacing: "-0.015em",
        lineHeight:    1,
        whiteSpace:    "nowrap",
      }}
    >
      Socratic
      <span style={{ color: c, padding: "0 0.08em" }}>·</span>
      OT
    </span>
  );
}

/** Combined lockup: mark + wordmark, horizontal. */
export function BrandLockup({
  markSize = 28,
  wordmarkSize = 18,
  mono = false,
  className,
}: {
  markSize?:     number;
  wordmarkSize?: number;
  mono?:         boolean;
  className?:    string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-2.5 ${className ?? ""}`}
      aria-label="Socratic-OT"
    >
      <MarkSynapse size={markSize} mono={mono} label="" />
      <Wordmark size={wordmarkSize} mono={mono} />
    </span>
  );
}
