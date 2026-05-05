// frontend/components/architecture/StateMachineDiagram.tsx
"use client";
import { useCallback, useMemo, useState } from "react";
import {
  ReactFlow, Background, Controls, MiniMap, MarkerType,
  Position, type Node, type Edge, type NodeProps,
  Handle,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

/* ── Node metadata ─────────────────────────────────────────────────────────
 * Each node maps 1:1 to a Python file in backend/graph/nodes/.
 * The `detail` field is shown in the side panel when a node is hovered/clicked.
 */

type NodeKind = "entry" | "manager" | "retrieval" | "classifier"
              | "generator" | "gate" | "exit";

type NodeData = {
  label: string;
  file: string;
  kind: NodeKind;
  oneLiner: string;
  detail: string;
};

const NODES_DATA: Record<string, NodeData> = {
  rapport_node: {
    label: "rapport_node",
    file: "graph/nodes/rapport_node.py",
    kind: "entry",
    oneLiner: "Casual opener — chitchat, no teaching",
    detail:
      "When the student writes 'hi' with no concept named, the graph routes here. Its job is to greet, ask about NBCOT prep, and elicit a concept the student wants to study. Once a concept lands in current_concept, the next turn flows into the Socratic loop.",
  },
  manager_agent: {
    label: "manager_agent",
    file: "graph/nodes/manager_agent.py",
    kind: "manager",
    oneLiner: "Concept extraction + locking",
    detail:
      "Runs every turn. Reads the latest student message and the locked concept and decides whether to keep the concept (most turns) or switch (when the student says 'what about the rotator cuff?'). The locked-concept design prevents mid-loop concept drift.",
  },
  retrieval_node: {
    label: "retrieval_node",
    file: "graph/nodes/retrieval_node.py",
    kind: "retrieval",
    oneLiner: "Corrective RAG + cross-encoder rerank",
    detail:
      "Wraps corrective_retrieve(): synonym-expand, embed, ChromaDB cosine top-15, cross-encoder rerank, confidence threshold. Per-concept cache skips rerunning on follow-up turns of the same concept (saves 2-7s per turn).",
  },
  response_classifier: {
    label: "response_classifier",
    file: "graph/nodes/response_classifier.py",
    kind: "classifier",
    oneLiner: "Labels: correct / incorrect / idk / questioning",
    detail:
      "Haiku classifier with three guards: rule-based IDK regex, misspelling promoter (difflib ≥0.82), and a topic-announcement override. The label drives routing — 'correct' goes to step_advancer, 'incorrect' to hint_error_node, 'idk' increments a counter that gates reveal.",
  },
  teacher_socratic: {
    label: "teacher_socratic",
    file: "graph/nodes/teacher_socratic.py",
    kind: "generator",
    oneLiner: "Turn-0 Socratic opener",
    detail:
      "Generates the very first 'what do you already know about X?' question. Pulls retrieved chunks for grounding but never names the concept. Streams tokens to the frontend.",
  },
  hint_error_node: {
    label: "hint_error_node",
    file: "graph/nodes/hint_error_node.py",
    kind: "generator",
    oneLiner: "Acknowledge wrong, hint toward right",
    detail:
      "Validates the partial truth, hints at the missing piece without revealing the concept name. Hints get progressively more specific each turn — at turn 1 broad, at turn 2 pointing at the specific structural feature.",
  },
  teach_node: {
    label: "teach_node",
    file: "graph/nodes/teach_node.py",
    kind: "generator",
    oneLiner: "Reveal at turn-gate or 2× idk",
    detail:
      "Triggers on turn ≥3 with student_attempted=True, or 2 consecutive 'idk' messages. Reveals the concept name with a grounded textbook explanation (~3 sentences) and offers the A/B/C mastery menu.",
  },
  step_advancer: {
    label: "step_advancer",
    file: "graph/nodes/step_advancer.py",
    kind: "generator",
    oneLiner: "Mastery confirmer — 'correct' path",
    detail:
      "Sets mastery_level (strong if turn<3, weak otherwise), drops the concept from weak_topics, offers the A/B/C menu: clinical question, next topic, or stop. Pure Python — Dean still validates the draft.",
  },
  dean_node: {
    label: "dean_node",
    file: "graph/nodes/dean_node.py",
    kind: "gate",
    oneLiner: "Quality gate — leak / format / grounding",
    detail:
      "Every generated draft passes through Dean before delivery. Catches concept-name leaks (Python pre-check + LLM rubric), format violations (mastery menu missing items), grounding issues. Up to 2 revision loops back to the source generator before falling through to fallback_scaffold.",
  },
  deliver_response: {
    label: "deliver_response",
    file: "graph/_stream.py",
    kind: "exit",
    oneLiner: "Final delivery + state persistence",
    detail:
      "Writes the approved draft to the message stream, updates checkpoint via PostgresSaver (Cloud SQL), emits the SSE 'response' event the frontend renders.",
  },
};

/* Vibrant color palette — gradient backgrounds + saturated borders for
 * eye-catching nodes without going cartoonish. Each kind has its own
 * accent so the graph reads at a glance. */
const KIND_STYLES: Record<NodeKind, {
  bg: string;
  border: string;
  text: string;
  glow: string;
  pulseColor: string;
}> = {
  entry:      { bg: "from-amber-100 to-amber-50",      border: "border-amber-400",   text: "text-amber-900",   glow: "shadow-amber-200/50",   pulseColor: "bg-amber-400" },
  manager:    { bg: "from-violet-100 to-violet-50",    border: "border-violet-400",  text: "text-violet-900",  glow: "shadow-violet-200/50",  pulseColor: "bg-violet-400" },
  retrieval:  { bg: "from-sky-100 to-sky-50",          border: "border-sky-400",     text: "text-sky-900",     glow: "shadow-sky-200/50",     pulseColor: "bg-sky-400" },
  classifier: { bg: "from-teal-100 to-teal-50",        border: "border-teal-400",    text: "text-teal-900",    glow: "shadow-teal-200/50",    pulseColor: "bg-teal-400" },
  generator:  { bg: "from-emerald-100 to-emerald-50",  border: "border-emerald-400", text: "text-emerald-900", glow: "shadow-emerald-200/50", pulseColor: "bg-emerald-400" },
  gate:       { bg: "from-rose-100 to-rose-50",        border: "border-rose-400",    text: "text-rose-900",    glow: "shadow-rose-200/50",    pulseColor: "bg-rose-400" },
  exit:       { bg: "from-slate-200 to-slate-100",     border: "border-slate-500",   text: "text-slate-900",   glow: "shadow-slate-300/50",   pulseColor: "bg-slate-500" },
};

/* ── Custom node component ────────────────────────────────────────────── */

function GraphNode({ data, selected }: NodeProps<Node<NodeData>>) {
  const s = KIND_STYLES[data.kind];
  return (
    <div
      className={`group relative rounded-xl border-2 bg-gradient-to-br px-3.5 py-2.5 shadow-md transition-all duration-200 hover:scale-105 hover:shadow-xl ${s.bg} ${s.border} ${s.glow} ${
        selected ? "scale-105 ring-2 ring-slate-900 ring-offset-2 shadow-xl" : ""
      }`}
      style={{ minWidth: 200, maxWidth: 240 }}
    >
      {/* Soft pulsing dot in the corner — different speed per kind */}
      <span className="absolute -right-1.5 -top-1.5 flex h-3 w-3">
        <span
          className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${s.pulseColor}`}
        />
        <span
          className={`relative inline-flex h-3 w-3 rounded-full ${s.pulseColor}`}
        />
      </span>

      <Handle type="target" position={Position.Top} className="!h-2.5 !w-2.5 !bg-slate-500 !border-2 !border-white" />
      <div className={`text-sm font-bold tracking-tight ${s.text}`}>
        {data.label}
      </div>
      <div className={`mt-1 text-[11px] leading-snug ${s.text} opacity-75`}>
        {data.oneLiner}
      </div>
      <Handle type="source" position={Position.Bottom} className="!h-2.5 !w-2.5 !bg-slate-500 !border-2 !border-white" />
    </div>
  );
}

const NODE_TYPES = { graph: GraphNode };

/* ── Graph layout ──────────────────────────────────────────────────────── */

/* Wider horizontal spread for the bottom row of 4 generators so the
 * subtitle text doesn't get clipped, and more vertical room between
 * tiers so the labelled edges don't crash into nodes. */
const NODES: Node<NodeData>[] = [
  { id: "rapport_node",        type: "graph", position: { x:   0,  y:   0  }, data: NODES_DATA.rapport_node },
  { id: "manager_agent",       type: "graph", position: { x: 340,  y:   0  }, data: NODES_DATA.manager_agent },
  { id: "retrieval_node",      type: "graph", position: { x: 340,  y: 140  }, data: NODES_DATA.retrieval_node },
  { id: "response_classifier", type: "graph", position: { x: 340,  y: 280  }, data: NODES_DATA.response_classifier },
  { id: "teacher_socratic",    type: "graph", position: { x: -60,  y: 470  }, data: NODES_DATA.teacher_socratic },
  { id: "hint_error_node",     type: "graph", position: { x: 240,  y: 470  }, data: NODES_DATA.hint_error_node },
  { id: "teach_node",          type: "graph", position: { x: 540,  y: 470  }, data: NODES_DATA.teach_node },
  { id: "step_advancer",       type: "graph", position: { x: 840,  y: 470  }, data: NODES_DATA.step_advancer },
  { id: "dean_node",           type: "graph", position: { x: 410,  y: 660  }, data: NODES_DATA.dean_node },
  { id: "deliver_response",    type: "graph", position: { x: 410,  y: 800  }, data: NODES_DATA.deliver_response },
];

/* All edges animated by default → dashed flowing line effect.
 * The classifier-output edges get distinctive colors for the routing
 * legend, with thicker stroke and arrow markers. */
const COLORED_EDGE = (color: string) => ({
  animated: true,
  type: "smoothstep" as const,
  style: { stroke: color, strokeWidth: 2.5 },
  markerEnd: { type: MarkerType.ArrowClosed, color },
  labelStyle: { fill: color, fontWeight: 600, fontSize: 11 },
  labelBgStyle: { fill: "#ffffff", fillOpacity: 0.9 },
  labelBgPadding: [6, 3] as [number, number],
  labelBgBorderRadius: 4,
});

const NEUTRAL_EDGE = {
  animated: true,
  type: "smoothstep" as const,
  style: { stroke: "#94a3b8", strokeWidth: 2 },
  markerEnd: { type: MarkerType.ArrowClosed, color: "#94a3b8" },
};

const EDGES: Edge[] = [
  { id: "e-rapport-manager",  source: "rapport_node",        target: "manager_agent",        label: "concept named", ...COLORED_EDGE("#a16207") },
  { id: "e-mgr-ret",          source: "manager_agent",       target: "retrieval_node",        ...NEUTRAL_EDGE },
  { id: "e-ret-cls",          source: "retrieval_node",      target: "response_classifier",   ...NEUTRAL_EDGE },
  { id: "e-cls-soc",          source: "response_classifier", target: "teacher_socratic",      label: "questioning + turn 0", ...COLORED_EDGE("#0284c7") },
  { id: "e-cls-hint",         source: "response_classifier", target: "hint_error_node",       label: "incorrect / idk",      ...COLORED_EDGE("#d97706") },
  { id: "e-cls-teach",        source: "response_classifier", target: "teach_node",            label: "turn ≥ 3 OR 2× idk",   ...COLORED_EDGE("#dc2626") },
  { id: "e-cls-adv",          source: "response_classifier", target: "step_advancer",         label: "correct",              ...COLORED_EDGE("#059669") },
  { id: "e-soc-dean",         source: "teacher_socratic",    target: "dean_node",             ...NEUTRAL_EDGE },
  { id: "e-hint-dean",        source: "hint_error_node",     target: "dean_node",             ...NEUTRAL_EDGE },
  { id: "e-teach-dean",       source: "teach_node",          target: "dean_node",             ...NEUTRAL_EDGE },
  { id: "e-adv-dean",         source: "step_advancer",       target: "dean_node",             ...NEUTRAL_EDGE },
  { id: "e-dean-deliver",     source: "dean_node",           target: "deliver_response",      label: "passed",               ...COLORED_EDGE("#059669") },
];

/* ── Side panel ───────────────────────────────────────────────────────── */

function SidePanel({ data }: { data: NodeData | null }) {
  if (!data) {
    return (
      <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 p-5 text-sm text-slate-500">
        <div className="mb-2 font-medium text-slate-700">Interactive diagram</div>
        Hover any node to see its purpose, source file, and what it
        contributes to the Socratic loop. Edges are colored by classifier
        verdict (legend below).
      </div>
    );
  }
  const s = KIND_STYLES[data.kind];
  return (
    <div
      className={`rounded-xl border-2 bg-gradient-to-br p-5 shadow-md transition-all ${s.bg} ${s.border}`}
    >
      <div className={`text-base font-bold ${s.text}`}>{data.label}</div>
      <div className="mt-1 font-mono text-[11px] text-slate-600">{data.file}</div>
      <div className={`mt-3 text-sm font-semibold ${s.text}`}>{data.oneLiner}</div>
      <p className="mt-2 text-sm leading-relaxed text-slate-700">{data.detail}</p>
    </div>
  );
}

/* Edge color legend below the diagram. Helps the reader read the routing. */
function EdgeLegend() {
  const items = [
    { color: "#059669", label: "correct → step_advancer" },
    { color: "#d97706", label: "incorrect / idk → hint" },
    { color: "#dc2626", label: "turn ≥ 3 OR 2× idk → reveal" },
    { color: "#0284c7", label: "questioning + turn 0 → opener" },
    { color: "#94a3b8", label: "neutral pipeline edge" },
  ];
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-slate-600">
      {items.map((i) => (
        <span key={i.color} className="inline-flex items-center gap-2">
          <span
            className="inline-block h-0.5 w-6 rounded-full"
            style={{ backgroundColor: i.color }}
          />
          <span>{i.label}</span>
        </span>
      ))}
    </div>
  );
}

/* ── Main exported component ──────────────────────────────────────────── */

export function StateMachineDiagram() {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [pinnedId, setPinnedId]   = useState<string | null>(null);

  const onNodeMouseEnter = useCallback((_: unknown, node: Node) => {
    setHoveredId(node.id);
  }, []);
  const onNodeMouseLeave = useCallback(() => {
    setHoveredId(null);
  }, []);
  const onNodeClick = useCallback((_: unknown, node: Node) => {
    setPinnedId((prev) => (prev === node.id ? null : node.id));
  }, []);

  const activeId = hoveredId ?? pinnedId;
  const activeData = useMemo(
    () => (activeId ? NODES_DATA[activeId] : null),
    [activeId]
  );

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_340px]">
        <div className="h-[760px] overflow-hidden rounded-xl border border-slate-200 bg-gradient-to-br from-white via-slate-50 to-white shadow-inner">
          <ReactFlow
            nodes={NODES}
            edges={EDGES}
            nodeTypes={NODE_TYPES}
            onNodeMouseEnter={onNodeMouseEnter}
            onNodeMouseLeave={onNodeMouseLeave}
            onNodeClick={onNodeClick}
            fitView
            fitViewOptions={{ padding: 0.18 }}
            proOptions={{ hideAttribution: true }}
            minZoom={0.4}
            maxZoom={1.5}
          >
            <Background gap={18} size={1} color="#cbd5e1" />
            <Controls showInteractive={false} className="!shadow-md" />
            <MiniMap
              pannable
              zoomable
              className="!bg-white"
              nodeColor={(n) => {
                const kind = (n.data as NodeData | undefined)?.kind;
                if (!kind) return "#94a3b8";
                const map: Record<NodeKind, string> = {
                  entry: "#fbbf24", manager: "#a78bfa", retrieval: "#38bdf8",
                  classifier: "#2dd4bf", generator: "#34d399",
                  gate: "#fb7185", exit: "#64748b",
                };
                return map[kind];
              }}
              nodeStrokeWidth={2}
            />
          </ReactFlow>
        </div>
        <div className="lg:sticky lg:top-4 lg:self-start">
          <SidePanel data={activeData} />
        </div>
      </div>
      <div className="rounded-lg border border-slate-200 bg-white p-3">
        <EdgeLegend />
      </div>
    </div>
  );
}
