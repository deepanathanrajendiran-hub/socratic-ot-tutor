// frontend/components/architecture/StateMachineDiagram.tsx
"use client";
import { useCallback, useMemo, useState } from "react";
import {
  ReactFlow, Background, Controls, MiniMap,
  Position, type Node, type Edge, type NodeProps,
  Handle,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

/* ── Node metadata ─────────────────────────────────────────────────────────
 * Each node maps 1:1 to a Python file in backend/graph/nodes/.
 * The `detail` field is shown in the side panel when a node is hovered/clicked.
 * Keep these descriptions tight — the side panel is small.
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
    oneLiner: "Casual opener for chitchat — no teaching yet",
    detail:
      "When the student writes 'hi' or 'hey' with no concept named, the graph routes here. Its job is to greet, optionally ask about NBCOT prep, and elicit a concept the student wants to study. Once a concept lands in current_concept, the next turn flows into the Socratic loop.",
  },
  manager_agent: {
    label: "manager_agent",
    file: "graph/nodes/manager_agent.py",
    kind: "manager",
    oneLiner: "Concept extraction + locking",
    detail:
      "Runs every turn. Reads the latest student message and the locked concept and decides whether to keep the concept (most turns) or switch to a new topic (when the student says 'what about the rotator cuff?'). The locked-concept design is what prevents mid-loop concept drift.",
  },
  retrieval_node: {
    label: "retrieval_node",
    file: "graph/nodes/retrieval_node.py",
    kind: "retrieval",
    oneLiner: "Corrective RAG with cross-encoder rerank",
    detail:
      "Wraps corrective_retrieve(): synonym-expand the query, embed it, search the domain's ChromaDB collection, run cross-encoder reranking, and gate on a confidence threshold. Per-concept cache skips rerunning on follow-up turns of the same concept (saves 2-7s per turn).",
  },
  response_classifier: {
    label: "response_classifier",
    file: "graph/nodes/response_classifier.py",
    kind: "classifier",
    oneLiner: "Labels student message: correct / incorrect / idk / questioning / irrelevant",
    detail:
      "Haiku-based classifier with three guards: rule-based IDK regex, misspelling promoter (difflib ≥0.82), and a topic-announcement override. The label drives routing — 'correct' goes to step_advancer, 'incorrect' to hint_error_node, 'idk' increments a counter that gates reveal.",
  },
  teacher_socratic: {
    label: "teacher_socratic",
    file: "graph/nodes/teacher_socratic.py",
    kind: "generator",
    oneLiner: "Turn-0 Socratic opener",
    detail:
      "Generates the very first 'what do you already know about X?' question. Pulls the retrieved chunks for grounding but never names the concept. Streams tokens to the frontend.",
  },
  hint_error_node: {
    label: "hint_error_node",
    file: "graph/nodes/hint_error_node.py",
    kind: "generator",
    oneLiner: "Acknowledges the wrong answer, hints toward the correct one",
    detail:
      "Two responsibilities: validate what the student said (don't dismiss the partial truth) and hint at the missing piece without revealing the concept name. Hints get progressively more specific each turn — at turn 1 it stays broad, at turn 2 it points at the specific structural feature.",
  },
  teach_node: {
    label: "teach_node",
    file: "graph/nodes/teach_node.py",
    kind: "generator",
    oneLiner: "Reveal node — fires after SOCRATIC_TURN_GATE or IDK threshold",
    detail:
      "Two trigger conditions: turn ≥3 with student_attempted=True, or 2 consecutive 'idk' messages. Reveals the concept name with a grounded textbook explanation (~3 sentences) and offers the A/B/C mastery menu.",
  },
  step_advancer: {
    label: "step_advancer",
    file: "graph/nodes/step_advancer.py",
    kind: "generator",
    oneLiner: "Mastery confirmer — fires when classifier says 'correct'",
    detail:
      "Sets mastery_level (strong if turn<3, weak otherwise), drops the concept from weak_topics, and offers the A/B/C menu: clinical question, next topic, or stop. Pure Python — Dean still validates the draft.",
  },
  dean_node: {
    label: "dean_node",
    file: "graph/nodes/dean_node.py",
    kind: "gate",
    oneLiner: "Quality gate — leak / format / grounding checks",
    detail:
      "Every generated draft passes through Dean before delivery. Catches concept-name leaks (Python pre-check + LLM rubric), format violations (mastery menu missing items, wrong tone), and grounding issues. Up to 2 revision loops back to the source generator before falling through to fallback_scaffold.",
  },
  deliver_response: {
    label: "deliver_response",
    file: "graph/_stream.py",
    kind: "exit",
    oneLiner: "Final delivery + state persistence",
    detail:
      "Writes the approved draft to the message stream, updates checkpoint via PostgresSaver (Cloud SQL), and emits the SSE 'response' event the frontend renders.",
  },
};

const KIND_STYLES: Record<NodeKind, { bg: string; border: string; text: string }> = {
  entry:      { bg: "bg-amber-50",   border: "border-amber-300",   text: "text-amber-900" },
  manager:    { bg: "bg-violet-50",  border: "border-violet-300",  text: "text-violet-900" },
  retrieval:  { bg: "bg-sky-50",     border: "border-sky-300",     text: "text-sky-900" },
  classifier: { bg: "bg-teal-50",    border: "border-teal-300",    text: "text-teal-900" },
  generator:  { bg: "bg-emerald-50", border: "border-emerald-300", text: "text-emerald-900" },
  gate:       { bg: "bg-rose-50",    border: "border-rose-300",    text: "text-rose-900" },
  exit:       { bg: "bg-slate-50",   border: "border-slate-300",   text: "text-slate-900" },
};

/* ── Custom node component ────────────────────────────────────────────── */

function GraphNode({ data, selected }: NodeProps<Node<NodeData>>) {
  const s = KIND_STYLES[data.kind];
  return (
    <div
      className={`rounded-lg border-2 px-3 py-2 shadow-sm transition-all ${
        s.bg
      } ${s.border} ${selected ? "ring-2 ring-slate-900 ring-offset-2" : ""}`}
      style={{ minWidth: 160 }}
    >
      <Handle type="target" position={Position.Top} className="!bg-slate-400" />
      <div className={`text-sm font-semibold ${s.text}`}>{data.label}</div>
      <div className={`mt-0.5 text-[11px] leading-tight ${s.text} opacity-70`}>
        {data.oneLiner}
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-slate-400" />
    </div>
  );
}

const NODE_TYPES = { graph: GraphNode };

/* ── Graph layout ──────────────────────────────────────────────────────── */

const NODES: Node<NodeData>[] = [
  { id: "rapport_node",        type: "graph", position: { x:   0, y:   0 }, data: NODES_DATA.rapport_node },
  { id: "manager_agent",       type: "graph", position: { x: 240, y:   0 }, data: NODES_DATA.manager_agent },
  { id: "retrieval_node",      type: "graph", position: { x: 240, y: 110 }, data: NODES_DATA.retrieval_node },
  { id: "response_classifier", type: "graph", position: { x: 240, y: 220 }, data: NODES_DATA.response_classifier },
  { id: "teacher_socratic",    type: "graph", position: { x:   0, y: 360 }, data: NODES_DATA.teacher_socratic },
  { id: "hint_error_node",     type: "graph", position: { x: 240, y: 360 }, data: NODES_DATA.hint_error_node },
  { id: "teach_node",          type: "graph", position: { x: 480, y: 360 }, data: NODES_DATA.teach_node },
  { id: "step_advancer",       type: "graph", position: { x: 720, y: 360 }, data: NODES_DATA.step_advancer },
  { id: "dean_node",           type: "graph", position: { x: 360, y: 500 }, data: NODES_DATA.dean_node },
  { id: "deliver_response",    type: "graph", position: { x: 360, y: 620 }, data: NODES_DATA.deliver_response },
];

const EDGES: Edge[] = [
  { id: "e-rapport-manager",  source: "rapport_node",        target: "manager_agent",        animated: true,  label: "concept named" },
  { id: "e-mgr-ret",          source: "manager_agent",       target: "retrieval_node" },
  { id: "e-ret-cls",          source: "retrieval_node",      target: "response_classifier" },
  { id: "e-cls-soc",          source: "response_classifier", target: "teacher_socratic",     label: "questioning + turn 0",
    style: { stroke: "#0ea5e9" } },
  { id: "e-cls-hint",         source: "response_classifier", target: "hint_error_node",      label: "incorrect / idk",
    style: { stroke: "#f59e0b" } },
  { id: "e-cls-teach",        source: "response_classifier", target: "teach_node",           label: "turn ≥ 3 OR 2× idk",
    style: { stroke: "#dc2626" } },
  { id: "e-cls-adv",          source: "response_classifier", target: "step_advancer",        label: "correct",
    style: { stroke: "#10b981" } },
  { id: "e-soc-dean",         source: "teacher_socratic",    target: "dean_node" },
  { id: "e-hint-dean",        source: "hint_error_node",     target: "dean_node" },
  { id: "e-teach-dean",       source: "teach_node",          target: "dean_node" },
  { id: "e-adv-dean",         source: "step_advancer",       target: "dean_node" },
  { id: "e-dean-deliver",     source: "dean_node",           target: "deliver_response",     label: "passed",
    style: { stroke: "#10b981" } },
];

/* ── Side panel ───────────────────────────────────────────────────────── */

function SidePanel({ data }: { data: NodeData | null }) {
  if (!data) {
    return (
      <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-500">
        Hover or click a node to see its purpose, source file, and code excerpt.
      </div>
    );
  }
  const s = KIND_STYLES[data.kind];
  return (
    <div className={`rounded-lg border ${s.border} ${s.bg} p-4`}>
      <div className={`text-base font-semibold ${s.text}`}>{data.label}</div>
      <div className="mt-1 font-mono text-[11px] text-slate-600">{data.file}</div>
      <div className={`mt-3 text-sm font-medium ${s.text}`}>{data.oneLiner}</div>
      <p className="mt-3 text-sm leading-relaxed text-slate-700">{data.detail}</p>
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
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
      <div className="h-[640px] rounded-lg border border-slate-200 bg-slate-50">
        <ReactFlow
          nodes={NODES}
          edges={EDGES}
          nodeTypes={NODE_TYPES}
          onNodeMouseEnter={onNodeMouseEnter}
          onNodeMouseLeave={onNodeMouseLeave}
          onNodeClick={onNodeClick}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={16} size={1} color="#cbd5e1" />
          <Controls showInteractive={false} />
          <MiniMap pannable zoomable className="!bg-white" />
        </ReactFlow>
      </div>
      <div className="lg:sticky lg:top-4 lg:self-start">
        <SidePanel data={activeData} />
      </div>
    </div>
  );
}
