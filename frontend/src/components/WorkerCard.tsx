import type { WorkerLoad } from "../api/client";

interface Props {
  worker: WorkerLoad;
}

const ROLE_STYLES: Record<WorkerLoad["role"], string> = {
  prefill: "bg-violet-900 text-violet-300",
  decode: "bg-sky-900 text-sky-300",
  mixed: "bg-gray-800 text-gray-400",
};

const CIRCUIT_STYLES: Record<WorkerLoad["circuit_state"], string> = {
  CLOSED: "hidden",
  HALF_OPEN: "bg-yellow-900 text-yellow-300",
  OPEN: "bg-red-900 text-red-300",
};

export default function WorkerCard({ worker }: Props) {
  const utilPct = Math.round(worker.cache_utilization * 100);
  const barColor =
    utilPct > 80 ? "bg-red-500" : utilPct > 50 ? "bg-yellow-500" : "bg-emerald-500";
  const isVllm = worker.type === "vllm";
  const circuitStyle = CIRCUIT_STYLES[worker.circuit_state ?? "CLOSED"];

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <p className="font-mono text-sm text-white font-medium">{worker.id}</p>
            <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${ROLE_STYLES[worker.role ?? "mixed"]}`}>
              {worker.role ?? "mixed"}
            </span>
            <span className="text-xs px-1.5 py-0.5 rounded font-medium bg-gray-800 text-gray-500">
              {worker.type}
            </span>
            {circuitStyle !== "hidden" && (
              <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${circuitStyle}`}>
                {worker.circuit_state}
              </span>
            )}
          </div>
          <p className="text-xs text-gray-400 mt-0.5">{worker.model}</p>
        </div>
        <span className="text-xs bg-gray-800 text-gray-300 px-2 py-0.5 rounded-full whitespace-nowrap">
          {worker.tokens_in_flight} tok in-flight
        </span>
      </div>

      {isVllm ? (
        <div>
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>GPU cache</span>
            <span>{utilPct}%</span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${barColor}`}
              style={{ width: `${utilPct}%` }}
            />
          </div>
        </div>
      ) : (
        <p className="text-xs text-gray-600">GPU cache not available for OpenAI workers</p>
      )}
    </div>
  );
}
