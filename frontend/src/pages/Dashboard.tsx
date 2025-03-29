import { useState, useEffect } from "react";
import { useHealth } from "../hooks/useHealth";
import StatBadge from "../components/StatBadge";
import WorkerCard from "../components/WorkerCard";
import CacheChart from "../components/CacheChart";

interface ChartPoint {
  t: string;
  hits: number;
  trie: number;
}

function GatewayFlag({ label, active }: { label: string; active: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full font-medium ${
        active ? "bg-emerald-900 text-emerald-300" : "bg-gray-800 text-gray-500"
      }`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${active ? "bg-emerald-400" : "bg-gray-600"}`} />
      {label}
    </span>
  );
}

export default function Dashboard() {
  const { data, isError, isLoading } = useHealth();
  const [history, setHistory] = useState<ChartPoint[]>([]);

  useEffect(() => {
    if (!data) return;
    const t = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    setHistory((prev) => [...prev.slice(-29), { t, hits: data.cache_hits_total, trie: data.trie_entries }]);
  }, [data]);

  if (isLoading) {
    return <p className="text-gray-500 mt-8 text-center">Connecting to gateway…</p>;
  }

  if (isError || !data) {
    return (
      <div className="mt-8 text-center">
        <p className="text-red-400 font-medium">Gateway unreachable</p>
        <p className="text-gray-500 text-sm mt-1">Make sure InferMesh is running on port 8000.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 flex-wrap">
        <h1 className="text-xl font-semibold">Gateway Overview</h1>
        <span
          className={`text-xs px-2 py-0.5 rounded-full font-medium ${
            data.status === "ok" ? "bg-emerald-900 text-emerald-300" : "bg-red-900 text-red-300"
          }`}
        >
          {data.status}
        </span>
        <GatewayFlag
          label={data.redis_enabled && data.redis_transport ? `Redis (${data.redis_transport})` : "Redis"}
          active={data.redis_enabled}
        />
        <GatewayFlag label="Disaggregation" active={data.disaggregation_enabled} />
        <GatewayFlag label="Prometheus" active={data.prometheus_enabled} />
        <GatewayFlag label="OTEL" active={data.otel_enabled} />
        {data.prometheus_enabled && (
          <a
            href="/metrics"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-gray-500 hover:text-gray-300 underline underline-offset-2"
          >
            /metrics ↗
          </a>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <StatBadge label="Workers" value={data.workers} />
        <StatBadge label="Trie entries" value={data.trie_entries} />
        <StatBadge label="Cache hits" value={data.cache_hits_total} />
        <StatBadge
          label="Avg GPU cache"
          value={(() => {
            const vllm = data.worker_loads.filter((w) => w.type === "vllm");
            return vllm.length
              ? `${Math.round((vllm.reduce((s, w) => s + w.cache_utilization, 0) / vllm.length) * 100)}%`
              : "—";
          })()}
        />
        {data.active_sessions !== undefined && (
          <StatBadge label="Active sessions" value={data.active_sessions} />
        )}
        {data.session_counters && (() => {
          const sc = data.session_counters!;
          const total = sc.sticky_hits_total + sc.sticky_misses_total;
          const hitRate = total > 0 ? Math.round((sc.sticky_hits_total / total) * 100) : 0;
          return (
            <>
              <StatBadge label="Sticky hit rate" value={`${hitRate}%`} />
              <StatBadge label="Session evictions" value={sc.session_eviction_total} />
            </>
          );
        })()}
      </div>

      {data.session_counters && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <h2 className="text-sm font-medium text-gray-400 mb-3 uppercase tracking-wider">Disaggregation</h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-6 gap-y-2 text-sm">
            {[
              ["Prefill complete", data.session_counters.prefill_complete_received_total],
              ["Sticky hits", data.session_counters.sticky_hits_total],
              ["Sticky misses", data.session_counters.sticky_misses_total],
              ["Evictions", data.session_counters.session_eviction_total],
            ].map(([label, val]) => (
              <div key={label as string}>
                <p className="text-gray-500 text-xs">{label}</p>
                <p className="text-white font-mono font-medium">{val}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.redis_events && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="flex items-baseline justify-between mb-3">
            <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wider">
              Redis subscriber
            </h2>
            <span className="text-xs text-gray-500 font-mono">
              {data.redis_transport ?? "—"}
            </span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-6 gap-y-2 text-sm mb-3">
            <div>
              <p className="text-gray-500 text-xs">Received</p>
              <p className="text-white font-mono font-medium">
                {data.redis_events.received_total}
              </p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Dropped</p>
              <p
                className={`font-mono font-medium ${
                  data.redis_events.dropped_total > 0 ? "text-yellow-400" : "text-white"
                }`}
              >
                {data.redis_events.dropped_total}
              </p>
            </div>
            {Object.entries(data.redis_events.by_type).map(([type, count]) => (
              <div key={`type-${type}`}>
                <p className="text-gray-500 text-xs font-mono">{type}</p>
                <p className="text-white font-mono font-medium">{count}</p>
              </div>
            ))}
          </div>
          {Object.keys(data.redis_events.drops_by_reason).length > 0 && (
            <div className="flex gap-3 text-xs flex-wrap pt-3 border-t border-gray-800">
              <span className="text-gray-500">Drops:</span>
              {Object.entries(data.redis_events.drops_by_reason).map(([reason, count]) => (
                <span key={`drop-${reason}`} className="text-yellow-400 font-mono">
                  {reason}={count}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      <CacheChart data={history} />

      <div>
        <h2 className="text-sm font-medium text-gray-400 mb-3 uppercase tracking-wider">Workers</h2>
        {data.worker_loads.length === 0 ? (
          <p className="text-gray-600 text-sm">No workers registered.</p>
        ) : (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {data.worker_loads.map((w) => (
              <WorkerCard key={w.id} worker={w} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
