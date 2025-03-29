import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface Point {
  t: string;
  hits: number;
  trie: number;
}

interface Props {
  data: Point[];
}

export default function CacheChart({ data }: Props) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <p className="text-sm font-medium text-gray-300 mb-4">Cache hits &amp; trie entries over time</p>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="t" tick={{ fontSize: 10, fill: "#6b7280" }} />
          <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
          <Tooltip
            contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", borderRadius: 8 }}
            labelStyle={{ color: "#e5e7eb", fontSize: 11 }}
            itemStyle={{ fontSize: 11 }}
          />
          <Line type="monotone" dataKey="hits" stroke="#6366f1" strokeWidth={2} dot={false} name="Cache hits" />
          <Line type="monotone" dataKey="trie" stroke="#10b981" strokeWidth={2} dot={false} name="Trie entries" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
