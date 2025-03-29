import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "../api/client";

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 5000,
    staleTime: 4000,
  });
}
