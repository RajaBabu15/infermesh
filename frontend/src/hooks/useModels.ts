import { useQuery } from "@tanstack/react-query";
import { fetchModels } from "../api/client";

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: fetchModels,
    staleTime: 60_000,
  });
}
