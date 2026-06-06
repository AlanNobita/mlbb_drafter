import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Recommendation } from "@/lib/draft/types";
import { HeroTile } from "./HeroTile";
import { RadialScore } from "./RadialScore";
import { Swords, HandHeart, MapPin, ShieldAlert, Check } from "lucide-react";
import { cn } from "@/lib/utils";

function BreakdownBar({ label, value, tint }: { label: string; value: number; tint: string }) {
  const pct = Math.max(0, Math.min(100, Math.abs(value)));
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-muted-foreground">
        <span>{label}</span>
        <span className="font-semibold text-foreground">+{value.toFixed(1)}</span>
      </div>
      <div className="h-1 overflow-hidden rounded bg-border">
        <div className={`h-full ${tint}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function HeroPill({
  name,
  team,
  tone = "team",
}: {
  name: string;
  team: "ally" | "enemy";
  tone?: "team" | "warning";
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-1.5 rounded-md border px-1.5 py-1",
        tone === "warning"
          ? "border-amber-500/50 bg-amber-500/15"
          : "border-border/60 bg-surface/60",
      )}
    >
      <HeroTile name={name} team={team} size="sm" />
      <span
        className={cn(
          "max-w-[80px] truncate text-[11px] font-semibold",
          tone === "warning"
            ? "text-amber-200"
            : team === "ally"
              ? "text-ally"
              : "text-enemy",
        )}
      >
        {name}
      </span>
    </div>
  );
}

export function RecommendationRow({ rec }: { rec: Recommendation }) {
  const hasCounters = rec.counters.length > 0;
  const hasSynergies = rec.synergies.length > 0;
  const hasAnyContext = hasCounters || hasSynergies || rec.countered_by.length > 0;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="flex items-center gap-2.5 rounded-md border border-border/70 bg-surface-2/60 p-2 transition hover:border-recommend/40 hover:bg-surface-2">
          <HeroTile name={rec.heroName} team="recommend" size="sm" />
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-semibold text-foreground">{rec.heroName}</p>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
              {rec.fallback ? "Fallback pick" : "Top recommendation"}
            </p>
          </div>
          <RadialScore value={rec.composite_score} />
        </div>
      </TooltipTrigger>
      <TooltipContent
        side="left"
        sideOffset={8}
        className="w-72 max-w-[90vw] space-y-3 border border-border/80 bg-popover p-3 text-popover-foreground shadow-2xl"
      >
        <div className="flex items-center justify-between border-b border-border/60 pb-2">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
              Why this pick
            </p>
            <p className="text-sm font-bold text-foreground">{rec.heroName}</p>
          </div>
          <div className="rounded-md border border-recommend/40 bg-recommend/10 px-2 py-1">
            <span className="text-base font-bold text-recommend">
              {rec.composite_score.toFixed(2)}
            </span>
          </div>
        </div>

        {hasAnyContext && (
          <div className="space-y-2">
            {hasCounters && (
              <div>
                <p className="mb-1 flex items-center gap-1 text-[10px] font-bold uppercase tracking-widest text-enemy">
                  <Swords className="h-3 w-3" />
                  Counters ({rec.counters.length})
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {rec.counters.map((name) => (
                    <HeroPill key={name} name={name} team="enemy" />
                  ))}
                </div>
              </div>
            )}
            <div>
              <p className="mb-1 flex items-center gap-1 text-[10px] font-bold uppercase tracking-widest text-amber-400">
                <ShieldAlert className="h-3 w-3" />
                Countered by ({rec.countered_by.length})
              </p>
              {rec.countered_by.length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {rec.countered_by.map((name) => (
                    <HeroPill key={name} name={name} team="enemy" tone="warning" />
                  ))}
                </div>
              ) : (
                <p className="flex items-center gap-1 text-[11px] text-muted-foreground">
                  <Check className="h-3 w-3 text-emerald-400" />
                  No enemy counters
                </p>
              )}
            </div>
            {hasSynergies && (
              <div>
                <p className="mb-1 flex items-center gap-1 text-[10px] font-bold uppercase tracking-widest text-ally">
                  <HandHeart className="h-3 w-3" />
                  Synergizes with ({rec.synergies.length})
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {rec.synergies.map((name) => (
                    <HeroPill key={name} name={name} team="ally" />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {!hasAnyContext && (
          <p className="text-[11px] italic text-muted-foreground">
            Add enemy or ally picks to see why this hero fits.
          </p>
        )}

        <div className="space-y-1.5 border-t border-border/60 pt-2">
          <p className="flex items-center gap-1 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
            <MapPin className="h-3 w-3" />
            Score breakdown
          </p>
          <BreakdownBar label="Synergy" value={rec.breakdown.synergy} tint="bg-ally" />
          <BreakdownBar label="Counter" value={rec.breakdown.counter} tint="bg-enemy" />
          <BreakdownBar label="Lane Match" value={rec.breakdown.laneMatch} tint="bg-recommend" />
        </div>
      </TooltipContent>
    </Tooltip>
  );
}
